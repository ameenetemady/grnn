local autograd = require 'autograd'
local myUtil = myUtil or require('../MyCommon/util.lua')
local syngUtil = syngUtil or require('./syngUtil.lua')
local gurobiW = require('./gurobiWrap.lua')

local SyngV7 = {}

do
	local fuSyngV7 = function(teX, teTheta, bias)
		local nD = teX:size(2)
		local nM = teX:size(1)

		local teW = torch.narrow(teTheta, 1,  1, 2*nD+1)

		local teT = torch.narrow(teW, 1, 1, nD+1)
		local teB = torch.narrow(teW, 1, nD+2, nD)
		local teP = torch.narrow(teTheta, 1, teW:size(1)+1, nD)

		local teH = torch.exp(torch.cmul(teX,
																				torch.expand(torch.view(teP, 1, nD), 
																										 nM, nD)))

		local teTop = torch.bmm(torch.view(torch.cat(torch.ones(nM, 1), teH), 1, teH:size(1), teH:size(2)+1),
														torch.view(teT,1, teH:size(2)+1, 1))

		local teBut = torch.add(torch.bmm(torch.view(teH, 1, teH:size(1), teH:size(2)),
																			torch.view(teB, 1, teH:size(2), 1)), 1)

		local teR = torch.view(torch.cdiv(teTop, teBut), nM, 1)

		return teR
	end

	function SyngV7.getMSE(teX, teTheta, teY)
		local teYPred = fuSyngV7(teX, teTheta)
		return torch.dist(teY, teYPred)
	end

  function  SyngV7.new(taWeight)
		local weight = torch.rand(7)*2-1 -- ToDo:  temporary! but wrong
		--local weight = torch.Tensor()
		if taWeight ~= nil and taWeight[1] ~= nil then
			weight = taWeight[1]
		end

    return autograd.nn.AutoModule('SyngV7')(fuSyngV7, weight:clone())
  end

	-- ***  getInitWeights related functions (includes optimization): ***
	
	function SyngV7.getOptimGels(teX, teY, teP)
		-- goal: argMin |Ax-B|
		-- b) construct A
		local nD = teX:size(2)
		local nM = teX:size(1)
		local teH = torch.exp(torch.cmul(teX,
																		torch.expand(torch.view(teP, 1, nD), 
																								 nM, nD)))

		local teYH = torch.cmul(torch.expand(torch.view(teY, nM, 1), nM, nD),
														teH)

		local teA = torch.cat({torch.ones(nM, 1),
													 teH,
													 torch.mul(teYH, -1)})

		-- c) construct B
		local teB = torch.Tensor(nM) -- Note: 2d if real gels used
		teB:copy(teY)
		
--		local teW = torch.gels(teB, teA):squeeze() --ToDo: remove this, it's  old Line when using actual gels
		local nLStart = teA:size(2) - nD
		local status, teW = gurobiW.gelsNonNegative(teA, teB, nLStart, nD)
		if status ~= 2 then
			io.write(string.format(" !!! status:%d !!!! ", status))
		end

		return teW
	end


	function SyngV7.fuLoss(teP, teW, teX, teY)
		local nD = teX:size(2)
		local nM = teX:size(1)
		-- Loss function: |Aw-B| ToDo:consider the original cost function

		-- a) construct A
		local teH = torch.exp(torch.cmul(teX,
																				torch.expand(torch.view(teP, 1, nD), 
																										 nM, nD)))
		local teYH = torch.cmul(torch.expand(torch.view(teY, nM, 1), nM, nD),
														teH)

		local teA = torch.cat({torch.ones(nM, 1),
													 teH,
													 torch.mul(teYH, -1)})

		-- c) construct B
		local teB = torch.Tensor(nM, 1)
		teB:select(2, 1):copy(teY)
		
		local teResBase = torch.add(torch.bmm(torch.view(teA, 1, teA:size(1), teA:size(2)), 
																					torch.view(teW, 1, teW:size(1), 1)),
															 torch.mul(teB, -1))

		local dRes = torch.bmm(torch.transpose(teResBase, 2, 3), teResBase)[1][1][1]

		--adding regularization here:
		--todo: try this with fuLoss2 instead:
		local dLambda= 0.00
		dRes = dRes + dLambda * torch.sum(torch.pow(teP, 2)) 

		return dRes

	end

	function SyngV7.fuForOptim(teX, teY, teInitParam) -- ToDo: test(although minmal code)
		-- a) assume const teInitParam (i.e. p1, p2) and get teW
	 local teW = SyngV7.getOptimGels(teX, teY, teInitParam)

		-- b) assume const teW and calculate grad for p1, p2
		local fuGradLoss = autograd(SyngV7.fuLoss)
		local teGradParams, loss = fuGradLoss(teInitParam, teW, teX, teY)

		return loss, teGradParams, teW
	end

	function SyngV7.getInitWeights(teInputSlice, teTargetSclice, teKOSlice) -- ToDo: test
    local teX, teY = syngUtil.getPresent(teInputSlice, teTargetSclice, teKOSlice)
		local nD = teX:size(2)


		local teWeightOptim = nil
		local fuEval = function(teParam)
			local loss, teGradParams, teCurrWeights = SyngV7.fuForOptim(teX, teY, teParam)
    	teWeightOptim = teCurrWeights:clone()
			return loss, teGradParams
		end

		-- OuterLoop for multiple initializations
		local nMaxRounds = 50
		local teBestTheta = torch.Tensor(nD*3+1)
		local dBestErr = math.huge
      local dGoodEnoughErr = teY:max() * teY:nElement() /50

		for r=1, nMaxRounds do

			local teInitParam =  (torch.rand(nD) - 0.5) 
			local teParamOptim, lossOptim 
			for i=1, 2 do
				teParamOptim, lossOptim = optim.cg(fuEval, teInitParam)
				teInitParam = teParamOptim
			end
			local teCurrTheta = torch.cat(teWeightOptim, teParamOptim, 1)

			local dCurrErr = SyngV7.getMSE(teX, teCurrTheta, teY)
			if dCurrErr < dBestErr then
				io.write(string.format("(%f).", dCurrErr))
				dBestErr = dCurrErr
				teBestTheta:copy(teCurrTheta)
			end

         -- Early Stop Criteria
         if dBestErr < dGoodEnoughErr then
            io.write(" *^* ")
            break
         end

		end

		return { teBestTheta }
	end

	return SyngV7
end

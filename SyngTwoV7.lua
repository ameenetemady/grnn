local autograd = require 'autograd'
local syngUtil = syngUtil or require('./syngUtil.lua')

local SyngTwoV7 = {}

do
	local fuSyngTwoV7 = function(teInput, teWeights, bias)
    local t0 = teWeights[1]; 
    local t1 = teWeights[2]; local t2 = teWeights[4]
    local b1 = teWeights[3]; local b2 = teWeights[5]
    local p1 = teWeights[6]; local p2 = teWeights[7]


    local teX1 = torch.narrow(teInput, 2, 1, 1)
    local teX2 = torch.narrow(teInput, 2, 2, 1)

    local teH1 = torch.exp(torch.mul(teX1, p1))
    local teH2 = torch.exp(torch.mul(teX2, p2))

    local teTop = torch.add(torch.add(torch.mul(teH1, t1), 
                                      torch.mul(teH2, t2)),
                          t0)

    local teBut = torch.add(torch.add(torch.mul(teH1, b1),
                                      torch.mul(teH2, b2)),
                            1)

    return torch.cdiv(teTop,teBut)
	end
  
  function  SyngTwoV7.new(weight)
    weight = weight or torch.rand(7)*2-1

    return autograd.nn.AutoModule('SyngTwoV7')(fuSyngTwoV7, weight:clone())
  end

	-- ***  getInitWeights related functions (includes optimization): ***

	function SyngTwoV7.getOptimGels(teX, teY, teP)
		-- goal: argMin |Ax-B|
		-- b) construct A
		local teX1 = teX:select(2, 1)
		local teX2 = teX:select(2, 2)

		local teH1 = torch.exp(torch.mul(teX1, teP[1]))
		local teH2 = torch.exp(torch.mul(teX2, teP[2]))

		local nRows = torch.size(teY, 1)
		local teA = torch.cat({torch.ones(nRows, 1),
													teH1,
													torch.mul(torch.cmul(teY, teH1), -1),
													teH2,
													torch.mul(torch.cmul(teY, teH2), -1)})

		-- c) construct B
		local teB = torch.Tensor(nRows, 1)
		teB:select(2, 1):copy(teY)

		local teWeights = torch.gels(teB, teA):squeeze()
		local taWeightsEst ={ t0 = teWeights[1], t1 = teWeights[2], t2 = teWeights[4], b1 = teWeights[3], b2 = teWeights[5], p1=teP[1], p2=teP[2] }

		return taWeightsEst
	end

	function SyngTwoV7.fuLoss(myParams1, teW, teX, teY)

			local nRows = torch.size(teY, 1)

			-- Loss function: |Aw-B|
			
			-- a) construct A
			local teX1 = torch.narrow(teX, 2, 1, 1)
			local teX2 = torch.narrow(teX, 2, 2, 1)

			local teH1 = torch.exp(torch.mul(teX1, myParams1[1]))
			local teH2 = torch.exp(torch.mul(teX2, myParams1[2]))

			local teA = torch.cat({torch.ones(nRows, 1),
														teH1,
														torch.mul(torch.cmul(teY, teH1), -1),
														teH2,
														torch.mul(torch.cmul(teY, teH2), -1)})

			-- b) construct B
			local teB = torch.Tensor(nRows, 1)
			torch.select(teB, 2, 1):copy(teY)

			local teResBase = torch.add(torch.bmm(torch.view(teA, 1, teA:size(1), teA:size(2)), 
																						torch.view(teW, 1, teW:size(1), teW:size(2))),
																 torch.mul(teB, -1))

			local dRes = torch.bmm(torch.transpose(teResBase, 2, 3), teResBase)[1][1][1]

			--adding regularization here:
			--todo: try this with fuLoss2 instead:
			local dLambda= 0.005
			dRes = dRes + dLambda * torch.sum(torch.pow(myParams1, 2)) 

			return dRes
		end


	function SyngTwoV7.fuForOptim(teX, teY, teInitParam)
		-- a) assume const teInitParam (i.e. p1, p2) and get teW
	 local taWeights = SyngTwoV7.getOptimGels(teX, teY, teInitParam)
	 local teW = torch.Tensor({{taWeights.t0}, 
														 {taWeights.t1},
														 {taWeights.b1},
														 {taWeights.t2},
														 {taWeights.b2}})
	 
		-- b) assume const teW and calculate grad for p1, p2
		local fuGradLoss = grad(SyngTwoV7.fuLoss)
		local teGradParams, loss = fuGradLoss(teInitParam, teW, teX, teY)

		return loss, teGradParams, taWeights
	end


	function SyngTwoV7.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
    -- filter out the KO records (not useful for training)
    local teX, teY = syngUtil.getPresent(teInputSlice, teTargetSclice, teKOSlice)

  	local teInitParam =  torch.Tensor({-1.5,1.5}) -- torch.rand(2)*2 - 1 
    
		local taWeightOptim = nil
		local fuEval = function(teParam)
			local loss, teGradParams, taCurrWeights = SyngTwoV7.fuForOptim(teX, teY, teParam)
    	taWeightOptim = myUtil.shallowCopy(taCurrWeights)
			return loss, teGradParams
		end

		local teParamOptim, lossOptim = optim.cg(fuEval, teInitParam, {maxIter = 25})


    return torch.Tensor({taWeightOptim.t0, taWeightOptim.t1, taWeightOptim.b1, taWeightOptim.t2, taWeightOptim.b2, 
									 			 teParamOptim[1], teParamOptim[2] })
  end

	return SyngTwoV7
end

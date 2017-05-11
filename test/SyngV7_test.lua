local syngV7 = require('../SyngV7.lua')

function genData1(synthWeight, nSize)
  local teX = torch.rand(nSize, 2)*2
  local mlp = syngV7.new({synthWeight})
  local teY = mlp:forward(teX):reshape(nSize, 1)

  return teX, teY
end

function test1_gels()
  local synthWeight = torch.Tensor({0, 1, 3, 0.5, 0.5, 1, 3})
  local nSize = 50

  local teInput, teTarget = genData1(synthWeight, nSize)

	local teInitP = torch.Tensor({1, 3})
	local teW = syngV7.getOptimGels(teInput, teTarget, teInitP)
	print(teW)
end

function test2_getInitWeights()
	 local synthWeight = torch.Tensor({1, 1, 3, 0.5, 0.5, 1, 3})
  local nSize = 500

  local teInput, teTarget = genData1(synthWeight, nSize)
	local teKOSlice = torch.ones(teInput:size(1), 1)
	local weight = syngV7.getInitWeights(teInput, teTarget, teKOSlice)
	print("weight")
	print(weight[1])

end

--test1_gels()
test2_getInitWeights()

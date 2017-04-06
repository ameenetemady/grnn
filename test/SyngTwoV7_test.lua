require 'nn'
require 'gnuplot'

local myUtil = require('../../MyCommon/util.lua')
local testerPool = testerPool or require('../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../grnnTrainerPool.lua')
local grnnUtil = require('../grnnUtil.lua')
local syngTwoV7 = require('../SyngTwoV7.lua')
require('../FnnAdapter.lua')
local taTrainParams = { nMaxIteration = 200, strOptimMethod = "SGD"}

local SyngTwoV7_test = {}

  torch.manualSeed(3)
function SyngTwoV7_test.test1()

--  local synthWeight = torch.Tensor({0, 1, 3, 0.5, 0.5, 1, 3})
  local synthWeight = torch.rand(7)*2 - 1
  local nSize = 50

  local teInput, teTarget = genData1(synthWeight, nSize)


  --local weight = torch.Tensor({1.5, 0, 0, 0, 0, 2, 2})
  local weight = torch.Tensor({0, 0, 0, 0, 0, 0, 0})
  local mlp = syngTwoV7.new(weight)
	local mNetAdapter = FnnAdapter.new(nil, mlp)

  grnnUtil.logParams(mlp)
	trainerPool.trainGrnnMNetAdapter(mNetAdapter, teInput, teTarget, taTrainParams)

  print("estimated parameters:")
  grnnUtil.logParams(mlp)

  print("expected parameters:")
  print(synthWeight)

	print("MSE error: " .. testerPool.getMSE(mNetAdapter:getRaw(), teInput, teTarget))

end

function SyngTwoV7_test.test2()

  local synthWeight = torch.Tensor({0, 1, 3, 0.5, 0.5, 1, 3})
  local nSize = 50

  local teInput, teTarget = genData1(synthWeight, nSize)


	local teKOSlice = torch.ones(teInput:size(1), 1)
	local weight = syngTwoV7.getInitWeights(teInput, teTarget, teKOSlice)
  local mlp = syngTwoV7.new(weight)
	local mNetAdapter = FnnAdapter.new(nil, mlp)

  grnnUtil.logParams(mlp)

	print("MSE error: " .. testerPool.getMSE(mNetAdapter:getRaw(), teInput, teTarget))

end

function genData1(synthWeight, nSize)
  local teX = torch.rand(nSize, 2)*2
  local mlp = syngTwoV7.new(synthWeight)
  local teY = mlp:forward(teX):reshape(nSize, 1)

  return teX, teY
end

function SyngTwoV7_test.all()
  SyngTwoV7_test.test1()
--  SyngTwoV7_test.test2()
end

SyngTwoV7_test.all()

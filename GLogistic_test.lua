require 'nn'
require('./GLogistic.lua')
require 'gnuplot'

local myUtil = require('../MyCommon/util.lua')
local trainerPool = require('../MyCommon/trainerPool.lua')
local grnnUtil = require('./grnnUtil.lua')
autograd = require 'autograd'

local GLogistic_test = {}


function GLogistic_test.GLogistic_getOutput_test1()
  local input = torch.linspace(0, 1, 200)
  local output1 = GLogistic_getOutput(input, torch.Tensor({1, 2, 3, 2}))
  local output2 = GLogistic_getOutput(input, torch.Tensor({1, -2, 3, 2}))

  gnuplot.plot({'1', input, output1, 'points pt 2 ps 0.4'})
--               {'2', input, output2, 'points pt 2 ps 0.2 lc rgb "red"'})

end

function GLogistic_test.forward_test1()
  local gLogistic = nn.GLogistic(torch.Tensor({1, -2, 3, 2}))

  local input = torch.Tensor({0, 1, 2, 3, 4, 5})
  local output = gLogistic:forward(input)

  print(output)
end

function GLogistic_test.updateGradInput_test1()
  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({1, 2, 3, 2})
  local gLogistic = nn.GLogistic(weight)

  local input = torch.Tensor({1, 2, 3})
  local target = torch.Tensor({2.8, 2.3, 2})

  local output = gLogistic:forward(input)
  print(output)

  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = gLogistic:updateGradInput(input, df_do)


  print(target)
  print(gradInput)
end

function  GLogistic_test.accGradParameters_test1()
  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({1, 2, 3, 2})
  local gLogistic = nn.GLogistic(weight)

  local input = torch.Tensor({1, 2, 3})
  local target = torch.Tensor({2.8, 2.3, 2})

  local output = gLogistic:forward(input)

  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = gLogistic:updateGradInput(input, df_do)

  local _, gradParams = gLogistic:getParameters()
  print(gradParams)
  gLogistic:accGradParameters(input, df_do, 1)
  _, gradParams = gLogistic:getParameters()
  print(gradParams)
end

local fuGlogisticAuto = function (input, weight, bias)

    local a = weight[1]
    local b = weight[2]
    local c = weight[3]
    local d = weight[4]

    local denominator = torch.add(torch.exp(torch.mul(torch.add(torch.mul(input, -1), c), b)), 1)

    local output = torch.add(torch.mul(torch.pow(denominator, -1), a), d)

    return output
end


function GLogistic_test.autograd_test1()
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight = torch.Tensor({1, 2, 3, 2})
  local nSize = 100 

  local taData = GLogistic_test.genGLogisticData1(synthWeight, nSize)
  
  local initWeight = torch.Tensor({0, 0, 0, 0}) -- modify b from 1 to 2
  local gLogistic  = autograd.nn.AutoModule('GLogisticAuto')(fuGlogisticAuto, initWeight:clone())

  grnnUtil.logParams(gLogistic)
  trainerPool.full_CG(taData, gLogistic)
  grnnUtil.logParams(gLogistic)

  print("expected parameters")
  print(synthWeight)
end

-- Train b, a, k parameters using single GLogistic
function  GLogistic_test.accGradParameters_test2() 
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight = torch.Tensor({1, 2, 3, 2})
  local nSize = 100 

  local taData = GLogistic_test.genGLogisticData1(synthWeight, nSize)
  
  local initWeight = torch.Tensor({0, 0, 0, 0}) -- modify b from 1 to 2
  local gLogistic = nn.GLogistic(initWeight)

  grnnUtil.logParams(gLogistic)
  trainerPool.full_CG(taData, gLogistic)
  grnnUtil.logParams(gLogistic)

  print("expected parameters")
  print(synthWeight)

end

-- train with multiple outputs
function GLogistic_test.accGradParameters_test4()
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight1 = torch.Tensor({1, 2, 3, 2})
  local synthWeight2 = torch.Tensor({5, 4, 3, 2})
  local nSize = 100

  local taData, synthModel = GLogistic_test.genGLogisticData3(synthWeight1, synthWeight2, nSize)


  local weight1 = torch.Tensor({1.5, 2.2, 3, 2})
  local weight2 = torch.Tensor({0.1, 2.5, 3, 2})
  local m1 = nn.GLogistic(weight1)
  local m2 = nn.GLogistic(weight2)

  local model = grnnUtil.getSeqConModule(m1, m2)

--  --[[
  grnnUtil.logParams(model)
  trainerPool.full_CG(taData, model)
  grnnUtil.logParams(model)

  print("expected parameters")
  local synthParams, _ = synthModel:getParameters()
  print(synthParams)
  --]]

end

function GLogistic_test.genGLogisticData3(weight1, weight2, nSize)
  local teX = torch.rand(nSize,1)*10
  local m1 = nn.GLogistic(weight1)
  local m2 = nn.GLogistic(weight2)

  local seq = grnnUtil.getSeqConModule(m1, m2)
  local teY= seq:forward(teX)

  print(teY)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { torch.Tensor(1):copy(teX[i]), teY[i] }
    table.insert(taData, taRow)
  end

  return taData,seq 

end

function GLogistic_test.genGLogisticData1(weight, nSize)
  local teX = torch.rand(nSize)*5
  local gLogistic = nn.GLogistic(weight)
  local teY= gLogistic:forward(teX)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { torch.Tensor(1):fill(teX[i]), torch.Tensor(1):fill(teY[i]) }
    table.insert(taData, taRow)
  end

  return taData
--  gnuplot.plot('synthetic', teX, output, 'points pt 2 ps 0.4')
--  print(output)
end


function  GLogistic_test.all()
--  GLogistic_test.GLogistic_getOutput_test1()
--  GLogistic_test.forward_test1()
--  GLogistic_test.updateGradInput_test1()
--  GLogistic_test.accGradParameters_test1()
  GLogistic_test.accGradParameters_test2()
--GLogistic_test.autograd_test1()
--  GLogistic_test.accGradParameters_test4()
end

GLogistic_test.all()

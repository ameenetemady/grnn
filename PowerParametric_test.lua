require 'nn'
require 'gnuplot'
require('./PowerParametric.lua')

local myUtil = require('../MyCommon/util.lua')
local trainerPool = require('../MyCommon/trainerPool.lua')
local grnnUtil = require('./grnnUtil.lua')


local PowerParametric_test = {}

local epsilon = 0.00001
function PowerParametric_test.PowerParametric_getOutput_test1()
  local input =  torch.cat(torch.linspace(epsilon, 5, 20), torch.linspace(epsilon, 5, 20), 2)

  local weight = torch.Tensor({{2, 2}, {4, 2}})
 local output =  PowerParametric_getOutput(input, weight)
 print(output)
end

function  PowerParametric_test.PowerParametric_updateOutput_test1()
  local input =  torch.cat({torch.linspace(epsilon, 5, 20), torch.linspace(epsilon, 5, 20), torch.linspace(epsilon, 5, 20)}, 2)
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 0}})

  local mlp = nn.PowerParametric(3, weight)
  local output = mlp:forward(input)
  print(output)

end

function PowerParametric_test.PowerParametric_getGrad_x_test1()
  local input =  torch.cat({torch.linspace(epsilon, 5, 20), torch.linspace(epsilon, 5, 20), torch.linspace(epsilon, 5, 20)}, 2)
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 0}})

  local gradx = PowerParametric_getGrad_x(input, weight)
  print(gradx)
end

function PowerParametric_test.PowerParametric_updateGradInput_test1()
  local input =  torch.cat({torch.linspace(epsilon, 5, 20), torch.linspace(epsilon, 5, 20), torch.linspace(epsilon, 5, 20)}, 2)
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 0}})

  local mlp = nn.PowerParametric(3, weight)
  local output = mlp:forward(input)

  local gradInput = mlp:updateGradInput(input, output:mul(0.001) )
  print(gradInput)

end

function  PowerParametric_test.accGradParameters_test1()
  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 1}})
  local mlp = nn.PowerParametric(3, weight)

  local input = torch.Tensor({{1, 2, 3}})
--  local target = torch.Tensor({{ 7.3891, 20.0855, 54.5982 }})
  local target = torch.Tensor({{1, 20, 70}})

  local output = mlp:forward(input)

  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = mlp:updateGradInput(input, df_do)
  local _, gradParams = mlp:getParameters()
  print(gradParams)
  mlp:accGradParameters(input, df_do, 1)
  _, gradParams = mlp:getParameters()
  print(gradParams)

end

function PowerParametric_test.plot()
  torch.manualSeed(1)
  local nSize = 10 

  local teXA, teYA = PowerParametric_test.genData1Tensor(torch.Tensor({{2, 1.5}}), nSize)
  local teXB, teYB = PowerParametric_test.genData1Tensor(torch.Tensor({{1, 1}}), nSize)

--   gnuplot.plot({'1', teXA:squeeze(), teYA:squeeze(), 'points pt 2 ps 0.4'},
--                {'2', teXB:squeeze(), teYB:squeeze(), 'points pt 2 ps 0.2 lc rgb "red"'})

  local criterion = nn.MSECriterion()
  local f = criterion:forward(teYB, teYA) -- output, target


  local mlp = nn.PowerParametric(1, torch.Tensor({{1, 1}}))

  -- estimate df/dW
  local df_do = criterion:backward(teYB, teYA)
  local gradInput = mlp:updateGradInput(teXB, df_do)
  local a = torch.cat({teYA, teYB,gradInput,  df_do}, 2)
--  print(a)

  local param, gradParams = mlp:getParameters()
  print(gradParams)
--  print(param)
  mlp:accGradParameters(teXB, df_do, 0.1)
  param, gradParams = mlp:getParameters()
  print(gradParams)
--  print(param)


end

function  PowerParametric_test.accGradParameters_test2() 
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  --local synthWeight = torch.Tensor({{5, 2.5}, {1, 2}, {4, 5}})
  local synthWeight = torch.Tensor({{5, 2}})

  local nSize = 1000 

  local taData = PowerParametric_test.genData1(synthWeight, nSize)
  
  --local initWeight = torch.Tensor({{0.5, 0.5}, {0.5, 0.5}, {4, 3}})
  local initWeight = torch.Tensor({{4, 2}})
  local mlp = nn.PowerParametric(1, initWeight)

  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)

  print("expected parameters")
  print(synthWeight)


end

function PowerParametric_test.genData1Tensor(synthWeight, nSize)
  local nInputWidth = synthWeight:size(1)
  local teX = torch.rand(nSize, nInputWidth)
  local mlp = nn.PowerParametric(nInputWidth, synthWeight)
  local teY = mlp:forward(teX)

  return teX, teY
end


function PowerParametric_test.genData1(synthWeight, nSize)
  local nInputWidth = synthWeight:size(1)
  local teX = torch.rand(nSize, nInputWidth)*10
  local mlp = nn.PowerParametric(nInputWidth, synthWeight)
  local teY = mlp:forward(teX)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { teX[i], teY[i] }
    table.insert(taData, taRow)
  end


  return taData
end

function PowerParametric_test.all()
-- PowerParametric_test.PowerParametric_getOutput_test1()
--  PowerParametric_test.PowerParametric_updateOutput_test1()

--  PowerParametric_test.PowerParametric_getGrad_x_test1()
--  PowerParametric_test.PowerParametric_updateGradInput_test1()
--  PowerParametric_test.accGradParameters_test1()
  PowerParametric_test.accGradParameters_test2()
--  PowerParametric_test.plot()

end

PowerParametric_test.all()

require 'nn'
require('./SyngOne.lua')

local myUtil = require('../MyCommon/util.lua')
local trainerPool = require('../MyCommon/trainerPool.lua')
local grnnUtil = require('./grnnUtil.lua')


local SyngOne_test = {}

function SyngOne_test.SyngOne_getOutput_test1()
  local input =  torch.cat(torch.linspace(0.1, 5, 20), torch.linspace(0.1, 5, 20), 2)

  local weight = torch.Tensor({{3, 4}, {1, 2}})
 local output =  SyngOne_getOutput(input, weight)
 print(output)
end

function  SyngOne_test.SyngOne_updateOutput_test1()
  local input =  torch.cat({torch.linspace(0.1, 5, 20), torch.linspace(0.1, 5, 20), torch.linspace(0.1, 5, 20)}, 2)
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 0}})

  local mlp = nn.SyngOne(3, weight)
  local output = mlp:forward(input)
  print(output)

end

function SyngOne_test.SyngOne_getGrad_x_test1()
  local input =  torch.cat({torch.linspace(0.1, 5, 20), torch.linspace(0.1, 5, 20), torch.linspace(0.1, 5, 20)}, 2)
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 0}})

  local gradx = SyngOne_getGrad_x(input, weight)
  print(gradx)
end

function SyngOne_test.SyngOne_updateGradInput_test1()
  local input =  torch.cat({torch.linspace(0.1, 5, 20), torch.linspace(0.1, 5, 20), torch.linspace(0.1, 5, 20)}, 2)
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 0}})

  local mlp = nn.SyngOne(3, weight)
  local output = mlp:forward(input)

  local gradInput = mlp:updateGradInput(input, output:mul(0.001) )
  print(gradInput)

end

function  SyngOne_test.accGradParameters_test1()
  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 1}})
  local mlp = nn.SyngOne(3, weight)

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

function  SyngOne_test.accGradParameters_test2() 
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight = torch.Tensor({{3, 1}, {1, 4}, {0, 1}})

  local nSize = 100 

  local taData = SyngOne_test.genData1(synthWeight, nSize)
  
  local initWeight = torch.Tensor({{0, 0}, {0, 0}, {0, 0}})
  local mlp = nn.SyngOne(3, initWeight)

  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)

  print("expected parameters")
  print(synthWeight)


end

function SyngOne_test.genData1(synthWeight, nSize)
  local nInputWidth = synthWeight:size(1)
  local teX = torch.rand(nSize, nInputWidth)
  local mlp = nn.SyngOne(nInputWidth, synthWeight)
  local teY = mlp:forward(teX)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { teX[i], teY[i] }
    table.insert(taData, taRow)
  end


  return taData
end


function SyngOne_test.all()
-- SyngOne_test.SyngOne_getOutput_test1()
--  SyngOne_test.SyngOne_updateOutput_test1()

--  SyngOne_test.SyngOne_getGrad_x_test1()
--  SyngOne_test.SyngOne_updateGradInput_test1()
--  SyngOne_test.accGradParameters_test1()
  SyngOne_test.accGradParameters_test2()
end

SyngOne_test.all()

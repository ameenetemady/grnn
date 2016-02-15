require 'nn'
require 'gnuplot'

require('./SyngOne.lua')
require('./ExpParametric.lua')

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

function SyngOne_test.SyngOne_getOutput_test2()
  local input = torch.reshape(torch.linspace(0, 1, 200), 200,1)

  local synthWeight_SyngOne = torch.Tensor({{3, 2}})
  local synthWeight_ExpParametric= torch.Tensor({{-2, 6}})

  local mlp = nn.Sequential()
  mlp:add(nn.ExpParametric(1, synthWeight_ExpParametric))
  mlp:add(nn.SyngOne(1, synthWeight_SyngOne))




  local output1 = mlp:forward(input)

  gnuplot.plot({'1', input:squeeze(), output1:squeeze(), 'points pt 2 ps 0.4'})

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

function SyngOne_test.SyngOne_ExpParametric_integrate_test1()
  torch.manualSeed(1)
  --local synthWeight_SyngOne = torch.Tensor({{3, 1}, {1, 4}, {0, 1}})
  --local synthWeight_ExpParametric= torch.Tensor({{3, 1}, {1, 4}, {0, 1}})
  local synthWeight_SyngOne = torch.Tensor({{3, 2}})
  local synthWeight_ExpParametric= torch.Tensor({{-2, 6}})

  local fuMLPFactory = function(weightSynOne, weightExpParametric)
    local mlp = nn.Sequential()
    mlp:add(nn.ExpParametric(1, weightExpParametric))
    mlp:add(nn.SyngOne(1, weightSynOne))

    return mlp
  end

  local nSize = 500
  local taData = SyngOne_test.genData2(fuMLPFactory, nSize, synthWeight_SyngOne, synthWeight_ExpParametric)

  --local mlp = fuMLPFactory(torch.Tensor({{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}}), torch.Tensor({{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}}))
  local mlp = fuMLPFactory(torch.Tensor({{0, 0}}), torch.Tensor({{0, 0}}))


  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)

  print("expected parameters")
  print(synthWeight_SyngOne)
  print(synthWeight_ExpParametric)

end

function SyngOne_test.genData2(fuMLPFactory, nSize, weightSynOne, weightExpParametric)
  local nInputWidth = weightSynOne:size(1)
  local teX = torch.rand(nSize, nInputWidth)

  local mlp = fuMLPFactory(weightSynOne, weightExpParametric)

  local teY = mlp:forward(teX)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { teX[i], teY[i] }
    table.insert(taData, taRow)
  end

  return taData
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
--  SyngOne_test.SyngOne_getOutput_test2()
-- SyngOne_test.SyngOne_getOutput_test1()
--  SyngOne_test.SyngOne_updateOutput_test1()

--  SyngOne_test.SyngOne_getGrad_x_test1()
--  SyngOne_test.SyngOne_updateGradInput_test1()
--  SyngOne_test.accGradParameters_test1()
--  SyngOne_test.accGradParameters_test2()
  SyngOne_test.SyngOne_ExpParametric_integrate_test1()
end

SyngOne_test.all()

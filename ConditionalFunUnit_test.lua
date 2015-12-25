require 'nn'
require('./ConditionalFunUnit.lua')
require('./Hill.lua')

local myUtil = require('../ANN/common/util.lua')
local trainerPool = require('./trainerPool.lua')

local ConditionalFunUnit_test = {}

function ConditionalFunUnit_test.getMyMul10()
  return nn.MyMul10()
end

function ConditionalFunUnit_test.getHill(weight)
  return  nn.Hill(weight)

end

function ConditionalFunUnit_test.forward_test1()

  local condFunUnit = ConditionalFunUnit(ConditionalFunUnit_test.getHill)

  local teInput = torch.Tensor({{10, 0}, {12, 1}
  , {12, 0}
  })

  local output = condFunUnit:forward(teInput)

  print("output: ")
  print(output)

end

function ConditionalFunUnit_test.backward_test1()

  local nSize = 7 
  local teX = ConditionalFunUnit_test.getRandInput(nSize)

  -- 1) Generate target
  local fuFun1 = function()
    local weight = torch.Tensor({1, 2, 3, 2})
    return  nn.Hill(weight)
  end
  local condFunUnit1 = ConditionalFunUnit(fuFun1)
  local target = condFunUnit1:forward(teX)

  -- 2) Generate output
  local fuFun2 = function()
    local weight = torch.Tensor({1.5, 2, 3, 2})
    return  nn.Hill(weight)
  end
  local condFunUnit2 = ConditionalFunUnit(fuFun2)
  local output = condFunUnit2:forward(teX)

  -- 3) assess error
  local criterion = nn.MSECriterion()
  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = condFunUnit2:updateGradInput(teX, df_do)

  print("input:")
  print(teX)

  print("target")
  print(target)

  print("gradInput")
  print(gradInput)

end

function ConditionalFunUnit_test.train_test1()

  local nSize = 7 
  local teX = ConditionalFunUnit_test.getRandInput(nSize)

  -- 1) Generate target
  local fuFun1 = function()
    local weight = torch.Tensor({1, 2, 3, 2})
    return  nn.Hill(weight)
  end
  local condFunUnit1 = ConditionalFunUnit(fuFun1)
  local target = condFunUnit1:forward(teX)

  -- 2) Create Model
  local fuFun2 = function()
    local weight = torch.Tensor({1.5, 2.5, 4, 2})
    return  nn.Hill(weight)
  end
  local condFunUnit2 = ConditionalFunUnit(fuFun2)


  local taData = ConditionalFunUnit_test.getTable(teX, target)

  -- 3) train Model
  ConditionalFunUnit_test.logParams(condFunUnit2)
  trainerPool.full_CG(taData, condFunUnit2)
  ConditionalFunUnit_test.logParams(condFunUnit2)

end


function  ConditionalFunUnit_test.Cascade_forward_test1_getWeights_synth()
  local weight = torch.Tensor({{1.5, 2.5, 4, 2},
                              {1.5, 2.5, 4, 2},
                              {1.5, 2.5, 4, 2},
                              {1.5, 2.5, 4, 2}})
  return weight
end

function  ConditionalFunUnit_test.OneLayerCascade_forward_test1()
  local nSize = 7
  local nGenes = 4
  local teX = ConditionalFunUnit_test.getRandInput(nSize, nGenes)
  print(teX)


  local fuFun = function()
    local weight = torch.Tensor({1, 2, 3, 2})
    return  nn.Hill(weight)
  end

  local mlp_OneLayer = OneLayer_ConditionalFunUnit(fuFun, nGenes, 2)

  local target = mlp_OneLayer:forward(teX)

  print(target)

end


function  ConditionalFunUnit_test.MultiLayerCascade_forward_test1()
  local nSize = 7
  local nGenes = 4
  local teX = ConditionalFunUnit_test.getRandInput(nSize, nGenes)
  print(teX)


  local fuFun = function()
    local weight = torch.Tensor({1, 2, 3, 2})
    return  nn.Hill(weight)
  end

  local mlp_multiLayer = MultiLayer_ConditionalFunUnit(fuFun, nGenes)

  local target = mlp_multiLayer:forward(teX)

  print(target)

end



function ConditionalFunUnit_test.getTable(teX, teY)
  local nSize = teX:size(1)
  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { teX[i], torch.Tensor(1):fill(teY[i]) }
    table.insert(taData, taRow)
  end

  return taData
end

function ConditionalFunUnit_test.logParams(model)

  local parameters, gradParams = model:getParameters()
  print("parameters:")
  print(parameters)
end

function ConditionalFunUnit_test.getRandInput(nSize, nGenes)
  local nWidth = (nGenes or 1) + 1

  local teX = torch.rand(nSize, nWidth)
  teX:select(2, 1):mul(10)

  for i=2, nWidth do
    teX:select(2, i):round()
  end

  return teX
end

function  ConditionalFunUnit_test.all()
  torch.manualSeed(2)
--    ConditionalFunUnit_test.forward_test1()
--  ConditionalFunUnit_test.backward_test1()
--  ConditionalFunUnit_test.train_test1()
--  ConditionalFunUnit_test.OneLayerCascade_forward_test1()
  ConditionalFunUnit_test.MultiLayerCascade_forward_test1()

end

ConditionalFunUnit_test.all()

require 'nn'
autograd = require 'autograd'
require('./ExpParametric.lua')

local myUtil = require('../MyCommon/util.lua')
local trainerPool = require('../MyCommon/trainerPool.lua')
local grnnUtil = require('./grnnUtil.lua')

local fuExpParametric = function (input, weight, bias)
  local output = nil

  local nInputWidth = weight:size(1)
  assert(input:size(2) == nInputWidth, "dimentions don't match")

  for i=1, nInputWidth do
    local a = weight[i][1]
    local b = weight[i][2]
    local value = torch.exp(torch.add(torch.mul(torch.narrow(input, 2, i, 1), a), b))

    if output == nil then
      output = value

    else
      output = torch.cat(output, value, 2)
    end
  end

  return output
end


function test1()
  local weight = torch.Tensor({{1, 2}, {1, 2}, {1, 0}})
--  local bias = torch.Tensor(3,1):fill(-1.5)
  local autoExpParametric = autograd.nn.AutoModule('AutoExpParametric')(fuExpParametric, weight:clone())--, bias:clone())
 
  local input =  torch.cat({torch.linspace(-5, 5, 20), torch.linspace(-5, 5, 20), torch.linspace(-5, 5, 20)}, 2)
  local output = autoExpParametric:forward(input)
  local gradInput = autoExpParametric:updateGradInput(input, output:mul(0.001) )
  print(gradInput)
end

function test2()
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight = torch.Tensor({{3, 1}, {1, 4}, {0, 1}})

  local nSize = 100 

  local taData = genData1(synthWeight, nSize)
  
  local weight = torch.Tensor({{0, 0}, {0, 0}, {0, 0}})
  local mlp = autograd.nn.AutoModule('AutoExpParametric')(fuExpParametric, weight:clone())--, bias:clone())

  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)

  print("expected parameters")
  print(synthWeight)

end

function genData1(synthWeight, nSize)
  local nInputWidth = synthWeight:size(1)
  local teX = torch.rand(nSize, nInputWidth)
  local mlp = nn.ExpParametric(nInputWidth, synthWeight)
  local teY = mlp:forward(teX)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { teX[i], teY[i] }
    table.insert(taData, taRow)
  end


  return taData
end



--test1()
test2()


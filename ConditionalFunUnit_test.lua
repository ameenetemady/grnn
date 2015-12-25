require 'nn'
require('./ConditionalFunUnit.lua')
require('./Hill.lua')

local ConditionalFunUnit_test = {}

function ConditionalFunUnit_test.getMyMul10()
  return nn.MyMul10()
end

function ConditionalFunUnit_test.getHill()
  local weight = torch.Tensor({1, 2, 3, 2})
  return  nn.Hill(weight)

end

function ConditionalFunUnit_test.E2EA()

  local condFunUnit = ConditionalFunUnit(ConditionalFunUnit_test.getHill)
  local input = 10
  local cond = 0
  --local teInput = torch.Tensor({{input, cond}})
  local teInput = torch.Tensor({{10, 0}, {12, 1}
  , {12, 0}
  })

  local output = condFunUnit:forward(teInput)

  print("output: ")
 -- print(output:size())
  print(output)
--  print( output[1])
--  print( output[2])

end

function  ConditionalFunUnit_test.all()
    ConditionalFunUnit_test.E2EA()

end

ConditionalFunUnit_test.all()

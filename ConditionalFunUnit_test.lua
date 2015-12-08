require 'nn'
require('./ConditionalFunUnit.lua')

local ConditionalFunUnit_test = {}

function ConditionalFunUnit_test.getMyMul10()
  return nn.MyMul10()
end

function ConditionalFunUnit_test.E2EA()

  local condFunUnit = ConditionalFunUnit(ConditionalFunUnit_test.getMyMul10)
  local input = 10
  local cond = 0
  local teInput = torch.Tensor({{input, cond}})

  local output = condFunUnit:forward(teInput)

  print("output: ")
  print( output)

end

function  ConditionalFunUnit_test.all()
    ConditionalFunUnit_test.E2EA()

end

ConditionalFunUnit_test.all()

require 'nn'
require('./Hill.lua')

local Hill_test = {}

function Hill_test.hill_getOutput_test1()
  local input = torch.Tensor({1, 2, 3})
  local weight = torch.Tensor({1, 2, 3, 2})
  local output = hill_getOutput(input, weight)

  print(output)
end

function Hill_test.forward_test1()
  local weight = torch.Tensor({1, 2, 3, 2})
  local hill = nn.Hill(weight)

  local input = torch.Tensor({1, 2, 3})
  local output = hill:forward(input)

  print(output)

end

function  Hill_test.all()
--    Hill_test.E2EA()
--  Hill_test.hill_getOutput_test1()
  Hill_test.forward_test1()
end

Hill_test.all()

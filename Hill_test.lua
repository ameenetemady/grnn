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

function Hill_test.updateGradInput_test1()

  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({1, 2, 3, 2})
  local hill = nn.Hill(weight)

  local input = torch.Tensor({1, 2, 3})
  local target = torch.Tensor({2.8, 2.3, 2})

  local output = hill:forward(input)
  print(output)

  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = hill:updateGradInput(input, df_do)


  print(target)
  print(gradInput)

end


function  Hill_test.accGradParameters_test1()
  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({1, 2, 3, 2})
  local hill = nn.Hill(weight)

  local input = torch.Tensor({1, 2, 3})
  local target = torch.Tensor({2.8, 2.3, 2})

  local output = hill:forward(input)

  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = hill:updateGradInput(input, df_do)

  local _, gradParams = hill:getParameters()
  print(gradParams)
  hill:accGradParameters(input, df_do, 1)
  _, gradParams = hill:getParameters()
  print(gradParams)
end

-- Train for the b param
function  Hill_test.accGradParameters_test2() 
  local criterion = nn.MSECriterion()
  local synthWeight = torch.Tensor({1, 2, 3, 2})
  local nSize = 100 -- Todo: Ameen, continue from here
end

function  Hill_test.all()
--    Hill_test.E2EA()
--  Hill_test.hill_getOutput_test1()
--  Hill_test.forward_test1()
--  Hill_test.updateGradInput_test1()
--  Hill_test.accGradParameters_test1()
  Hill_test.accGradParameters_test2()
end

Hill_test.all()

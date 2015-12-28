require 'nn'
require('./CMulNoParam.lua')

local myUtil = require('../MyCommon/util.lua')

local CMulNoParam_test = {}

function CMulNoParam_test.CMulNoParam_getOutput_test1()
  local input = torch.Tensor({{1, 2}, {3, 4}, {5, 6}})
  local module = nn.CMulNoParam()
  local output = module:forward({input:select(2, 1), input:select(2, 2)})

  print(output)
end

function CMulNoParam_test.CMulNoParam_getGradInput_test1()
  local input = torch.Tensor({{1, 2}, {3, 4}, {5, 6}})
  local module = nn.CMulNoParam()
  local taInputSplit = {input:select(2, 1), input:select(2, 2)}
  local output = module:forward(taInputSplit)

  local gradOutput = torch.Tensor({0.1, 0.2, 0.1})
  local gradInput =   module:updateGradInput(taInputSplit, gradOutput)

  print(gradInput[1])
  print(gradInput[2])
end


function  CMulNoParam_test.all()
--  CMulNoParam_test.CMulNoParam_getOutput_test1()
  CMulNoParam_test.CMulNoParam_getGradInput_test1()
end

CMulNoParam_test.all()

require 'nn'

local CMulNoParamBatch, parent = torch.class('nn.CMulNoParamBatch', 'nn.Module')

function CMulNoParamBatch:__init() 
  parent.__init(self)
  self.gradInput = torch.Tensor()
end

function CMulNoParamBatch:updateOutput(input)
  local nCols = input:size(2)

  assert(nCols == 2, 'input must have two colunms')
  local a = input:narrow(2, 1, 1)
  local b = input:narrow(2, 2, 1)

  self.output:resize(a:size()):zero()
  self.output:cmul(a, b)

  return self.output
end


function CMulNoParamBatch:updateGradInput(input, gradOutput)
  local nCols = input:size(2)

  assert(nCols == 2, 'input must have two colunms')
  local a = input:narrow(2, 1, 1)
  local b = input:narrow(2, 2, 1)

  self.gradInput:resizeAs(input)

  self.gradInput:narrow(2, 1, 1):cmul(gradOutput, a)
  self.gradInput:narrow(2, 2, 1):cmul(gradOutput, b)

  return self.gradInput
end


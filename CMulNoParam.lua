require 'nn'

local CMulNoParam, parent = torch.class('nn.CMulNoParam', 'nn.Module')

function CMulNoParam:__init() 
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function CMulNoParam:updateOutput(input)
  assert(#input == 2, 'input must be a pair of minibatches ')
  local a, b = table.unpack(input)

  self.output:resize(a:size()):zero()
  self.output:cmul(a, b)

  return self.output
end


function CMulNoParam:updateGradInput(input, gradOutput)
  assert(#input == 2, 'input must be a pair of minibatches ')
  local a, b = table.unpack(input)

  self.gradInput[1]:resizeAs(a)
  self.gradInput[2]:resizeAs(b)


  self.gradInput[1]:cmul(gradOutput, b)
  self.gradInput[2]:cmul(gradOutput, a)


  return self.gradInput
end



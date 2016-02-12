require 'nn'


local SyngOne, parent = torch.class('nn.SyngOne', 'nn.Module')

-- *** allways assumes two dimentional input, hence number of columns (width) should be specified ***
function SyngOne:__init(nInputWidth, weight_initial)
  assert(nInputWidth > 0, "invalid nInputWidth parameter")
  parent.__init(self)
  self.weight = torch.Tensor(nInputWidth, 2):zero() -- {a0, a1}
  self.gradWeight = torch.Tensor(nInputWidth, 2):zero()
  self.weight_initial = weight_initial

  self:reset()
end

function SyngOne:reset()
  -- lets start with some const initial weight, will consider ranmodness later

  if self.weight_initial ~= nil then
    self.weight:copy(self.weight_initial)
  else
    self.weight:fill(1)
  end
end

function SyngOne:updateOutput(input)
  self.output = SyngOne_getOutput(input, self.weight)
  return self.output
end

function SyngOne:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  local gradx = SyngOne_getGrad_x(input, self.weight)

  self.gradInput:cmul(gradx, gradOutput)

  return self.gradInput
end

function SyngOne:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1
  local nInputWidth = input:size(2)

  for i=1, nInputWidth do
    self.gradWeight[i][1] = self.gradWeight[i][1] + scale * SyngOne_getGrad_a0(input:narrow(2, i, 1), self.weight[i]):dot(gradOutput:narrow(2, i, 1))

    self.gradWeight[i][2] = self.gradWeight[i][2] + scale * SyngOne_getGrad_a1(input:narrow(2, i, 1), self.weight[i]):dot(gradOutput:narrow(2, i, 1))
  end
end

function SyngOne_getGrad_a0(input, weight)

  local grada = input:clone():fill(0)
  local a0 = weight[1]
  local a1 = weight[2]
  grada:copy(input:clone():add(1):pow(-1))

  return grada
end

function SyngOne_getGrad_a1(input, weight)
  local grada = input:clone():fill(0)
  local a0 = weight[1]
  local a1 = weight[2]

  grada:copy(input:clone():cdiv(input:clone():add(1)))

  return grada
end


function SyngOne_getOutput(input, weight)
  local output = input:clone():fill(0)
  local nInputWidth = weight:size(1)
  assert(input:size(2) == nInputWidth, "dimentions don't match")

  for i=1, nInputWidth do
    local a0 = weight[i][1]
    local a1 = weight[i][2]
    local x = input:clone():narrow(2, i, 1)
    output:narrow(2, i, 1):copy(x:clone():mul(a1):add(a0):cdiv(x:clone():add(1)) )
  end

  return output
end

function SyngOne_getGrad_x(input, weight)
  local gradx = input:clone()
  local nInputWidth = weight:size(1)
  assert(input:size(2) == nInputWidth, "dimentions don't match")

   for i=1, nInputWidth do
    local a0 = weight[i][1]
    local a1 = weight[i][2]
    gradx:narrow(2, i, 1):copy( input:clone():narrow(2, i, 1):add(1):pow(-2):mul(a1-a0) )
  end

  return gradx
end



require 'nn'


local ExpParametric, parent = torch.class('nn.ExpParametric', 'nn.Module')

-- *** allways assumes two dimentional input, hence number of columns (width) should be specified ***
function ExpParametric:__init(nInputWidth, weight_initial)
  assert(nInputWidth > 0, "invalid nInputWidth parameter")
  parent.__init(self)
  self.weight = torch.Tensor(nInputWidth, 2):zero() -- {a, b}
  self.gradWeight = torch.Tensor(nInputWidth, 2):zero()
  self.weight_initial = weight_initial

  self:reset()
end

function ExpParametric:reset()
  -- lets start with some const initial weight, will consider ranmodness later

  if self.weight_initial ~= nil then
    self.weight:copy(self.weight_initial)
  else
    self.weight:fill(1)
  end
end

function ExpParametric:updateOutput(input)
  self.output = ExpParametric_getOutput(input, self.weight)
  return self.output
end

function ExpParametric:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  local gradx = ExpParametric_getGrad_x(input, self.weight)

  self.gradInput:cmul(gradx, gradOutput)

  return self.gradInput
end

function ExpParametric:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1
  local nInputWidth = input:size(2)

  for i=1, nInputWidth do
    self.gradWeight[i][1] = self.gradWeight[i][1] + scale * ExpParametric_getGrad_a(input:narrow(2, i, 1), self.weight[i]):dot(gradOutput:narrow(2, i, 1))

    self.gradWeight[i][2] = self.gradWeight[i][2] + scale * ExpParametric_getGrad_b(input:narrow(2, i, 1), self.weight[i]):dot(gradOutput:narrow(2, i, 1))
  end
end

function ExpParametric_getGrad_a(input, weight)

  local grada = input:clone():fill(0)
  local a = weight[1]
  local b = weight[2]
  grada:copy(input:clone():mul(a):add(b):exp():cmul(input))

  return grada
end

function ExpParametric_getGrad_b(input, weight)
  local grada = input:clone():fill(0)
  local a = weight[1]
  local b = weight[2]

  grada:copy(input:clone():mul(a):add(b):exp())

  return grada
end


function ExpParametric_getOutput(input, weight)
  local output = input:clone():fill(0)
  local nInputWidth = weight:size(1)
  assert(input:size(2) == nInputWidth, "dimentions don't match")

  for i=1, nInputWidth do
    local a = weight[i][1]
    local b = weight[i][2]
    output:narrow(2, i, 1):copy( input:clone():narrow(2, i, 1):mul(a):add(b):exp() )
  end

  return output
end

function ExpParametric_getGrad_x(input, weight)
  local gradx = input:clone()
  local nInputWidth = weight:size(1)
  assert(input:size(2) == nInputWidth, "dimentions don't match")

   for i=1, nInputWidth do
    local a = weight[i][1]
    local b = weight[i][2]
    gradx:narrow(2, i, 1):copy( input:clone():narrow(2, i, 1):mul(a):add(b):exp():mul(a) )
  end

  return gradx
end



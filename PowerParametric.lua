require 'nn'


local PowerParametric, parent = torch.class('nn.PowerParametric', 'nn.Module')

local epsilon = 0.01
-- *** allways assumes two dimentional input, hence number of columns (width) should be specified ***
function PowerParametric:__init(nInputWidth, weight_initial)
  assert(nInputWidth > 0, "invalid nInputWidth parameter")
  parent.__init(self)
  self.weight = torch.Tensor(nInputWidth, 2):zero() -- {a, b}
  self.gradWeight = torch.Tensor(nInputWidth, 2):zero()
  self.weight_initial = weight_initial

  self:reset()
end

function PowerParametric:reset()
  -- lets start with some const initial weight, will consider ranmodness later

  if self.weight_initial ~= nil then
    self.weight:copy(self.weight_initial)
  else
    self.weight:fill(1)
  end
end

function PowerParametric:updateOutput(input)
  self.output = PowerParametric_getOutput(input, self.weight)
  return self.output
end

function PowerParametric:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  local gradx = PowerParametric_getGrad_x(input, self.weight)

  self.gradInput:cmul(gradx, gradOutput)

  return self.gradInput
end

function PowerParametric:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1
  local nInputWidth = input:size(2)

  for i=1, nInputWidth do
    self.gradWeight[i][1] = self.gradWeight[i][1] + scale * PowerParametric_getGrad_n(input:narrow(2, i, 1), self.weight[i]):dot(gradOutput:narrow(2, i, 1))

    self.gradWeight[i][2] = self.gradWeight[i][2] + scale * PowerParametric_getGrad_k(input:narrow(2, i, 1), self.weight[i]):dot(gradOutput:narrow(2, i, 1))
  end

  --[[
  print("********* accGradParameters:gradWeight **********")
  print(self.gradWeight)
  print("********* accGradParameters:weight **********")
  print(self.weight)
  --]]

end

function PowerParametric_getGrad_n(input, weight)

  local gradn = input:clone():fill(0)
  local n = weight[1]
  local k = weight[2]
  gradn:copy(input:clone():pow(n^2 + 1):mul(2*n*k):cmul(torch.log(input)))
  --[[
  print("############ begin")
  print("******* n ****** ")
  print(n)
  print("***** base: *********")
  print(base)

  print("***** base:log: *********")
  print(base:clone():log())
  print("***** base:pow n^2+1: *********")
  print(torch.pow(base, n^2+1))

  print("############ end")
  --]]
  return gradn
end

function PowerParametric_getGrad_k(input, weight)
  local gradk = input:clone():fill(0)
  local n = weight[1]
  local k = weight[2]

  gradk:copy(input:clone():pow(n^2+1))

  return gradk
end


function PowerParametric_getOutput(input, weight)
  local output = input:clone():fill(0)
  local nInputWidth = weight:size(1)
  assert(input:size(2) == nInputWidth, "dimentions don't match")

  for i=1, nInputWidth do
    local n = weight[i][1]
    local k = weight[i][2]
    output:narrow(2, i, 1):copy( input:clone():narrow(2, i, 1):pow(n^2 + 1):mul(k) )
  end

  return output
end

function PowerParametric_getGrad_x(input, weight)
  local gradx = input:clone()
  local nInputWidth = weight:size(1)
  assert(input:size(2) == nInputWidth, "dimentions don't match")

   for i=1, nInputWidth do
    local n = weight[i][1]
    local k = weight[i][2]
    gradx:narrow(2, i, 1):copy( input:clone():narrow(2, i, 1):pow(n^2):mul(k*(n^2+1)) )
  end

  return gradx
end


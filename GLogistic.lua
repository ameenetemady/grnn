require 'nn'

local GLogistic, parent = torch.class('nn.GLogistic', 'nn.Module')

function GLogistic:__init(weight_initial)
  parent.__init(self)
  self.weight = torch.Tensor(4):zero() -- {a, b, c, d}
  self.gradWeight = torch.Tensor(4):zero()
  self.weight_initial = weight_initial

  self:reset()
end

function GLogistic:reset()
  -- lets start with some const initial weight, will consider ranmodness later

  if self.weight_initial ~= nil then
    self.weight:copy(self.weight_initial)
  else
    self.weight:fill(1)
  end
end

function GLogistic:updateOutput(input)
  self.output = GLogistic_getOutput(input, self.weight)
  return self.output
end

function GLogistic:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  local gradx = GLogistic_getGrad_x(input, self.weight)

  self.gradInput:cmul(gradx, gradOutput)

  return self.gradInput
end

function GLogistic:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1

  self.gradWeight[1] = self.gradWeight[1] + scale * GLogistic_getGrad_a(input, self.weight):dot(gradOutput)

  self.gradWeight[2] = self.gradWeight[2] + scale * GLogistic_getGrad_b(input, self.weight):dot(gradOutput)

  self.gradWeight[3] = self.gradWeight[3] + scale * GLogistic_getGrad_c(input, self.weight):dot(gradOutput)

  self.gradWeight[4] = self.gradWeight[4] + scale * GLogistic_getGrad_d(input, self.weight):dot(gradOutput)
end

function GLogistic_getOutput(input, weight)
  local output = input:clone()
  local a, b, c, d = GLogistic_extract_params(weight)

  output:apply(
    function(x)
      local denominator = 1 + math.exp(b * (c-x))

      local y = a/denominator + d
      return y
    end)

  return output
end

function GLogistic_getGrad_x(input, weight)
  local gradx = input:clone()

  local a, b, c, d = GLogistic_extract_params(weight)

  gradx:apply(
    function(x)
      local expComp = math.exp(b * (c-x))
      local y = (a*b*expComp)/math.pow(1+expComp, 2)
      return y
    end)

  return gradx
end

function GLogistic_getGrad_a(input, weight)
  local grada = input:clone()

  local a, b, c, d = GLogistic_extract_params(weight)

  grada:apply(
    function(x)
      local expComp = math.exp(b * (c-x))
      local y = 1/(1 + expComp)
      return y
    end)

  return grada
end

function GLogistic_getGrad_b(input, weight)
  local gradb = input:clone()

  local a, b, c, d = GLogistic_extract_params(weight)

  gradb:apply(
    function(x)
      local expComp = math.exp(b * (c-x))
      local y = -(a*(c-x)*expComp)/math.pow(1 + expComp, 2)
      return y
    end)

  return gradb
end

function GLogistic_getGrad_c(input, weight)
  local gradc = input:clone()

  local a, b, c, d = GLogistic_extract_params(weight)

  gradc:apply(
    function(x)
      local expComp = math.exp(b * (c-x))
      local y = -(a*b*expComp)/math.pow(1 + expComp, 2)
      return y
    end)

  return gradc
end

function GLogistic_getGrad_d(input, weight)
  local gradd = input:clone():fill(1)
  return gradd
end


function GLogistic_extract_params(weight)
  return weight[1], weight[2], weight[3], weight[4]
end


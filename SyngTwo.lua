require 'nn'

local SyngTwo, parent = torch.class('nn.SyngTwo', 'nn.Module')

function SyngTwo:__init(weight_initial)
  parent.__init(self)

  self.weight = torch.Tensor(5):zero() -- {b0, b1, b2, b3, b4, p}
  self.gradWeight = torch.Tensor(5):zero()
  self.weight_initial = weight_initial

  self:reset()
end

function SyngTwo:reset()
  -- lets start with some const initial weight, will consider ranmodness later

  if self.weight_initial ~= nil then
    self.weight:copy(self.weight_initial)
  else
    self.weight:fill(1)
  end
end

function SyngTwo:updateOutput(input)
  self.output = SyngTwo_getOutput(input, self.weight)
  return self.output
end

function SyngTwo:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  local common = SyngTwo_getGrad_common(input, self.weight, self.output)

  local gradx1 = SyngTwo_getGrad_x1(common)
  local gradx2 = SyngTwo_getGrad_x2(common)

  self.gradInput:narrow(2, 1, 1):cmul(gradx1, gradOutput)
  self.gradInput:narrow(2, 2, 1):cmul(gradx2, gradOutput)

  return self.gradInput
end

function SyngTwo:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1

  local common = SyngTwo_getGrad_common(input, self.weight, self.output)

  self.gradWeight[1] = self.gradWeight[1] + scale * SyngTwo_getGrad_b0(common):dot(gradOutput)

  self.gradWeight[2] = self.gradWeight[2] + scale * SyngTwo_getGrad_b1(common):dot(gradOutput)

  self.gradWeight[3] = self.gradWeight[3] + scale * SyngTwo_getGrad_b2(common):dot(gradOutput)

  self.gradWeight[4] = self.gradWeight[4] + scale * SyngTwo_getGrad_b3(common):dot(gradOutput)

  self.gradWeight[5] = self.gradWeight[5] + scale * SyngTwo_getGrad_p(common):dot(gradOutput)
end

function SyngTwo_getGrad_common(input, weight, output)
  local c = {}

  c.b0, c.b1, c.b2, c.b3, c.p = SyngTwo_extract_params(weight)
  c.a0, c.a1, c.a2, c.a3, c.rho = SyngTwo_extract_params_aMode(weight)
  c.d = c.b0^2 + c.b1^2 + c.b2^2 + c.b3^2

  c.x1 = input:narrow(2, 1, 1):clone()
  c.x2 = input:narrow(2, 2, 1):clone()

  c.fu = SyngTwo_fu(input, weight)
  c.fl = SyngTwo_fl(input, weight)
  c.f = output

  return c
end

function SyngTwo_getGrad_b0(c)
  local term1 = -torch.mul(c.f, 2*c.b0/c.d^2)
  local term2 =  torch.pow(c.fl, -1):mul(2*c.b0/c.d)

  return term1 + term2
end

function SyngTwo_getGrad_b1(c)
  local term1 = -torch.mul(c.f, 2*c.b1/c.d^2)
  local term2 = torch.pow(c.fl, -1):cmul(c.x1):mul(2*c.b1/d)

  return term1 + term2
end

function SyngTwo_getGrad_b2(c)
  local term1 = -torch.mul(c.f, 2*c.b2/c.d^2)
  local term2 = torch.pow(c.fl, -1):cmul(c.x2):mul(2*c.b2/d)

  return term1 + term2
end

function SyngTwo_getGrad_b3(c)
end

function SyngTwo_getGrad_p(c)
end

function SyngTwo_getGrad_x1(c)
  local term1 = torch.mul(c.x2, c.rho*c.a3):add(c.a1):cdiv(c.fl)
  local term2 = torch.mul(c.x2, c.rho):add(1):cmul(c.fu):cmul(torch.pow(c.fl, -2))

  return term1 - term2
end

function SyngTwo_getGrad_x2(c)
  local term1 = torch.mul(c.x1, c.rho*c.a3):add(c.a2):cdiv(c.fl)
  local term2 = torch.mul(c.x1, c.rho):add(1):cmul(c.fu):cmul(torch.pow(c.fl, -2))

  return term1 - term2
end


function SyngTwo_getOutput(input, weight)
  local fu = SyngTwo_fu(input, weight)
  local fl = SyngTwo_fl(input, weight)

  return torch.cdiv(fu, fl)
end

function SyngTwo_fu(input, weight)
  local b0, b1, b2, b3, p = SyngTwo_extract_params(weight)
  local d = b0^2 + b1^2 + b2^2 + b3^2

  local x1 = input:narrow(2, 1, 1):clone()
  local x2 = input:narrow(2, 2, 1):clone()

  local sum = torch.mul(x1, b1^2) + torch.mul(x2, b2^2) + torch.cmul(x1, x2):mul(b3^2):mul(SyngTwo_rho(p)) + b0^2

  return sum:div(d)
end

function SyngTwo_fl(input, weight)
  local b0, b1, b2, b3, p = SyngTwo_extract_params(weight)

  local x1 = input:narrow(2, 1, 1):clone()
  local x2 = input:narrow(2, 2, 1):clone()

  return  x1 + x2 + torch.cmul(x1, x2):mul(SyngTwo_rho(p)) + 1
end

function SyngTwo_extract_params(weight)
  return weight[1], weight[2], weight[3], weight[4], weight[5]
end

function SyngTwo_extract_params_aMode(weight)
  local b0, b1, b2, b3, p = SyngTwo_extract_params(weight)
  local d = b0^2 + b1^2 + b2^2 + b3^2

  return b0^2/d, b1^2/d, b2^2/d, b3^2/d, SyngTwo_rho(p)
end


function SyngTwo_rho(p)
  return 1/(1+p^2)
end

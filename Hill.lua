require 'nn'

local Hill, parent = torch.class('nn.Hill', 'nn.Module')

function Hill:__init(weight_initial) -- later consider the multiple output case
  parent.__init(self)
  self.weight = torch.Tensor(4):zero() -- {b, a, k, n}
  self.gradWeight = torch.Tensor(4):zero()
  self.weight_initial = weight_initial

  self:reset()

end

function Hill:reset()
  -- lets start with some const initial weight, will consider ranmodness later

  if self.weight_initial ~= nil then
    self.weight:copy(self.weight_initial)
  else
    self.weight:fill(1)
  end

end

function Hill:ensureConstraint()
  self.weight:abs()
end

function Hill:updateOutput(input)
  --self:ensureConstraint()
  self.output = hill_getOutput(input, self.weight)
  return self.output
end

function Hill:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  local gradx = hill_getGrad_x(input, self.weight)

  self.gradInput:cmul(gradx, gradOutput) 

  return self.gradInput
end

function Hill:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1

--  print(input)
--  print(self.weight)
--  print(gradOutput)
  self.gradWeight[1] = self.gradWeight[1] + scale * hill_getGrad_b(input, self.weight):dot(gradOutput)

  self.gradWeight[2] = self.gradWeight[2] + scale * hill_getGrad_a(input, self.weight):dot(gradOutput)

  self.gradWeight[3] = self.gradWeight[3] + scale * hill_getGrad_k(input, self.weight):dot(gradOutput)

  self.gradWeight[4] = self.gradWeight[4] + scale * hill_getGrad_n(input, self.weight):dot(gradOutput)

end

function hill_extract_params(weight)
  return weight[1], weight[2], weight[3], weight[4]
end

function hill_getGrad_b(input, weight)
  local gradb = input:clone():fill(1)
  return gradb
end

function hill_getGrad_a(input, weight)
  local grada = input:clone()

  local b, a, k, n = hill_extract_params(weight)

  grada:apply(
    function(x)
      local denominator = 1 + math.pow(x/k, n)
      assert( denominator ~= 0, "(x/k) cannot be -1")
      local y = 1/denominator
      return y
    end)

  return grada

end

function hill_getGrad_k(input, weight)
  local gradk = input:clone()

  local b, a, k, n = hill_extract_params(weight)

  gradk:apply(
    function(x)
      local denominator = math.pow(k, n) + math.pow(x, n)
      assert( denominator ~= 0, "k^n + x^n cannot be zero")
      local y = a * n * math.pow(k, n-1) * (1/denominator - math.pow(k, n)/math.pow(denominator, 2))
      return y
    end)

  return gradk
end

function hill_getGrad_n(input, weight)
  local gradn = input:clone()
  -- returning zero for now, until find a way for "constrained" optimization and conforming with the domain
  gradn:zero()

  --[[
  local b, a, k, n = hill_extract_params(weight)

  assert( k > 0, "k should be greater than zero")

  gradn:apply(
    function(x)
      assert( x > 0, "x should be greater than zero")

      local denominator = math.pow(k, n) + math.pow(x, n)
      assert( denominator ~= 0, "k^n + x^n cannot be zero" .. ", k:" .. k .. ", x:" .. x .. ", n:" .. n .. ", denominator:" .. denominator )
      local y = a * math.pow(k, n) * math.log(k) / denominator - (math.pow(k, n) * math.log(k) + math.pow(x, n) * math.log(x))/ math.pow(denominator, 2)
      return y
    end)

    --]]


  return gradn
end

function hill_getOutput(input, weight)
  local output = input:clone()
  local b, a, k, n = hill_extract_params(weight)
  assert(k ~= 0, "weight[3] (k), cannot be zero")

  output:apply(
    function(x)
      local denominator = 1 + math.pow(x/k, n)
      assert( denominator ~= 0, "(x/k) cannot be -1")
      local y = b + a/denominator
      return y
    end)

  return output
end

function hill_getGrad_x(input, weight)
  local gradx = input:clone()

  local b, a, k, n = hill_extract_params(weight)
  assert(k ~= 0, "weight[3] (k), cannot be zero")

  gradx:apply(
    function(x)
      local y = -a * math.pow(k, n) * math.pow(math.pow(k, n) + math.pow(x, n), -2) * n * math.pow(x, n-1)
      return y
    end)

  return gradx

end


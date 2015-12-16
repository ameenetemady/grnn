require 'nn'

local Hill, parent = torch.class('nn.Hill', 'nn.Module')

function Hill:__init(weight_initial) -- later consider the multiple output case
  parent.__init(self)
  self.weight = torch.Tensor(4) -- {b, a, k, n}
  self.gradWeight = torch.Tensor(4)
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

function Hill:updateOutput(input)
  self.output = hill_getOutput(input, self.weight)
  return self.output
end

function Hill:updateGradInput(input, gradOutput)
  self.gradInput = hill_getGrad_x(input, self.weight)

  return self.gradInput
end

function Hill:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1

  self.gradWeight[1] = self.gradWeight[1] + scale * hill_getGrad_b():dot(gradOutput)

  self.gradWeight[2] = self.gradWeight[2] + scale * hill_getGrad_a(input, self.weight):dot(gradOutput)

  self.gradWeight[3] = self.gradWeight[3] + scale * hill_getGrad_k(input, self.weight):dot(gradOutput)

  self.gradWeight[4] = self.gradWeight[4] + scale * hill_getGrad_n(input, self.weight):dot(gradOutput)

end

function hill_extract_params(weight)
  return weight[1], weight[2], weight[3], weight[4]
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

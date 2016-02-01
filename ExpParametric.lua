require 'nn'

local ExpParametric, parent = torch.class('nn.ExpParametric', 'nn.Module')

-- *** allways assume two dimentional input, hence number of columns (width) should be specified ***
function ExpParametric:__init(nInputWidth, weight_initial)
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

function ExpParametric_getOutput(input, weight)
  local output = input:clone()
  local nInputWidth = weight:size(1)

  output:apply(
    function(x)
      local y = x:clone():fill(0)

      for i=1, nInputWidth do
        local a = weight[i][1]
        local b = weight[i][2]
        y[i] = math.exp(a*x + b)

      end

      return y
    end)

  return output
end


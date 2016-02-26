local autograd = require 'autograd'
local SyngOneAuto = {}

do
  local fuAutoSyngOneFull = function(input, weight, bias)
    local output = nil

    local nInputWidth = weight:size(1)
    assert(input:size(2) == nInputWidth, "dimentions don't match")

    for i=1, nInputWidth do
      local a0 = weight[i][1]
      local a1 = weight[i][2]
      local b = weight[i][3]
      local c = weight[i][4]

      local y = torch.exp(torch.add(torch.mul(input, b), c))
      local value = torch.add(torch.mul(y, a1), a0)

      if output == nil then
        output = value

      else
        output = torch.cat(output, value, 2)
      end
    end

    return output
  end

  function  SyngOneAuto.new(weight)
    weight = weight or torch.rand(1, 4)*2-1

    return autograd.nn.AutoModule('AutoSyngOneFull')(fuAutoSyngOneFull, weight:clone())
  end

  return SyngOneAuto
end

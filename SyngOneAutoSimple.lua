local autograd = require 'autograd'
local SyngOneAutoSimple = {}

do
  local fuAutoSyngOneFull = function(input, weight, bias)

      local a0 = weight[1]
      local a1 = weight[2]
      local b =  weight[3]
      local c =  weight[4]

      local y = torch.exp(torch.add(torch.mul(input, b), c))
      local value = torch.add(torch.mul(y, a1), a0)

    return value 
  end

  function  SyngOneAutoSimple.new(weight)
    weight = weight or torch.rand(4)*2-1

    return autograd.nn.AutoModule('AutoSyngOneFullSimple')(fuAutoSyngOneFull, weight:clone())
  end

  return SyngOneAutoSimple
end

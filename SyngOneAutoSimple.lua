local autograd = require 'autograd'
local SyngOneAutoSimple = {}

do
  local fuAutoSyngOneFull = function(input, weight, bias)

      local a = weight[1]
      local b = weight[2]
      local c =  weight[3]
      local d =  weight[4]

      local y = torch.exp(torch.add(torch.mul(input, -b), b*c))
      local but = torch.add(y, 1)
      local output = torch.add(torch.mul(torch.pow(but, -1), a), d)

    return output
  end

  function  SyngOneAutoSimple.new(weight)
    weight = weight or torch.rand(4)*2-1

    return autograd.nn.AutoModule('AutoSyngOneFullSimple')(fuAutoSyngOneFull, weight:clone())
  end

  return SyngOneAutoSimple
end

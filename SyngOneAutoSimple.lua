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
      local output = torch.add(torch.mul(torch.pow(but, -1), a-d), d)

    return output
  end

  function  SyngOneAutoSimple.new(weight)
    weight = weight or torch.rand(4)*2-1

    return autograd.nn.AutoModule('AutoSyngOneFullSimple')(fuAutoSyngOneFull, weight:clone())
  end

  function SyngOneAutoSimple.pri_isActivator(teInputSlice, teTargetSclice, teKOSlice)
    local tePresent = torch.ByteTensor(teKOSlice:size()):copy(teKOSlice)
    local teTargetPresent = teTargetSclice:maskedSelect(tePresent)
    local teInputPresent = teInputSlice:maskedSelect(tePresent)

    local __ , minInputId = torch.min(teInputPresent, 1)
    local __ , maxInputId = torch.max(teInputPresent, 1)

    local dA = teTargetPresent[minInputId:squeeze()]
    local dB = teTargetPresent[maxInputId:squeeze()]

    return dA < dB
  end

  function SyngOneAutoSimple.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
    local isActivator = SyngOneAutoSimple.pri_isActivator(teInputSlice, teTargetSclice, teKOSlice)
    local a = torch.max(teTargetSclice)
    local b = isActivator and 2 or -2
    local c = (torch.max(teInputSlice) + torch.min(teInputSlice))/2
    local d = torch.min(teTargetSclice)

    return torch.Tensor({a, b, c, d})
  end 

  return SyngOneAutoSimple
end

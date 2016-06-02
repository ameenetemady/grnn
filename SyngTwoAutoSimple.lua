local autograd = require 'autograd'
local SyngTwoAutoSimple = {}

do
  local fuAutoSyngTwoFull = function(input, weight, bias)

    local a = weight[1]
    local b1 = weight[2]
    local c1 =  weight[3]
    local b2 =  weight[4]
    local c2 =  weight[5]
    local d =  weight[6]

    local x1 = torch.narrow(input, 2, 1, 1)
    local x2 = torch.narrow(input, 2, 2, 1)


    local exponent = torch.add(torch.add(torch.mul(x1, -b1), b1*c1),
                               torch.add(torch.mul(x2, -b2), b2*c2))
    local but = torch.add(torch.exp(exponent), 1)

    local output = torch.add(torch.mul(torch.pow(but, -1), a-d), d)

    return output
  end

  function  SyngTwoAutoSimple.new(weight)
    weight = weight or torch.rand(6)*2-1

    return autograd.nn.AutoModule('AutoSyngTwoFullSimple')(fuAutoSyngTwoFull, weight:clone())
  end

  function SyngTwoAutoSimple.pri_isActivator(teInputSlice, teTargetSclice, teKOSlice)
    local tePresent = torch.ByteTensor(teKOSlice:size()):copy(teKOSlice)
    local teTargetPresent = teTargetSclice:maskedSelect(tePresent)
    local teInputPresent = teInputSlice:maskedSelect(tePresent)

    local __ , minInputId = torch.min(teInputPresent, 1)
    local __ , maxInputId = torch.max(teInputPresent, 1)

    local dA = teTargetPresent[minInputId:squeeze()]
    local dB = teTargetPresent[maxInputId:squeeze()]

    return dA < dB
  end

  function SyngTwoAutoSimple.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
--    return torch.rand(6)*2-1

    local isX1Activator = SyngTwoAutoSimple.pri_isActivator(teInputSlice:narrow(2, 1, 1), teTargetSclice, teKOSlice)
    local isX2Activator = SyngTwoAutoSimple.pri_isActivator(teInputSlice:narrow(2, 2, 1), teTargetSclice, teKOSlice)
    local a = torch.max(teTargetSclice)
    local b1 = isX1Activator and 2 or -2
    local b2 = isX2Activator and 2 or -2
    local c1 = (torch.max(teInputSlice:narrow(2, 1, 1)) + torch.min(teInputSlice:narrow(2, 1, 1)))/2
    local c2 = (torch.max(teInputSlice:narrow(2, 2, 1)) + torch.min(teInputSlice:narrow(2, 2, 1)))/2
    local d = torch.min(teTargetSclice)

    return torch.Tensor({a, b1, c1, b2, c2, d})
  end 

  return SyngTwoAutoSimple
end

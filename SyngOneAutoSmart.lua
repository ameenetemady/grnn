local autograd = require 'autograd'
local syngUtil = syngUtil or require('./syngUtil.lua')

local SyngOneAutoSmart = {}

do
  local fuAutoSyngOneFull = function(input, weight, bias)

      local a = weight[1]
      local b = weight[2]
      local c =  weight[3]
      local d =  weight[4]

      local y = torch.exp(torch.add(torch.mul(input, b), c))
      local but = torch.add(y, 1)
      local output = torch.add(torch.mul(torch.pow(but, -1), a-d), d)

    return output
  end

  function  SyngOneAutoSmart.new(weight)
    weight = weight or torch.rand(4)*2-1

    return autograd.nn.AutoModule('AutoSyngOneFullSimple')(fuAutoSyngOneFull, weight:clone())
  end


  function SyngOneAutoSmart.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
    if teKOSlice:sum() < 1 then
      return torch.zeros(4)
    end

    -- filter out the KO records (not useful for training)
    local teX, teY = syngUtil.getPresent(teInputSlice, teTargetSclice, teKOSlice)

    local a = torch.max(teY)
    local d = torch.min(teY)

    local teH = torch.cdiv(torch.add(-teY, a), torch.add(teY, -d))
    local teB = torch.log(teH)

    -- filter out the records that will cause NaN
    local nRows = teY:size(1)
    local teMask = torch.gt(teB, -math.huge)
    teMask:maskedFill(torch.eq(teB, math.huge), 0)


    local teBValid = syngUtil.getMasked(teB, teMask)
    local teXValid = syngUtil.getMasked(teX, teMask)
    nRows = teBValid:size(1)


    -- contruct teA
    local teAValid = torch.cat(teXValid, torch.ones(nRows, 1), 2)

    -- solve
    local teSubWeights = torch.gels(teBValid, teAValid):squeeze()

    local b = teSubWeights[1]
    local c = teSubWeights[2]

    return torch.Tensor({a, b, c, d})
  end 

  return SyngOneAutoSmart
end

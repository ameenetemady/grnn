local autograd = require 'autograd'
local syngUtil = syngUtil or require('./syngUtil.lua')

local SyngTwoAutoSimpleMult = {}

do
  local fuAutoSyngTwoFull = function(input, weight, bias)

    local a = weight[1]
    local b1 = weight[2]
    local b2 =  weight[3]
    local c =  weight[4]
    local d =  weight[5]
    local rho = weight[6]

    local x1 = torch.narrow(input, 2, 1, 1)
    local x2 = torch.narrow(input, 2, 2, 1)


    local exponent = torch.add(torch.add(torch.add(torch.mul(x1, b1),
                                         torch.mul(x2, b2)),
                               torch.mul(torch.cmul(x1, x2), rho)),
                               c)

    local but = torch.add(torch.exp(exponent), 1)

    local output = torch.add(torch.mul(torch.pow(but, -1), a-d), d)

    return output
  end

  function  SyngTwoAutoSimpleMult.new(weight)
    weight = weight or torch.rand(6)*2-1

    return autograd.nn.AutoModule('AutoSyngTwoFullSimple')(fuAutoSyngTwoFull, weight:clone())
  end


  function SyngTwoAutoSimpleMult.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
--    math.randomseed(os.time())

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
    local teX1mulX2 = torch.cmul(teXValid:narrow(2, 1, 1), teXValid:narrow(2, 2, 1))
    local teAValid = torch.cat({teXValid, teX1mulX2, torch.ones(nRows, 1)}, 2)

    -- solve
    local teSubWeights = torch.gels(teBValid, teAValid):squeeze()

    local b1 = teSubWeights[1]
    local b2 = teSubWeights[2]
    local rho = teSubWeights[3]
    local c = teSubWeights[4]

--    return torch.rand(6)*4
    return torch.Tensor({a, b1, b2, c, d, rho })
    
-- good:  0.4289  3.7169 -0.5862  3.2100  1.4077  4.0701  2.7392
-- good2:-0.0987  3.4431 -0.2491  5.3627  1.4041  1.8228  3.8337

  end 

  return SyngTwoAutoSimpleMult
end

local autograd = require 'autograd'
local syngUtil = syngUtil or require('./syngUtil.lua')

local SyngTwoAutoHill = {}

do
  local fuAutoSyngTwoFull = function(input, weight, bias)

    local a0 = weight[1]
    local a1 = weight[2]
    local a2 =  weight[3]
    local a3 =  weight[4]
    local rho =  weight[5]
    local n1 = weight[6]
    local n2 = weight[7]
    local k1 = weight[8]
    local k2 = weight[9]

    local x1 = torch.narrow(input, 2, 1, 1)
    local x2 = torch.narrow(input, 2, 2, 1)


    local v1 = torch.pow(torch.div(x1, k1), n1)
    local v2 = torch.pow(torch.div(x2, k2), n2)
    local v1v2 = torch.cmul(v1, v2)

    local top = torch.add(torch.mul(v1, a1),
                          torch.add(torch.mul(v2, a2),                         
                                    torch.add(torch.mul(v1v2, a3*rho),         
                                              a0                               
                                              )                                
                                    )                                          
                         )                                                     
                          
                                              

    local but = torch.add(v1,
                          torch.add(v2,                                     
                                    torch.add(torch.mul(v1v2, rho),         
                                              1                             
                                             )                              
                                    )                                       
                          )                                                 
                          

    local output = torch.cdiv(top, but)

    return output
  end

  function  SyngTwoAutoHill.new(weight)
    weight = weight or torch.rand(9)+1
 

    return autograd.nn.AutoModule('AutoSyngTwoFullHill')(fuAutoSyngTwoFull, weight:clone())
  end


  function SyngTwoAutoHill.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
--    math.randomseed(os.time())
--    return torch.rand(9)+1

    local teX, teY = syngUtil.getPresent(teInputSlice, teTargetSclice, teKOSlice)

    -- const values
    local n1 = math.random(1, 6)
    local n2 = math.random(1, 6)

    local k1 = math.random(0.4, 1)
    local k2 = math.random(0.4, 1)

    -- Forming A
    local teX1 = torch.narrow(teX, 2, 1, 1)
    local teX2 = torch.narrow(teX, 2, 2, 1)
    local teV1 = torch.pow(torch.div(teX1, k1), n1)
    local teV2 = torch.pow(torch.div(teX2, k2), n2)
    local teV1v2 = torch.cmul(teV1, teV2)

    local nRows = teY:size(1)
    local teA = torch.cat({torch.ones(nRows, 1), teV1, teV2, teV1v2, torch.cmul(teV1v2, teY) }, 
              2)

    -- Forming B
    local teB = torch.add(teY,
                          torch.add(torch.cmul(teV1, teY),
                                    torch.cmul(teV2, teY)
                                   )
                         )


    -- solve
    local teSubWeights = torch.gels(teB, teA):squeeze()

    local a0 = teSubWeights[1]
    local a1 = teSubWeights[2]
    local a2 = teSubWeights[3]
    local a3 = teSubWeights[4]
    local rho = teSubWeights[5]

    return torch.Tensor({ a0, a1, a2, a3, rho, n1, n2, k1, k2 })
  end 

  return SyngTwoAutoHill
end

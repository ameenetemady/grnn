local autograd = require 'autograd'
local SyngTwoAuto = {}

do
  local fuAutoSyngTwoFull = function(input, w, bias)
    local x1 = torch.select(input, 2, 1)
    local x2 = torch.select(input, 2, 2)

    local y1 = torch.exp(torch.mul(torch.add(torch.mul(x1, -1), w[6]), w[7]))
    local y2 = torch.exp(torch.mul(torch.add(torch.mul(x2, -1), w[8]), w[9]))

    local comb = torch.mul(torch.cmul(y1, y2), w[5])
    local top_comb = torch.mul(comb, w[6])
    local top_y1 = torch.mul(y1, w[2])
    local top_y2 = torch.mul(y2, w[3])

    local top = torch.add(torch.add(torch.add(top_comb, top_y2), top_y1), w[1])

    local but = torch.add(torch.add(torch.add(comb, y2), y1), 1)
    
    local output = torch.cdiv(top, but)


    return output
  end

  function  SyngTwoAuto.new(weight)
    return autograd.nn.AutoModule('AutoSyngTwoFull')(fuAutoSyngTwoFull, weight:clone())
  end

  return SyngTwoAuto
end

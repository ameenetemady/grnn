local syngTwoAuto = require('./SyngTwoAuto.lua')
local syngOneAuto = require('./SyngOneAuto.lua')


local grnnArchUnits = require('./grnnArchUnits.lua')

local grnnArchUnits_test = {}

function grnnArchUnits_test.bGx_t1(teInput, fu, nfArgs)
  local aSeq = nn.Sequential()
  aSeq:add(nn.Narrow(2, 1, nfArgs))
    local mdGx = grnnArchUnits.dGx(nfArgs, fu)
  aSeq:add(mdGx)
  local teOutput = aSeq:forward(teInput)

  print(teOutput)
end

function grnnArchUnits_test.cGx_t1(teInput, fu, nfArgs)
  local aSeq = nn.Sequential()
  aSeq:add(nn.Narrow(2, 1, nfArgs))
    local mcGx = grnnArchUnits.cGx(nfArgs, fu)
  aSeq:add(mcGx)
  local teOutput = aSeq:forward(teInput)

  print(teOutput)

end

function grnnArchUnits_test.all()
  local teInput = torch.Tensor({ {{1,10}, {2, 20}} , 
                                 {{3, 30}, {4, 40}} })
  local fu = function()
    return syngOneAuto.new()
  end
  local nfArgs = 1

  --grnnArchUnits_test.bGx_t1(teInput, fu, nfArgs)
  grnnArchUnits_test.cGx_t1(teInput, fu, nfArgs)
end

grnnArchUnits_test.all()

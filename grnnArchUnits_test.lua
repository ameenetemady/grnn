local syngTwoAuto = syngTwoAuto or require('./SyngTwoAuto.lua')
local syngOneAuto = syngOneAuto or require('./SyngOneAuto.lua')


local grnnArchUnits = grnnArchUnits or require('./grnnArchUnits.lua')

local grnnArchUnits_test = {}

function grnnArchUnits_test.bGx_t1(teInput, fu, nfArgs)
  local aSeq = nn.Sequential()
  aSeq:add(nn.Narrow(2, 1, nfArgs))
    local mdGx = grnnArchUnits.dGx(nfArgs, fu)
  aSeq:add(mdGx)
  local teOutput = aSeq:forward(teInput)

  print(teOutput)
end

function grnnArchUnits_test.cGx_t1(teInput, fu, nfArgs, nGid)
  local aSeq = nn.Sequential()
  aSeq:add(nn.Narrow(2, 1, nfArgs))
    local mcGx = grnnArchUnits.cGx(nfArgs, fu, nGid)
  aSeq:add(mcGx)
  local teOutput = aSeq:forward(teInput)

  print(teOutput)

end

function grnnArchUnits_test.bSeqGx_t1(teInput, fu, nfArgs, nGid)
  local aSeq = nn.Sequential()
  aSeq:add(nn.Narrow(2, 1, nfArgs))
    local mcGx = grnnArchUnits.bSeqGx(nfArgs, fu, nGid)
  aSeq:add(mcGx)
  local teOutput = aSeq:forward(teInput)

  print(teOutput)
end

function grnnArchUnits_test.all()
  local teInput = torch.Tensor({ {{1,10, 100}, {2, 20, 200}} , 
                                 {{3, 30, 400}, {4, 40, 400}} })
  local fu = function()
    return syngOneAuto.new()
  --  return nn.Identity()
  end
  local nfArgs = 1
  local nGid = 2

--  grnnArchUnits_test.bGx_t1(teInput, fu, nfArgs)
--  grnnArchUnits_test.cGx_t1(teInput, fu, nfArgs, nGid)
  grnnArchUnits_test.bSeqGx_t1(teInput, fu, nfArgs, nGid)
end

grnnArchUnits_test.all()

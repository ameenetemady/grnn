local syngTwoAuto = syngTwoAuto or require('./SyngTwoAuto.lua')
local syngOneAuto = syngOneAuto or require('./SyngOneAuto.lua')


local grnnArchUnits = grnnArchUnits or require('./grnnArchUnits.lua')

local grnnArchUnits_test = {}

function grnnArchUnits_test.dGx_t1(teInput, fu, nfArgs)
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
    local mbSeqGx = grnnArchUnits.bSeqGx(nfArgs, fu, nGid)
  aSeq:add(mbSeqGx)
  local teOutput = aSeq:forward(teInput)

  print(teOutput)
end

function grnnArchUnits_test.bGx_t1(teInput, fu, nfArgs,  nGid, nNonTFs)
  local aSeq = nn.Sequential()
  aSeq:add(nn.Narrow(2, 1, nfArgs))
    local mbGx = grnnArchUnits.bGx(nfArgs, fu, nGid, nNonTFs)
  aSeq:add(mbGx)
  local teOutput = aSeq:forward(teInput)

  print(teOutput)
end

function grnnArchUnits_test.aGx_t1(teInput, fu, nGid, nNonTFs, nTFid)
  local maGx = grnnArchUnits.aGx(nfArgs, fu, nGid, nNonTFs, nTFid)
  maGx:add(nn.Narrow(3, 1, 1))
  maGx:add(nn.Squeeze(3))
  local teOutput = maGx:forward(teInput)

  print(teOutput)
end

function grnnArchUnits_test.aGx_t3(teInput, fu2, nfArgs, nGid, nNonTFs, nTFid)
  local maGx = grnnArchUnits.aGx(nfArgs, fu2, nGid, nNonTFs, nTFid)
  maGx:add(nn.Narrow(3, 1, 1))
  maGx:add(nn.Squeeze(3))
  local teOutput = maGx:forward(teInput)

  print(teOutput)
end



function grnnArchUnits_test.aGx_t2(teInput, fu, nGid, nNonTFs, nTFid)
  local maGx = grnnArchUnits.aGx(nfArgs, fu, nGid, nNonTFs, nTFid)
  maGx:add(nn.Narrow(3, 1, 1))
  local teOutput = maGx:forward(teInput)

  local teTarget = teOutput + 0.01

  local criterion = nn.MSECriterion()
  local f = criterion:forward(teOutput, teTarget)

  -- estimate df/dW
  local df_do = criterion:backward(teOutput, teTarget)
  print(df_do)
  local gradInput = maGx:updateGradInput(teInput, df_do)
  print(gradInput)
end

function grnnArchUnits_test.all()
  local teInput = torch.Tensor({ {{1,10, 100}, {2, 20, 200}} , 
                                 {{3, 30, 400}, {4, 40, 400}} })
  local fu = function()
    return syngOneAuto.new()
  --  return nn.Identity()
  end

  local fu2 = function()
    return syngTwoAuto.new()
  end



  local nfArgs = 1
  local nGid = 2
  local nNonTFs = 2
  local nTFid = 1

--  grnnArchUnits_test.dGx_t1(teInput, fu, nfArgs)
--  grnnArchUnits_test.dGx_t1(teInput, fu2, 2)
--  grnnArchUnits_test.cGx_t1(teInput, fu, nfArgs, nGid)
--  grnnArchUnits_test.cGx_t1(teInput, fu2, 2, nGid)
--  grnnArchUnits_test.bSeqGx_t1(teInput, fu, nfArgs, nGid)
--  grnnArchUnits_test.bSeqGx_t1(teInput, fu2, 2, nGid)
--  grnnArchUnits_test.bGx_t1(teInput, fu, nfArgs, nGid, nNonTFs)
--  grnnArchUnits_test.bGx_t1(teInput, fu2, 2, nGid, nNonTFs)
--  grnnArchUnits_test.aGx_t1(teInput, fu, nGid, nNonTFs, nTFid)
--  grnnArchUnits_test.aGx_t2(teInput, fu, nGid, nNonTFs, nTFid)
--  grnnArchUnits_test.aGx_t3(teInput, fu2, 2, nGid, nNonTFs, nTFid)
end

grnnArchUnits_test.all()

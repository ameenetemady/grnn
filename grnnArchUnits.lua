require 'nn'
require('./torchNew/Squeeze.lua')
require('./torchNew/Unsqueeze.lua')
require('./CMulNoParamBatch.lua')

local grnnArchUnits = {}

do
  function grnnArchUnits.dGx(nfArgs, fu)
    local mRes = nn.Sequential()
    mRes:add(nn.Narrow(2, 1, nfArgs))
    mRes:add(nn.Narrow(3, 1, 1))
    mRes:add(nn.Squeeze(3, 3))
    mRes:add(fu())

    return mRes
  end

  function grnnArchUnits.cGx(nfArgs, fu, nGid)
    local mRes = nn.Concat(2)
      local mdGx = grnnArchUnits.dGx(nfArgs, fu)
    mRes:add(mdGx)
      local mToMult = nn.Sequential()
      mToMult:add(nn.Narrow(3, nGid +1, 1))
      mToMult:add(nn.Narrow(2, 1, 1))
    mRes:add(mToMult)

    return mRes
  end

  function grnnArchUnits.bSeqGx(nfArgs, fu, nGid)
    local mRes = nn.Sequential()
      local mcGx = grnnArchUnits.cGx(nfArgs, fu, nGid)
    mRes:add(mcGx)
    mRes:add(nn.CMulNoParamBatch())
    mRes:add(nn.View(-1, 1, 1))
    mRes:add(nn.Contiguous())

    return mRes
  end

  function grnnArchUnits.bGx(nfArgs, fu, nGid, nNonTFs)
    local mRes = nn.Concat(3)
      local mbSeqGx = grnnArchUnits.bSeqGx(nfArgs, fu, nGid)
    mRes:add(mbSeqGx)
      local mNonTFs = nn.Sequential()
      mNonTFs:add(nn.Narrow(3, 2, nNonTFs))
      mNonTFs:add(nn.Narrow(2, 1))
    mRes:add(mNonTFs)

    return mRes
  end

  function grnnArchUnits.aGx(nfArgs, fu, nGid, nNonTFs, nTFid)
    local mRes = nn.Sequential()
    mRes:add(nn.Narrow(2, nTFid, nfArgs))
      local mbGx = grnnArchUnits.bGx(nfArgs, fu, nGid, nNonTFs)
    mRes:add(mbGx)

    return mRes
  end

  return grnnArchUnits
end
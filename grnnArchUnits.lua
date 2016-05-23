require 'nn'
require('./torchNew/Squeeze.lua')
require('./torchNew/Unsqueeze.lua')
require('./CMulNoParamBatch.lua')

local grnnArchUnits = {}

do
  function grnnArchUnits.dGx(nfArgs, fu)
    local mRes = nn.Sequential()
    mRes:add(nn.Narrow(3, 1, nfArgs))
    mRes:add(nn.Squeeze(3, 3))
    mRes:add(fu())

    return mRes
  end

  function grnnArchUnits.cGx(nfArgs, fu, nGid)
    local mRes = nn.Concat(2)
      local mdGx = grnnArchUnits.dGx(nfArgs, fu)
    mRes:add(mdGx)
    mRes:add(nn.Narrow(3, nGid +1, 1))

    return mRes
  end

  function grnnArchUnits.bSeqGx(nfArgs, fu, nGid)
    local mRes = nn.Sequential()
      local mcGx = grnnArchUnits.cGx(nfArgs, fu, nGid)
    mRes:add(mcGx)
    mRes:add(nn.CMulNoParamBatch())

    return mRes
  end

  function grnnArchUnits.bGx(nfArgs, fu, nGid, nNonTFs)
    local mRes = nn.Concat(3)
      local mbSeqGx = grnnArchUnits.bSeqGx(nfArgs, fu, nGid)
      mbSeqGx:add(nn.Unsqueeze(3))
      mbSeqGx:add(nn.Contiguous())
    mRes:add(mbSeqGx)
    mRes:add(nn.Narrow(3, 2, nNonTFs))

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

local grnnArchUnits = {}

do
  function grnnArchUnits.dGx(nfArgs, fu, teWeight)
    local mRes = nn.Sequential()
    mRes:add(nn.Narrow(2, 1, nfArgs))
    mRes:add(nn.Narrow(3, 1, 1))
    mRes:add(nn.Squeeze(3, 3))
    mRes:add(fu(teWeight))

    return mRes
  end

  function grnnArchUnits.cGx(nfArgs, fu, nGid, teWeight)
    local mRes = nn.Concat(2)
      local mdGx = grnnArchUnits.dGx(nfArgs, fu, teWeight)
    mRes:add(mdGx)
      local mToMult = nn.Sequential()
      mToMult:add(nn.Narrow(3, nGid +1, 1))
      mToMult:add(nn.Narrow(2, 1, 1))
    mRes:add(mToMult)

    return mRes
  end

  function grnnArchUnits.bSeqGx(nfArgs, fu, nGid, teWeight)
    local mRes = nn.Sequential()
      local mcGx = grnnArchUnits.cGx(nfArgs, fu, nGid, teWeight)
    mRes:add(mcGx)
    mRes:add(nn.CMulNoParamBatch())
    mRes:add(nn.View(-1, 1, 1))
    mRes:add(nn.Contiguous())

    return mRes
  end

  function grnnArchUnits.bSeqGx_clonable(nfArgs, fu, nGid, teWeight)

    local fuUnitFactory = function(teWeightNew)
      local mUnit = grnnArchUnits.bSeqGx(nfArgs, fu, nGid, teWeightNew) 
      return mUnit
    end

    return ClonableUnit.new(fuUnitFactory, teWeight)
  end

  function grnnArchUnits.bGx(nfArgs, fu, nGid, nNonTFs, teWeight)
    local mRes = nn.Concat(3)
      local mbSeqGx = grnnArchUnits.bSeqGx(nfArgs, fu, nGid, teWeight)
    mRes:add(mbSeqGx)
      local mNonTFs = nn.Sequential()
      mNonTFs:add(nn.Narrow(3, 2, nNonTFs))
      mNonTFs:add(nn.Narrow(2, 1))
    mRes:add(mNonTFs)

    return mRes
  end

-- Input(nfArgs): is the umber of args for this gene
-- Input(fu): initialization function
-- Input(nGid): is the nonTF (that would be knocked out if 0)
-- Input(nTFid): is the column id for the first arg for this gene
-- Input(nNonTFs): total number of nonTF gense (constant in the network)
  function grnnArchUnits.aGx(nfArgs, fu, nGid, nNonTFs, nTFid, teWeight) 
    local mRes = nn.Sequential()
    mRes:add(nn.Narrow(2, nTFid, nfArgs))
      local mbGx = grnnArchUnits.bGx(nfArgs, fu, nGid, nNonTFs, teWeight)
    mRes:add(mbGx)

    return mRes
  end

  return grnnArchUnits
end

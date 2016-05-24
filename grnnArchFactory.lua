local syngTwoAuto = require('./SyngTwoAuto.lua')
local syngOneAuto = require('./SyngOneAuto.lua')
local grnnArchUnits = grnnArchUnits or require('./grnnArchUnits.lua')

do
  local archFactory = {}

  function archFactory.feedforward(param)
    param = param or {}

    local layer_SyngOne = nn.Concat(2)
    layer_SyngOne:add(nn.Identity())
    layer_SyngOne:add(syngOneAuto.new(param.wSyngOne))

    local layer_SyngTwo = nn.Concat(2)
    layer_SyngTwo:add(nn.Identity())
    layer_SyngTwo:add(syngTwoAuto.new(param.wSyngTwo))

    local main = nn.Sequential()
    main:add(layer_SyngOne)
    main:add(layer_SyngTwo)
    main:add(nn.Narrow(2, 2, 2))
    
    return main
  end

  function archFactory.dimA(param)
    param = param or {}


    local mlp_g2 = nn.Sequential()
    mlp_g2:add(nn.Narrow(2, 1, 1))
    mlp_g2:add(syngOneAuto.new(param.g2w))

    local mlp_g3 = syngTwoAuto.new(param.g3w)

    local mlp_g10 = nn.Sequential()
    mlp_g10:add(nn.Narrow(2, 2, 1))
    mlp_g10:add(syngOneAuto.new(param.g10w))

    local main = nn.Concat(2)
    main:add(mlp_g2)
    main:add(mlp_g3)
    main:add(mlp_g10)

    return main

  end

  function archFactory.pri_get_ConcatAbove(mUnit)
    local mRes = nn.Concat(2)
    mRes:add(mUnit)
    mRes:add(nn.Identity())
    return mRes
  end

  function archFactory.net9s(param)
    param = param or {}

    local fuS1 = function()
      return syngOneAuto.new()
    end

    local fuS2 = function()
      return syngTwoAuto.new()
    end


    local nNonTFs = 7

    local mSeqBranch1 = nn.Sequential()
      local mG5 = grnnArchUnits.aGx(1, fuS1, 5, nNonTFs, 2) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    mSeqBranch1:add(mG5)
      local mG6 = grnnArchUnits.aGx(1, fuS1, 6, nNonTFs, 1) --(nfArgs, fu, nGid, nNonTFs, nTFid)
      local mConH6 = archFactory.pri_get_ConcatAbove(mG6)
    mSeqBranch1:add(mConH6)
      local mG9 = grnnArchUnits.aGx(1, fuS1, 7, nNonTFs, 1) --(nfArgs, fu, nGid, nNonTFs, nTFid)
      local mConH9 = archFactory.pri_get_ConcatAbove(mG9)
    mSeqBranch1:add(mConH9)
      local mG1 = grnnArchUnits.aGx(1, fuS1, 1, nNonTFs, 1) --(nfArgs, fu, nGid, nNonTFs, nTFid)
      local mConH1 = archFactory.pri_get_ConcatAbove(mG1)
    mSeqBranch1:add(mConH1)

    -- 5 -> 6 -> 9 -> 1* -> 2

    local mSeqBranch2 = nn.Sequential()
      local mConH7 = nn.Concat(2)
      mConH7:add(nn.Narrow(2, 2, 1))
        local mG3 = grnnArchUnits.aGx(1, fuS1, 3, nNonTFs, 1) --(nfArgs, fu, nGid, nNonTFs, nTFid)
      mConH7:add(mG3)
    mSeqBranch2:add(mConH7)
      local mConH4 = nn.Concat(2)
        local mG4 = grnnArchUnits.aGx(2, fuS2, 4, nNonTFs, 1) --(nfArgs, fu, nGid, nNonTFs, nTFid)
      mConH4:add(mG4)
      mConH4:add(nn.Narrow(2, 2, 1))
    mSeqBranch2:add(mConH4)

    local mConHBranch1And2 = nn.Concat(2)
    mConHBranch1And2:add(mSeqBranch1)
    mConHBranch1And2:add(mSeqBranch2)

    
    local mFinalSeq = nn.Sequential()
    mFinalSeq:add(mConHBranch1And2)
      local mConH2 = nn.Concat(2)
        local mSeqG2 = nn.Sequential()
          local mConG2Input = nn.Concat(2)
          mConG2Input:add(nn.Narrow(2, 1, 1))
          mConG2Input:add(nn.Narrow(2, 5, 1))
        mSeqG2:add(mConG2Input)
          local mG2 = grnnArchUnits.aGx(2, fuS2, 2, nNonTFs, 1) --(nfArgs, fu, nGid, nNonTFs, nTFid)
        mSeqG2:add(mG2)
      mConH2:add(mSeqG2)
      mConH2:add(nn.Identity)
    mFinalSeq:add(mConH2)

    mFinalSeq:add(nn.Narrow(3, 1, 1))
    mFinalSeq:add(nn.Squeeze(3))


    -- todo: need to re-order (use nn.Concat(2) and number of nn.Narrow)

    return mFinalSeq
  end

  return archFactory
end

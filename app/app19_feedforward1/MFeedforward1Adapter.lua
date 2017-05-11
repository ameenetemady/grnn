require('../../requireBaseUnits.lua')

local mSyngTwo = mSyngTwo or require('../../SyngV7.lua')
local mSyngOne =  mSyngTwo
--local mSyngTwo = mSyngTwo or require('../../SyngTwoV7.lua')
--local mSyngOne = mSyngOne or require('../../SyngOneAutoSmart.lua')
local grnnArchUnits = grnnArchUnits or require('../../grnnArchUnits.lua')

local MFeedforward1Adapter, parent = MFeedforward1Adapter or torch.class("MFeedforward1Adapter", "AMNetAdapter")

function MFeedforward1Adapter:__init(taParam, taWeights)
  self.taParam = self.pri_cloneParams(taParam)
  self.taWeights = self.pri_cloneWeights(taWeights)
  self.mNet, self.taFu = self.getNewMNet(self.taWeights)
end

function MFeedforward1Adapter:clone()
  local taWeights = self:pri_getModelWeights()
  return self.new(self.taParam, taWeights)
end

function MFeedforward1Adapter:cloneNoWeight()
  return self.new(self.taParam)
end

-- ***************************
-- ****** Static Methods *****
-- ***************************
function MFeedforward1Adapter.pri_get_ConcatAbove(mUnit)
  local mRes = nn.Concat(2)
  mRes:add(mUnit)
  mRes:add(nn.Identity())
  return mRes
end

function MFeedforward1Adapter.getNewMNet(taWeights)
  taWeights = taWeights or {}

  local fuS1 = function(weight)
    return mSyngOne.new(weight)
  end

  local fuS2 = function(weight)
    return mSyngTwo.new(weight)
  end

  local fuInitS1 = function(teInputSlice, teTargetSclice, teKOSlice)
    return mSyngOne.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
  end

  local fuInitS2 = function(teInputSlice, teTargetSclice, teKOSlice)
    return mSyngTwo.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
  end


  local nNonTFs = 2

  local mSeqFinal = nn.Sequential()

    local mG6 = grnnArchUnits.aGx(1, fuS1, 1, nNonTFs, 1, taWeights.G6) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    local mConH6 = MFeedforward1Adapter.pri_get_ConcatAbove(mG6)
  mSeqFinal:add(mConH6) --d2: 6, 8
    local mG7 = grnnArchUnits.aGx(2, fuS2, 2, nNonTFs, 1, taWeights.G7) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    local mConH7 = MFeedforward1Adapter.pri_get_ConcatAbove(mG7) --d2: 7,6,8
  mSeqFinal:add(mConH7)

  mSeqFinal:add(nn.Narrow(3, 1, 1))
  mSeqFinal:add(nn.Squeeze(3))

  -- current order: 7, 6
    local mReOrder = nn.Concat(2)
      mReOrder:add(nn.Narrow(2, 2, 1))--6
      mReOrder:add(nn.Narrow(2, 1, 1))--7
  mSeqFinal:add(mReOrder)


  local taFu = { G6 = { fu = fuS1, fuInit = fuInitS1, mGx = mG6, taIn={"G8"}},
                 G7 = { fu = fuS2, fuInit = fuInitS2, mGx = mG7, taIn={"G6", "G8"}}
                }

  return mSeqFinal, taFu
end

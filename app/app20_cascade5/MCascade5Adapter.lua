require('../../requireBaseUnits.lua')

local mSyngOne = mSyngOne or require('../../SyngV7.lua')
local grnnArchUnits = grnnArchUnits or require('../../grnnArchUnits.lua')

local MCascade5Adapter, parent = MCascade5Adapter or torch.class("MCascade5Adapter", "AMNetAdapter")

function MCascade5Adapter:__init(taParam, taWeights)
  self.taParam = self.pri_cloneParams(taParam)
  self.taWeights = self.pri_cloneWeights(taWeights)
  self.mNet, self.taFu = self.getNewMNet(self.taWeights)
end

function MCascade5Adapter:clone()
  local taWeights = self:pri_getModelWeights()
  return self.new(self.taParam, taWeights)
end

function MCascade5Adapter:cloneNoWeight()
  return self.new(self.taParam)
end

-- ***************************
-- ****** Static Methods *****
-- ***************************
function MCascade5Adapter.pri_get_ConcatRight(mUnit)
  local mRes = nn.Concat(2)
  mRes:add(nn.Identity())
  mRes:add(mUnit)
  return mRes
end

function MCascade5Adapter.getNewMNet(taWeights)
  taWeights = taWeights or {}

  local fuS1 = function(weight)
    return mSyngOne.new(weight)
  end

  local fuInitS1 = function(teInputSlice, teTargetSclice, teKOSlice)
    return mSyngOne.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
  end

  local nNonTFs = 4

  local mSeqFinal = nn.Sequential()

    local mG2 = grnnArchUnits.aGx(1, fuS1, 1, nNonTFs, 1, taWeights.G2)
--    local mConH2 = MCascade5Adapter.pri_get_ConcatRight(mG2)
  mSeqFinal:add(mG2)

    local mG3 = grnnArchUnits.aGx(1, fuS1, 2, nNonTFs, 1, taWeights.G3)
    local mConH3 = MCascade5Adapter.pri_get_ConcatRight(mG3)
  mSeqFinal:add(mConH3)

    local mG4 = grnnArchUnits.aGx(1, fuS1, 3, nNonTFs, 2, taWeights.G4)
    local mConH4 = MCascade5Adapter.pri_get_ConcatRight(mG4)
  mSeqFinal:add(mConH4)

  ----[[
    local mG5 = grnnArchUnits.aGx(1, fuS1, 4, nNonTFs, 3, taWeights.G5)
    local mConH5 = MCascade5Adapter.pri_get_ConcatRight(mG5)
  mSeqFinal:add(mConH5)


  mSeqFinal:add(nn.Narrow(3, 1, 1))
  mSeqFinal:add(nn.Squeeze(3))
  --]]

  local taFu = { G2 = { fu = fuS1, fuInit = fuInitS1, mGx = mG2, taIn={"G1"}},
                 G3 = { fu = fuS1, fuInit = fuInitS1, mGx = mG3, taIn={"G2"}},
                 G4 = { fu = fuS1, fuInit = fuInitS1, mGx = mG4, taIn={"G3"}},
                 G5 = { fu = fuS1, fuInit = fuInitS1, mGx = mG5, taIn={"G4"}}
                }

  return mSeqFinal, taFu
end

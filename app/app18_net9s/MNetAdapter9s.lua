require('../../requireBaseUnits.lua')

local syngTwoAutoSimple = syngTwoAutoSimple or require('../../SyngTwoAutoSimple.lua')
local syngOneAutoSimple = syngOneAutoSimple or require('../../SyngOneAutoSimple.lua')
local grnnArchUnits = grnnArchUnits or require('../../grnnArchUnits.lua')

local MNetAdapter9s, parent = MNetAdapter9s or torch.class("MNetAdapter9s", "AMNetAdapter")

function MNetAdapter9s:__init(taParam, taWeights)
  self.taParam = self.pri_cloneParams(taParam)
  self.taWeights = self.pri_cloneWeights(taWeights)
  self.mNet, self.taFu = self.getNewMNet(self.taWeights)
end

function MNetAdapter9s:clone()
  local taWeights = self:pri_getModelWeights()
  return self.new(self.taParam, taWeights)
end

function MNetAdapter9s:cloneNoWeight()
  return self.new(self.taParam)
end

-- ***************************
-- ****** Static Methods *****
-- ***************************
function MNetAdapter9s.pri_get_ConcatAbove(mUnit)
  local mRes = nn.Concat(2)
  mRes:add(mUnit)
  mRes:add(nn.Identity())
  return mRes
end

function MNetAdapter9s.getNewMNet(taWeights)
  taWeights = taWeights or {}

  local fuS1 = function(weight)
    return syngOneAutoSimple.new(weight)
  end

  local fuS2 = function(weight)
    return syngTwoAutoSimple.new(weight)
  end

  local fuInitS1 = function(teInputSlice, teTargetSclice, teKOSlice)
    return syngOneAutoSimple.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
  end

  local fuInitS2 = function(teInputSlice, teTargetSclice, teKOSlice)
    return syngTwoAutoSimple.getInitWeights(teInputSlice, teTargetSclice, teKOSlice)
  end


  local nNonTFs = 7

  local mSeqBranch1 = nn.Sequential()
    local mG5 = grnnArchUnits.aGx(1, fuS1, 5, nNonTFs, 2, taWeights.G5) --(nfArgs, fu, nGid, nNonTFs, nTFid)
  mSeqBranch1:add(mG5)
    local mG6 = grnnArchUnits.aGx(1, fuS1, 6, nNonTFs, 1, taWeights.G6) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    local mConH6 = MNetAdapter9s.pri_get_ConcatAbove(mG6) --d2: 5,6
  mSeqBranch1:add(mConH6)
    local mG9 = grnnArchUnits.aGx(1, fuS1, 7, nNonTFs, 1, taWeights.G9) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    local mConH9 = MNetAdapter9s.pri_get_ConcatAbove(mG9) --d2: 9,5,6
  mSeqBranch1:add(mConH9)
    local mG1 = grnnArchUnits.aGx(1, fuS1, 1, nNonTFs, 1, taWeights.G1) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    local mConH1 = MNetAdapter9s.pri_get_ConcatAbove(mG1) --d2: 1,9,5,6
  mSeqBranch1:add(mConH1) -- d2: 1, 9, 5, 6

  -- 5 -> 6 -> 9 -> 1* -> 2

  local mSeqBranch2 = nn.Sequential()
    local mConH7 = nn.Concat(2)
    mConH7:add(nn.Narrow(2, 2, 1))
      local mG3 = grnnArchUnits.aGx(1, fuS1, 3, nNonTFs, 1, taWeights.G3) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    mConH7:add(mG3) --d2: 7,3
  mSeqBranch2:add(mConH7)
    local mConH4 = nn.Concat(2)
      local mG4 = grnnArchUnits.aGx(2, fuS2, 4, nNonTFs, 1, taWeights.G4) --(nfArgs, fu, nGid, nNonTFs, nTFid)
    mConH4:add(mG4)
    mConH4:add(nn.Narrow(2, 2, 1)) -- d2: 4,3
  mSeqBranch2:add(mConH4) -- d2: 4,3

  local mConHBranch1And2 = nn.Concat(2)
  mConHBranch1And2:add(mSeqBranch1)
  mConHBranch1And2:add(mSeqBranch2) -- d2: 1,9,5,6,4,3

  
  local mFinalSeq = nn.Sequential()
  mFinalSeq:add(mConHBranch1And2)
    local mConH2 = nn.Concat(2)
      local mSeqG2 = nn.Sequential()
        local mConG2Input = nn.Concat(2)
        mConG2Input:add(nn.Narrow(2, 1, 1))
        mConG2Input:add(nn.Narrow(2, 5, 1))
      mSeqG2:add(mConG2Input)
        local mG2 = grnnArchUnits.aGx(2, fuS2, 2, nNonTFs, 1, taWeights.G2) --(nfArgs, fu, nGid, nNonTFs, nTFid)
      mSeqG2:add(mG2)
    mConH2:add(mSeqG2)
    mConH2:add(nn.Identity) --d2: 2,1,9,5,6,4,3
  mFinalSeq:add(mConH2)

  mFinalSeq:add(nn.Narrow(3, 1, 1))
  mFinalSeq:add(nn.Squeeze(3))

  -- current order: 2,1,9,5,6,4,3
    local mReOrder = nn.Concat(2)
      mReOrder:add(nn.Narrow(2, 2, 1))--1
      mReOrder:add(nn.Narrow(2, 1, 1))--2
      mReOrder:add(nn.Narrow(2, 7, 1))--3
      mReOrder:add(nn.Narrow(2, 6, 1))--4
      mReOrder:add(nn.Narrow(2, 5, 1))--5
      mReOrder:add(nn.Narrow(2, 4, 1))--6
      mReOrder:add(nn.Narrow(2, 3, 1))--9

  mFinalSeq:add(mReOrder)


  local taFu = { G5 = { fu = fuS1, fuInit = fuInitS1, mGx = mG5, taIn={"G7"}},
                 G6 = { fu = fuS1, fuInit = fuInitS1, mGx = mG6, taIn={"G5"}},
                 G9 = { fu = fuS1, fuInit = fuInitS1, mGx = mG9, taIn={"G6"}},
                 G1 = { fu = fuS1, fuInit = fuInitS1, mGx = mG1, taIn={"G9"}},
                 G3 = { fu = fuS1, fuInit = fuInitS1, mGx = mG3, taIn={"G10"}},
                 G4 = { fu = fuS2, fuInit = fuInitS2, mGx = mG4, taIn={"G7", "G3"}},
                 G2 = { fu = fuS2, fuInit = fuInitS2, mGx = mG2, taIn={"G1", "G4"}}
                }

  return mFinalSeq, taFu
end

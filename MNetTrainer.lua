--
-- MNetTrainer uses and manipulates members of mNetAdapter
--

local MNetTrainer = MNetTrainer or torch.class("MNetTrainer")
local grnnArchUnits = grnnArchUnits or require("./grnnArchUnits.lua")
local myUtil = myUtil or require("../MyCommon/util.lua")

function MNetTrainer:__init(taParam, mNetAdapter)
  self.taParam = taParam
  self.nNonTFs = table.getn(mNetAdapter.taParam.taNonTFNames)
  self.mNetAdapter = mNetAdapter
end

function MNetTrainer:pri_getGeneSlice(strGene)
  local taColnames = self.taParam.taNonTFNames
  local nGid = myUtil.findIndexInArray(taColnames, strGene) -- todo: continue from here ...

  taColnames = self.taParam.taTFNames
end

function MNetTrainer:trainUnit(strGene)
  local taGeneInfo = self.mNetAdapter.taFu[strGene]

  -- Create unit to train
  local nIns = table.getn(taGeneInfo.taIn)
  local nGid = myUtil.findIndexInArray(self.taParam.taNonTFNames) -- this is to specify which gene is knocked out
  local mGx = grnnArchUnits.bSeqGx(nIns, taGeneInfo.fu, nGid, nil) -- left the last argument to nil for random initialization

  -- Create input
  local nRows = taParam.teInput:size(1)
  local teUnitInput = torch.Tensor(nRows, self.nIns, self.nNonTFs + 1)
  for i=1, self.nIns do
    local strCurrGene = taGeneInfo.taIn[i]
    local teSlice = self:pri_getGeneSlice(strCurrGene)
    teUnitInput:narrow(2, i, 1):select(3, 1):copy(teSlice)
  end

  -- Create target
  local teUnitTarget = self:pri_getGeneSlice(strGene)

  -- Train
  local dTrainErr
  dTrainErr, mGxTrained = self.taParam.fuTrainer(mGx, teUnitInput, teUnitTarget)
  print(string.format("trainUnit: train error for %s: %d", strGene, dTrainErr))

  -- Set Weight
  self.mNetAdapter.taWeights[strGene] = mGxTrained:parameters()[1]
  -- g4 = { fu = fuS2, mGx = mG4, taIn={"g7", "g3"}},
--  function grnnArchUnits.bSeqGx(nfArgs, fu, nGid, teWeight)
  --local mG5 = grnnArchUnits.aGx(1, fuS1, 5, nNonTFs, 2, taWeights.g5) --(nfArgs, fu, nGid, nNonTFs, nTFid)
  
end

function MNetTrainer:trainEachUnit()
  for strGene, v in paris(self.mNetAdapter.taFu) do
    self:trainUnit(strGene)
  end
end

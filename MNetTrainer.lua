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
  -- Check if in NonTFs
  local nGid = myUtil.findIndexInArray(self.taParam.taNonTFNames, strGene)
  if nGid ~= nil then
    return self.taParam.teTarget:narrow(2, nGid, 1)
  end

  -- Otherwise it should be in TFs
  nGid = myUtil.findIndexInArray(self.taParam.taTFNames, strGene)
  assert(nGid ~= nil, "Unexpected! Where is it then?")
  return self.taParam.teInput:narrow(2, nGid, 1):select(3, 1)
end

function MNetTrainer:trainUnit(strGene)
  local taGeneInfo = self.mNetAdapter.taFu[strGene]
  local nIns = table.getn(taGeneInfo.taIn)

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

 -- Create unit to train
  local nGid = myUtil.findIndexInArray(self.taParam.taNonTFNames) -- this is to specify which gene is knocked out
  local mGxClonable = grnnArchUnits.bSeqGx_clonable(nIns, taGeneInfo.fu, nGid, nil) -- the last argument is nil for random initialization
 
  -- Train
  local dTrainErr
  dTrainErr, mGxTrained = self.taParam.fuTrainer(mGxClonable, teUnitInput, teUnitTarget)
  print(string.format("trainUnit: train error for %s: %d", strGene, dTrainErr))

  -- Set Weight
  self.mNetAdapter.taWeights[strGene] = mGxTrained:parameters()[1]
end

function MNetTrainer:trainEachUnit()
  for strGene, v in paris(self.mNetAdapter.taFu) do
    self:trainUnit(strGene)
  end

  return self:mNetAdapter:clone()
end

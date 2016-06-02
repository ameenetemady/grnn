--
-- MNetTrainer uses and manipulates members of mNetAdapter
--

local MNetTrainer = MNetTrainer or torch.class("MNetTrainer")
local grnnArchUnits = grnnArchUnits or require("./grnnArchUnits.lua")
local myUtil = myUtil or require("../MyCommon/util.lua")

function MNetTrainer:__init(taParam, mNetAdapter)
  self.taParam = taParam
  self.nTargets = table.getn(mNetAdapter.taParam.taTargetNames)
  self.mNetAdapter = mNetAdapter
end

function MNetTrainer:pri_getGeneSlice(strGene)
  -- Check if in Targets
  local nGid = myUtil.findIndexInArray(self.mNetAdapter.taParam.taTargetNames, strGene)
  if nGid ~= nil then
    return self.taParam.teTarget:narrow(2, nGid, 1)
  end

  -- Otherwise it should be in TFs
  nGid = myUtil.findIndexInArray(self.mNetAdapter.taParam.taTFNames, strGene)
  assert(nGid ~= nil, "Unexpected! Where is it then?")
  return self.taParam.teInput:narrow(2, nGid, 1):select(3, 1)
end

function MNetTrainer:pri_getKOSlice()
  return self.taParam.teInput:narrow(3, 2, self.nTargets):narrow(2, 1, 1)
end

function MNetTrainer:trainUnit(strGene)
  local taGeneInfo = self.mNetAdapter.taFu[strGene]
  local nIns = table.getn(taGeneInfo.taIn)

    -- Create input
  local nRows = self.taParam.teInput:size(1)
  local teUnitInput = torch.Tensor(nRows, nIns, self.nTargets + 1)
  for i=1, nIns do
    local strCurrGene = taGeneInfo.taIn[i]
    local teSlice = self:pri_getGeneSlice(strCurrGene)
    teUnitInput:narrow(2, i, 1):select(3, 1):copy(teSlice)

    local teKOSlice = self:pri_getKOSlice()
    teUnitInput:narrow(2, i, 1):narrow(3, 2, self.nTargets):copy(teKOSlice)
    
  end

  -- Create target
  local teUnitTarget = self:pri_getGeneSlice(strGene)

 -- Create unit to train
  local nGid = myUtil.findIndexInArray(self.mNetAdapter.taParam.taTargetNames, strGene) -- this is to specify which gene is knocked out
  local mGxClonable = grnnArchUnits.bSeqGx_clonable(nIns, taGeneInfo.fu, nGid, nil) -- the last argument is nil for random initialization
 
  -- Train
  local dTrainErr
  dTrainErr, mGxTrained = self.taParam.fuTrainer(mGxClonable, teUnitInput, teUnitTarget)
  local dTestErr = self.taParam.fuTester(mGxClonable:getRaw(), teUnitInput, teUnitTarget)
  print(string.format("trainUnit: error for %s: %f", strGene, dTestErr))

  -- Set Weight
  self.mNetAdapter:setWeight(strGene, mGxTrained:getRaw():parameters()[1])
end

function MNetTrainer:trainEachUnit()
  for strGene, v in pairs(self.mNetAdapter.taFu) do
    self:trainUnit(strGene)
  end

  return self.mNetAdapter:clone()
end

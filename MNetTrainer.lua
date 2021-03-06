--
-- MNetTrainer uses and manipulates members of mNetAdapter
--

local MNetTrainer = MNetTrainer or torch.class("MNetTrainer")
local grnnArchUnits = grnnArchUnits or require("./grnnArchUnits.lua")
local myUtil = myUtil or require("../MyCommon/util.lua")

function MNetTrainer:__init(taParam, mNetAdapter)
  self.taParam = taParam
  self.nTargets = #mNetAdapter.taParam.taTargetNames
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
  assert(nGid ~= nil, string.format("Unexpected! Where is \"%s\" then?", strGene))
  return self.taParam.teInput:narrow(2, nGid, 1):select(3, 1)
end

function MNetTrainer:pri_getKOSlice()
  return self.taParam.teInput:narrow(3, 2, self.nTargets):narrow(2, 1, 1)
end

function MNetTrainer:trainUnit(strGene)
  local taGeneInfo = self.mNetAdapter.taFu[strGene]
  local nIns = #taGeneInfo.taIn

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
  local nGid = myUtil.findIndexInArray(self.mNetAdapter.taParam.taTargetNames, strGene) -- this is to specify which gene is knocked out (is this comment correct?!)
  assert(type(nGid) == "number" and nGid > 0, string.format("unExpected nGid:%d !", nGid))
  local mGxClonable = grnnArchUnits.bSeqGx_clonable(nIns, taGeneInfo.fu, nGid, 
                                                    self:pri_getInitWeights(teUnitInput, teUnitTarget, taGeneInfo.fuInit, nGid)) 
 
  -- Train
  local dTrainErr, mGxTrained
--  if strGene ~= "G7" then --ToDo: just for testing ...
    dTrainErr, mGxTrained = self.taParam.fuTrainer(mGxClonable, teUnitInput, teUnitTarget, self.taParam.taFuTrainerParams)
--  else
--    mGxTrained = mGxClonable:clone()
--  end

  local dTestErr = self.taParam.fuTester(mGxClonable:getRaw(), teUnitInput, teUnitTarget)
  print(string.format("trainUnit: error for %s: %f", strGene, dTestErr))

  -- Set Weight
  self.mNetAdapter:setWeight(strGene, mGxTrained:getRaw():parameters())
end

function MNetTrainer:pri_getInitWeights(teUnitInput, teUnitTarget, fuInit, nGid)
  local teUnitInputSlice = teUnitInput:select(3, 1, 1)
  local teUnitKOSlice = teUnitInput:narrow(2, 1, 1):narrow(3, 1 + nGid, 1):select(3, 1)
  
  return fuInit(teUnitInputSlice, teUnitTarget, teUnitKOSlice)
end

function MNetTrainer:trainEachUnit()
  for strGene, v in pairs(self.mNetAdapter.taFu) do
    self:trainUnit(strGene)
  end

  self.mNetAdapter:reload()
  
  
  local dTestErr = self.taParam.fuTester(self.mNetAdapter:getRaw(), self.taParam.teInput, self.taParam.teTarget)
  return dTestErr
end

function MNetTrainer:reload_taWeights()
  for strGene, taGeneInfo in pairs(self.mNetAdapter.taFu) do
    self.mNetAdapter:setWeight(strGene, taGeneInfo.mGx:parameters())
  end
end

function MNetTrainer:trainTogether()
    local dTrainErr
    dTrainErr, self.mNetAdapter = self.taParam.fuTrainer(self.mNetAdapter, self.taParam.teInput, self.taParam.teTarget, self.taParam.taFuTrainerParams)
    
    self:reload_taWeights()
    return dTrainErr, self.mNetAdapter
end


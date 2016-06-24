local dataLoad = dataLoad or require("../MyCommon/dataLoad.lua")
local KFoldRunner = torch.class("KFoldRunner")

function KFoldRunner:__init(taParam, fuFoldRunFactory)
  self.nFolds = taParam.nFolds
  self.nSeeds = taParam.nSeeds
  self.teInput = taParam.teInput
  self.teTarget = taParam.teTarget
  self.mNetAdapter = taParam.mNetAdapter
  self.fuTrainer = taParam.fuTrainer
  self.taFuTrainerParams = taParam.taFuTrainerParams
  self.fuTester = taParam.fuTester

  self.fuFoldRunFactory = fuFoldRunFactory

  self.nNextFoldId = 1
  self.taRunners = {}
end

function KFoldRunner:hasMore()
  return self.nNextFoldId <= self.nFolds
end

function KFoldRunner:pri_getFold(teInput, nFoldId)
  local nRows = teInput:size(1)

  -- test:
  local teIdxAll = torch.linspace(1, nRows, nRows)
  local teMaskDim1 = torch.mod(teIdxAll, self.nFolds):eq(torch.Tensor(nRows):fill(nFoldId-1))

  local teTest = dataLoad.getMaskedSelect(teInput, teMaskDim1)

  -- train:
  teIdxAll = torch.linspace(1, nRows, nRows)
  teMaskDim1 = torch.mod(teIdxAll, self.nFolds):ne(torch.Tensor(nRows):fill(nFoldId-1))

  local teTrain = dataLoad.getMaskedSelect(teInput, teMaskDim1)

  return teTrain, teTest
end

function KFoldRunner:pri_getNewFoldRun(taRunParam)
  if self.fuFoldRunFactory == nil then
    return FoldRun.new(taRunParam)
  end

  return self.fuFoldRunFactory(taRunParam)
end

function KFoldRunner:getNext()
  if not self:hasMore() then
    return nil
  end

  local teInput_train, teInput_test = self:pri_getFold(self.teInput, self.nNextFoldId)
  local teTarget_train, teTarget_test = self:pri_getFold(self.teTarget, self.nNextFoldId)

  local taRunParam = {
    taTrain = { teInput_train, teTarget_train },
    taTest = { teInput_test, teTarget_test },
    nSeeds = self.nSeeds,
    mNetAdapter = self.mNetAdapter,
    fuTrainer = self.fuTrainer,
    taFuTrainerParams = self.taFuTrainerParams,
    fuTester = self.fuTester
  }

  local foldRun = self:pri_getNewFoldRun(taRunParam)
  table.insert(self.taRunners, foldRun)

  self.nNextFoldId = self.nNextFoldId + 1

  return foldRun
end

function KFoldRunner:getAggrSummaryTable()
  local taAggrSummary = {}

  for i=1, self.nFolds do
    if self.taRunners[i] ~= nil then
      local taCurrSummary = self.taRunners[i]:getSummaryTable()
      table.insert(taAggrSummary, taCurrSummary)
    end
  end

  return taAggrSummary
end

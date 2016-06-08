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

function KFoldRunner:pri_getMaskedSelect(teInput, teMaskDim1)
  local nRows = teInput:size(1)

  -- expand tensor for maskedCopy
  local teMaskSize = teInput:size():fill(1)
  teMaskSize[1] = nRows
  local teMask = torch.ByteTensor(teMaskSize)

  local teInputMasked = nil
  if teInput:dim() == 2 then
    teMask:select(2, 1):copy(teMaskDim1)
    teMask = teMask:expandAs(teInput)
    teInputMasked = teInput:maskedSelect(teMask)
    teInputMasked:resize(teMaskDim1:sum(), teInput:size(2))

  elseif teInput:dim(2) == 3 then
    teMask:select(3, 1):select(2, 1):copy(teMaskDim1)
    teMask = teMask:expandAs(teInput)
    teInputMasked = teInput:maskedSelect(teMask)
    teInputMasked:resize(teMaskDim1:sum(), teInput:size(2), teInput:size(3))

  else
    error(string.format("nDim = %d not supported!", teInput:dim()))
  end

  return teInputMasked
end

function KFoldRunner:pri_getFold(teInput, nFoldId)
  local nRows = teInput:size(1)

  -- test:
  local teIdxAll = torch.linspace(1, nRows, nRows)
  local teMaskDim1 = torch.mod(teIdxAll, self.nFolds):eq(torch.Tensor(nRows):fill(nFoldId-1))

  local teTest = self:pri_getMaskedSelect(teInput, teMaskDim1)

  -- train:
  teIdxAll = torch.linspace(1, nRows, nRows)
  teMaskDim1 = torch.mod(teIdxAll, self.nFolds):ne(torch.Tensor(nRows):fill(nFoldId-1))

  local teTrain = self:pri_getMaskedSelect(teInput, teMaskDim1)

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
    local taCurrSummary = self.taRunners[i]:getSummaryTable()
    table.insert(taAggrSummary, taCurrSummary)
  end

  return taAggrSummary
end

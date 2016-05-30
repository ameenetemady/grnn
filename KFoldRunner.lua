local KFoldRunner = torch.class("KFoldRunner")

function KFoldRunner:__init(taParam)
  self.nFolds = taParam.nFolds
  self.nSeeds = taParam.nSeeds
  self.teInput = taParam.teInput
  self.teTarget = taParam.teTarget
  self.fuArchGen = taParam.fuArchGen
  self.fuTrainer = taParam.fuTrainer
  self.fuTester = taParam.fuTester

  self.nNextFoldId = 1
  self.taRunners = {}
end

function KFoldRunner:hasMore()
  return self.nNextFoldId <= self.nFolds
end

function self:pri_getMaskedSelect(teInput, teMaskDim1)
  local nRows = teInput:size(1)

  -- expand tensor for maskedCopy
  local teMaskSize = torch.ones(teInput:size())
  teMaskSize[1] = nRows
  local teMask = torch.ByteTesor(teMaskSize)
  teMask:select(1):copy(teMaskDim1)
  teMask:expandAs(teInput)

  local teInputMasked = teInput:maskedSelect(teMask)
  return teInputMasked
end

function self:pri_getFold(teInput, nFoldId)
  local nRows = teInput:size(1)

  -- test:
  local teIdxAll = torch.linspace(1, nRows, nRows)
  local teMaskDim1 = torch.mod(teIdxAll, self.nFolds):eq(torch.Tensor(nSize):fill(nFoldId-1))

  local teTest = self:pri_getMaskedSelect(teInput, teMaskDim1)

  -- train:
  teIdxAll = torch.linspace(1, nRows, nRows)
  teMaskDim1 = torch.mod(teIdxAll, self.nFolds):ne(torch.Tensor(nSize):fill(nFoldId-1))

  local teTrain = self:pri_getMaskedSelect(teInput, teMaskDim1)

  return teTrain, teTest
end

function KFoldRunner:getNext()
  if not self:hasMore() then
    return nil
  end

  local teInput_train, teInput_test = self:pri_getFold(self.teInput, self.nNextFoldId)
  local teTarget_train, teTarget_test = self:pri_getFold(self.teTarget, self.nNextFoldId)

  local taRunParam = {
    taTrain = { teInput_train, teTarget_train }
    taTest = { teInput_test, teInput_test },
    nSeeds = taParam.nSeeds,
    fuArchGen = self.fuArchGen,
    fuTrainer = self.fuTrainer,
    fuTester = self.fuTester
  }

  local foldRun = FoldRun.new(taRunParam)
  table.insert(self.taRunners, foldRun)

  self.nNextFoldId = self.nNextFoldId + 1

  return foldRun
end

function KFoldRunner:getAggrSummaryTable()
  local taAggrSummary = {}

  for i=1, self.nFolds do
    local taCurrSummary = taRunners[i]:getSummaryTable()
  end

  return taAggrSummary
end

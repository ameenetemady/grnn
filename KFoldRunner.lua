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

function KFoldRunner:getNext()
  if ~self:hasMore() then
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

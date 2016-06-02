local myUtil = myUtil or require('../MyCommon/util.lua')
local FoldRun = torch.class("FoldRun")

function FoldRun:__init(taParam)
    self.taTrain = taParam.taTrain
    self.taTest = taParam.taTest
    self.nSeeds = taParam.nSeeds
    self.mNetAdapter = taParam.mNetAdapter
    self.fuTrainer = taParam.fuTrainer
    self.fuTester = taParam.fuTester

    self.dTrainErr = math.huge
    self.taTestResult = nil
end

function FoldRun:Run()
  local dBestTrainErr = math.huge
  local mNetAdapterBest = self.mNetAdapter:clone()

  -- train (with all seeds)
  for i=1, self.nSeeds do
    torch.manualSeed(i)

    local teInput_train = self.taTrain[1]
    local teTarget_train = self.taTrain[2]

    print("i: " .. i)
    local dTrainErr = math.huge
    dTrainErr, self.mNetAdapter = self.fuTrainer(self.mNetAdapter:cloneNoWeight(), teInput_train, teTarget_train)

    if dTrainErr < dBestTrainErr then
      dBestTrainErr = dTrainErr
      mNetAdapterBest = self.mNetAdapter:clone()
    end
  end
  assert(dBestTrainErr < math.huge, "training didn't succeed as dBestTrainErr is still mat.huge!")
  self.dTrainErr = dBestTrainErr

  -- test
  self.taTestResult = self.fuTester(mNetAdapterBest:getRaw(), self.taTest[1], self.taTest[2])
end

function FoldRun:getSummaryTable()
  return {dTrainErr = self.dTrainErr,
          taTestResult = self.taTestResult}

end

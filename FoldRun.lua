local FoldRun = torch.class("FoldRun")

function FoldRun:__init(taParam)
    self.taTrain = taParam.taTrain
    self.taTest = taParam.taTest
    self.nSeeds = taParam.nSeeds
    self.fuArchGen = taParam.fuArchGen
    self.fuTrainer = taParam.fuTrainer
    self.fuTester = taParam.fuTester

    self.dTrainErr = math.huge
    self.taTestResult = nil
end

function FoldRun:Run()
  local dBestTrainErr = math.huge
  local mBestNet = nil
  local mNet, mNetInfo = self.fuArchGen()

  -- train (with all seeds)
  for i=1, self.nSeeds do
    torch.manualSeed(i)
    mNet:reset()

    local teInput_train = self.taTrain[1]
    local teTarget_train = self.taTrain[2]
    local dTrainErr = self.fuTrainer(mNet, teInput_train, teTarget_train, mNetInfo)

    if dTrainErr < dBestTrainErr then
      mBestNet = myUtil.getClone(mNet)
    end
  end
  assert(dBestTrainErr < math.huge, "training didn't succeed as dBestTrainErr is still mat.huge!")
  self.dTrainErr = dBestTrainErr

  -- test
  self.taTestResult = self.fuTester(mBestNet, self.taTest[1], self.taTest[2])
end

function FoldRun:getSummaryTable()
  return {dTrainErr = self.dTrainErr,
          taTestResult = self.taTestResult}

end

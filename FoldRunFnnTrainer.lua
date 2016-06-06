require("./FnnTrainer.lua")

local FoldRunFnnTrainer = torch.class("FoldRunFnnTrainer", "FoldRun")

function FoldRunFnnTrainer :__init(taParam)
    self.taTrain = taParam.taTrain
    self.taTest = taParam.taTest
    self.mNetAdapter = taParam.mNetAdapter
    self.fuTrainer = taParam.fuTrainer
    self.taFuTrainerParams = taParam.taFuTrainerParams
    self.fuTester = taParam.fuTester
    self.nSeeds = taParam.nSeeds

    self.dTrainErr = math.huge
    self.taTestResult = nil
end

function FoldRunFnnTrainer:Run()

  --train-init
  local taMNetTrainerParam = { teInput = self.taTrain[1],
                               teTarget = self.taTrain[2],
                               fuTrainer = self.fuTrainer,
                               taFuTrainerParams = self.taFuTrainerParams,
                               fuTester = self.fuTester
                             }
  local dBestTrainErr = math.huge
  local mNetAdapterBest = self.mNetAdapter:clone()

  -- train (with all seeds) 
  for i=1, self.nSeeds do
    torch.manualSeed(i)

    print("i: " .. i)
    local dTrainErr = math.huge
    local trainer = FnnTrainer.new(taMNetTrainerParam , self.mNetAdapter:cloneNoWeight())
    dTrainErr, self.mNetAdapter = trainer:trainTogether()

    if dTrainErr < dBestTrainErr then
      dBestTrainErr = dTrainErr
      mNetAdapterBest = self.mNetAdapter:clone()
    end
  end
  assert(dBestTrainErr < math.huge, "training didn't succeed as dBestTrainErr is still mat.huge!")
  self.dTrainErr = dBestTrainErr
  -- end train

  -- test
  self.taTestResult = self.fuTester(mNetAdapterBest:getRaw(), self.taTest[1], self.taTest[2])
end


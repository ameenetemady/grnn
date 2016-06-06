require("./MNetTrainer.lua")

local FoldRunMNetTrainer = torch.class("FoldRunMNetTrainer", "FoldRun")

function FoldRunMNetTrainer:__init(taParam)
    self.taTrain = taParam.taTrain
    self.taTest = taParam.taTest
    self.mNetAdapter = taParam.mNetAdapter
    self.fuTrainer = taParam.fuTrainer
    self.fuTester = taParam.fuTester

    self.dTrainErr = math.huge
    self.taTestResult = nil
end

function FoldRunMNetTrainer:Run()

  --train
  local taMNetTrainerParam = { teInput = self.taTrain[1],
                               teTarget = self.taTrain[2],
                               fuTrainer = self.fuTrainer,
                               fuTester = self.fuTester
                             }

  local mNetTrainer = MNetTrainer.new(taMNetTrainerParam, self.mNetAdapter)
  mNetTrainer:trainEachUnit()
  self.dTrainErr, self.mNetAdapter = mNetTrainer:trainTogether()

  -- test
  self.taTestResult = self.fuTester(self.mNetAdapter:getRaw(), self.taTest[1], self.taTest[2])
end


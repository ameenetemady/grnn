local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')
require('../../FoldRun.lua')
require('../../KFoldRunner.lua')
require('./MNetAdapter9s.lua')



require("../../MNetTrainer.lua")

local MNetTrainer_test = {}

function MNetTrainer_test.pri_getGeneSlice_test1()

  local exprSettings = lSettings.getExprSetting("d_1_small")
  local teInput, taTFNames, taKONames = cDataLoad.load3dInput(exprSettings)
  teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings)

  local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

  local mNetAdapter = MNetAdapter9s.new(taNetParam)
  local fuTrainer = trainerPool.trainGrnnMNetAdapter
  local fuTester = testerPool.getMSE

  local mNet = mNetAdapter:getRaw()
  local teTarget = mNet:forward(teInput)
  mNetAdapter:test_addToWeights(0.1)

  local taMNetTrainerParam = { teInput = teInput,
                               teTarget = teTarget,
                               fuTrainer = fuTrainer
                             }

  local mNetTrainer = MNetTrainer.new(taMNetTrainerParam, mNetAdapter)
  local teSlice = mNetTrainer:pri_getGeneSlice("G2")
  print(teSlice)
end

function MNetTrainer_test.trainUnit_test1()

  local exprSettings = lSettings.getExprSetting("d_1_small")
  local teInput, taTFNames, taKONames = cDataLoad.load3dInput(exprSettings)
  local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings)

  local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

  local mNetAdapter = MNetAdapter9s.new(taNetParam)
  local fuTrainer = trainerPool.trainGrnnMNetAdapter
  local fuTester = testerPool.getMSE

  local mNet = mNetAdapter:getRaw()

  local dTestErr = fuTester(mNetAdapter:getRaw(), teInput, teTarget)
  print("Before MSE error", dTestErr)

  local taMNetTrainerParam = { teInput = teInput,
                               teTarget = teTarget,
                               fuTrainer = fuTrainer,
                               fuTester = fuTester
                             }

  local mNetTrainer = MNetTrainer.new(taMNetTrainerParam, mNetAdapter)
  mNetTrainer:trainUnit("G3")
  mNetAdapter:reload()

  dTestErr = fuTester(mNetAdapter:getRaw(), teInput, teTarget)
  print("After MSE error", dTestErr)
end

function MNetTrainer_test.trainEachUnit_test1()

  local exprSettings = lSettings.getExprSetting("d_2")
  local teInput, taTFNames, taKONames = cDataLoad.load3dInput(exprSettings)
  local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings)

  local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

  local mNetAdapter = MNetAdapter9s.new(taNetParam)
  local fuTrainer = trainerPool.trainGrnnMNetAdapter
  local fuTester = testerPool.getMSE

  local mNet = mNetAdapter:getRaw()

  local dTestErr = fuTester(mNetAdapter:getRaw(), teInput, teTarget)
  print("MSE error:Initial", dTestErr)

  local taMNetTrainerParam = { teInput = teInput,
                               teTarget = teTarget,
                               fuTrainer = fuTrainer,
                               fuTester = fuTester
                             }

  local mNetTrainer = MNetTrainer.new(taMNetTrainerParam, mNetAdapter)
  mNetTrainer:trainEachUnit()

  dTestErr = fuTester(mNetAdapter:getRaw(), teInput, teTarget)
  print("MSE error:trainEachUnit", dTestErr)

  local dTrainErr
  dTrainErr, mNetAdapter = mNetTrainer:trainTogether()
  dTestErr = fuTester(mNetAdapter:getRaw(), teInput, teTarget)
  print("MSE error:trainTogether", dTestErr)

end


function MNetTrainer_test.all()
--  MNetTrainer_test.pri_getGeneSlice_test1()
--  MNetTrainer_test.trainUnit_test1()
  MNetTrainer_test.trainEachUnit_test1()
end

MNetTrainer_test.all()

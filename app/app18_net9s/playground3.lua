local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
--local grnnArchFactory = grnnArchFactory or require('../../grnnArchFactory.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('./cDataLoad.lua')
require('../../FoldRun.lua')
require('../../KFoldRunner.lua')
require('./MNetAdapter9s.lua')


local exprSettings = lSettings.getExprSetting("d_1_small")
local teInput, taTFNames, taKONames = cDataLoad.load3dInput(exprSettings)
--local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings)

local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

local mNetAdapter = MNetAdapter9s.new(taNetParam)
local fuTrainer = trainerPool.trainGrnn3dMNetAdapter
local fuTester = testerPool.getMSE


  local mNet = mNetAdapter:getRaw()
  local teTarget = mNet:forward(teInput)

  mNetAdapter:test_addToWeights(0.1)
  local taTestResult = fuTester(mNetAdapter:getRaw(), teInput, teTarget)
  print("MSE Before: " .. taTestResult)

  local dTrainErr = math.huge
  dTrainErr, mNetAdapter = fuTrainer(mNetAdapter:clone(), teInput, teTarget)

  taTestResult = fuTester(mNetAdapter:getRaw(), teInput, teTarget)

  print("MSE After: " .. taTestResult)



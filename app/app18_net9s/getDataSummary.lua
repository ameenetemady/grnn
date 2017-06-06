local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local myUtil = myUtil or require('../../../MyCommon/util.lua')
require('../../FoldRun.lua')
require('../../FoldRunFnnTrainer.lua')
require('../../KFoldRunner.lua')
require('../../FnnAdapter.lua')
require('../common/CDataLoader.lua')

function runExperiment(strExprName, isNoise)
  local exprSettings = lSettings.getExprSetting(strExprName)
  local dDist = isNoise and (2) or 1.5 
  local dataLoader = CDataLoader.new(exprSettings, isNoise, true, dDist)

  --load
  local teInput, taTFNames, taKONames = dataLoader:load2dInput()
  local teTarget, taTargetNames = dataLoader:loadTarget()

  local nRows = teTarget:size(1)
  print(string.format("Number of rows: %d *************************", nRows))

end

local isNoise = myUtil.getBoolFromStr(arg[1])
nMaxExprId = 20
for nExprId=1, nMaxExprId do
  local strExprName = string.format("d_%d", nExprId)
  print(string.format("********** Experiemnt %s ***********", strExprName))

  runExperiment(strExprName, isNoise, taFnnParam)
end


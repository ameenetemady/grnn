local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')
local myUtil = myUtil or require('../../../MyCommon/util.lua')
require('../../FoldRun.lua')
require('../../FoldRunMNetTrainer.lua')
require('../../KFoldRunner.lua')
require('./MFeedforward1Adapter.lua')

function runExperiment(strExprName, isNoise)
  local exprSettings = lSettings.getExprSetting(strExprName)
  local teInput, taTFNames, taKONames = cDataLoad.load3dInput(exprSettings, isNoise)
  local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings, isNoise)

  local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

  local fuFoldRunFactory = function(taFoldRunParam)
    return FoldRunMNetTrainer.new(taFoldRunParam)
  end

  local taParam = { 
    nFolds = 10, --10
    teInput = teInput, 
    teTarget = teTarget, 
    mNetAdapter = MFeedforward1Adapter.new(taNetParam),
    fuTrainer = trainerPool.trainGrnnMNetAdapter,
    taFuTrainerParams = { nMaxIteration = 10},--10
    fuTester = testerPool.getMSE}

    local kFoldRunner = KFoldRunner.new(taParam, fuFoldRunFactory)
    while kFoldRunner:hasMore() do
      local foldRun = kFoldRunner:getNext()
      foldRun:Run()
      print(foldRun:getSummaryTable())
    end

  return kFoldRunner:getAggrSummaryTable()
end

function getResultFilename(strExprName, isNoise)
  local strNoise = isNoise and "_noise" or ""
  return string.format("result/grnn_%s%s.table", strExprName, strNoise)
end


local isNoise = myUtil.getBoolFromStr(arg[1])
local nMaxExprId = 100 --100
for nExprId=1, nMaxExprId do
  local strExprName = string.format("d_%d", nExprId)
  print(string.format("********** Experiemnt %s ***********", strExprName))

  local taExprResult = runExperiment(strExprName, isNoise)
  local strResultFilename = getResultFilename(strExprName, isNoise)
  torch.save(strResultFilename, taExprResult, "ascii")
end


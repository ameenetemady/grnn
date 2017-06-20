require 'torch'
local lfs = require 'lfs'
package.path = string.format("%s/?.lua;", lfs.currentdir()) .. package.path
local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('lSettings')
print(lSettings)
local myUtil = myUtil or require('../../../MyCommon/util.lua')
require('../../FoldRun.lua')
require('../../FoldRunMNetTrainer.lua')
require('../../KFoldRunner.lua')
require('../common/CDataLoader.lua')

function runExperiment(strExprName, isNoise, dMinDist)
  local exprSettings = lSettings.getExprSetting(strExprName)
  local dataLoader = CDataLoader.new(exprSettings, isNoise, true, dMinDist)

  local teInput, taTFNames, taKONames = dataLoader:load3dInput()
  local teTarget, taTargetNames = dataLoader:loadTarget()

  local nRows = teTarget:size(1)
  print(string.format("Number of rows: %d *************************", nRows))


  local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

  local fuFoldRunFactory = function(taFoldRunParam)
    return FoldRunMNetTrainer.new(taFoldRunParam)
  end

  local taParam = { 
    nFolds = 5, --10
    teInput = teInput, 
    teTarget = teTarget, 
    mNetAdapter = lSettings.getNewMNet(taNetParam),
    fuTrainer = trainerPool.trainGrnnMNetAdapter,
    taFuTrainerParams = { nMaxIteration = 10, strOptimMethod = "CG"},--10
    fuTester = testerPool.getMSE}

  if nRows < taParam.nFolds then -- cannot have less rows than the number of folds
    return nil
  end

    local kFoldRunner = KFoldRunner.new(taParam, fuFoldRunFactory)
    while kFoldRunner:hasMore() do
      local foldRun = kFoldRunner:getNext()
      foldRun:Run()
      print(foldRun:getSummaryTable())
    end

  return kFoldRunner:getAggrSummaryTable()
end

function getResultFilename(strExprName, isNoise, dMinDist)
  local strNoise = isNoise and "_noise" or ""
  return string.format("result/grnn_%s%s_%.3f.table", strExprName, strNoise, dMinDist)
end

local isNoise = myUtil.getBoolFromStr(arg[1])
local dMinDist = arg[2] == nil and 1 or tonumber(arg[2])
local nExprId = tonumber(arg[3])

local strExprName = string.format("d_%d", nExprId)
print(string.format("********** Experiemnt %s ***********", strExprName))

local taExprResult = runExperiment(strExprName, isNoise, dMinDist)
assert(taExprResult, "No result, abort!")
local strResultFilename = getResultFilename(strExprName, isNoise, dMinDist)
torch.save(strResultFilename, taExprResult, "ascii")

collectgarbage()

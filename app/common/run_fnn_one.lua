local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('lSettings.lua')
local myUtil = myUtil or require('../../../MyCommon/util.lua')
require('../../FoldRun.lua')
require('../../FoldRunFnnTrainer.lua')
require('../../KFoldRunner.lua')
require('../../FnnAdapter.lua')
require('../common/CDataLoader.lua')

function runExperiment(strExprName, isNoise, taFnnParam, dMinDist)
  local exprSettings = lSettings.getExprSetting(strExprName)
  local dataLoader = CDataLoader.new(exprSettings, isNoise, true, dMinDist)

  --load
  local teInput, taTFNames, taKONames = dataLoader:load2dInput()
  local teTarget, taTargetNames = dataLoader:loadTarget()

  local nRows = teTarget:size(1)
  print(string.format("Number of rows: %d *************************", nRows))


  -- init params
  local taArchParam = { nHiddenLayers = taFnnParam.nHiddenLayers,
                        nNodesPerLayer = taFnnParam.nNodesPerLayer,
                        nInputs = teInput:size(2),
                        nOutputs = teTarget:size(2) }

  local fuFoldRunFactory = function(taFoldRunParam)
    return FoldRunFnnTrainer.new(taFoldRunParam)
  end

  local taParam = { 
    nFolds = 5, -- 10
    nSeeds = 10, --10
    teInput = teInput, 
    teTarget = teTarget, 
    mNetAdapter = FnnAdapter.new(taArchParam),
    fuTrainer = trainerPool.trainGrnnMNetAdapter,
    taFuTrainerParams = { nMaxIteration = 20}, --200
    fuTester = testerPool.getMSE }

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

function getResultFilename(taFnnParam, strExprName, isNoise, dMinDist)
  local strNoise = isNoise and "_noise" or ""
  return string.format("result/fnn_nh%d_nnpl%d_%s%s_%.3f.table", 
                                          taFnnParam.nHiddenLayers, taFnnParam.nNodesPerLayer, strExprName, strNoise, dMinDist)

end

local isNoise = myUtil.getBoolFromStr(arg[1])
local nHiddenLayers = arg[2] == nil and 0 or tonumber(arg[2])
local dMinDist = arg[3] == nil and 1 or tonumber(arg[3])
local nExprId = tonumber(arg[4])

local taFnnParam = { nNodesPerLayer = 4, 
                     nHiddenLayers = nHiddenLayers  }
local strExprName = string.format("d_%d", nExprId)
print(string.format("********** Experiemnt %s ***********", strExprName))

local taExprResult = runExperiment(strExprName, isNoise, taFnnParam, dMinDist)
assert(taExprResult, "No result, abort!")
local strResultFilename = getResultFilename(taFnnParam, strExprName, isNoise, dMinDist)
torch.save(strResultFilename, taExprResult, "ascii")

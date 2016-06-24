local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local myUtil = myUtil or require('../../../MyCommon/util.lua')
require('../../FoldRun.lua')
require('../../FoldRunFnnTrainer.lua')
require('../../KFoldRunner.lua')
require('../../FnnAdapter.lua')
require('../common/CDataLoader.lua')

function runExperiment(strExprName, isNoise, taFnnParam)
  local exprSettings = lSettings.getExprSetting(strExprName)
  local dataLoader = CDataLoader.new(exprSettings, isNoise, true)

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
    nSeeds = 5, --10
    teInput = teInput, 
    teTarget = teTarget, 
    mNetAdapter = FnnAdapter.new(taArchParam),
    fuTrainer = trainerPool.trainGrnnMNetAdapter,
    taFuTrainerParams = { nMaxIteration = 20}, --200
    fuTester = testerPool.getMSE }

    local kFoldRunner = KFoldRunner.new(taParam, fuFoldRunFactory)
    while kFoldRunner:hasMore() do
      local foldRun = kFoldRunner:getNext()
      foldRun:Run()
      print(foldRun:getSummaryTable())
    end

  return kFoldRunner:getAggrSummaryTable()
end

function getResultFilename(taFnnParam, strExprName, isNoise)
  local strNoise = isNoise and "_noise" or ""
  return string.format("result/fnn_nh%d_nnpl%d_%s%s.table", 
                                          taFnnParam.nHiddenLayers, taFnnParam.nNodesPerLayer, strExprName, strNoise)

end

local isNoise = myUtil.getBoolFromStr(arg[1])
local nHiddenLayers = arg[2] == nil and 0 or tonumber(arg[2])

local taFnnParam = { nNodesPerLayer = 4, 
                     nHiddenLayers = nHiddenLayers  }
local nMaxExprId = 10 --100
for nExprId=1, nMaxExprId do
  local strExprName = string.format("d_%d", nExprId)
  print(string.format("********** Experiemnt %s ***********", strExprName))

  local taExprResult = runExperiment(strExprName, isNoise, taFnnParam)
  local strResultFilename = getResultFilename(taFnnParam, strExprName, isNoise)

  torch.save(strResultFilename, taExprResult, "ascii")
end


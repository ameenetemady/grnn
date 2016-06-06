local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local lDataLoad = lDataLoad or require('./lDataLoad.lua')
require('../../FoldRun.lua')
require('../../FoldRunFnnTrainer.lua')
require('../../KFoldRunner.lua')
require('../../FnnAdapter.lua')

function runExperiment(strExprName, isNoise, taFnnParam)
  local exprSettings = lSettings.getExprSetting(strExprName)

  --load
  local teInput, taTFNames, taKONames = lDataLoad.load2dInput(exprSettings, isNoise)
  local teTarget, taTargetNames = lDataLoad.loadTarget(exprSettings, isNoise)

  -- init params
  local taArchParam = { nHiddenLayers = taFnnParam.nHiddenLayers,
                        nNodesPerLayer = taFnnParam.nNodesPerLayer,
                        nInputs = teInput:size(2),
                        nOutputs = teTarget:size(2) }

  local fuFoldRunFactory = function(taFoldRunParam)
    return FoldRunFnnTrainer.new(taFoldRunParam)
  end

  local taParam = { 
    nFolds = 3, 
    nSeeds = 5,
    teInput = teInput, 
    teTarget = teTarget, 
    mNetAdapter = FnnAdapter.new(taArchParam),
    fuTrainer = trainerPool.trainGrnnMNetAdapter,
    fuTester = testerPool.getMSE }

    local kFoldRunner = KFoldRunner.new(taParam, fuFoldRunFactory)
    while kFoldRunner:hasMore() do
      local foldRun = kFoldRunner:getNext()
      foldRun:Run()
      print(foldRun:getSummaryTable())
    end

  return kFoldRunner:getAggrSummaryTable()
end

local isNoise = false
local taFnnParam = { nNodesPerLayer = 4, nHiddenLayers = 0 }
local nMaxExprId = 20
for nExprId=1, nMaxExprId do
  local strExprName = string.format("d_%d", nExprId)
  print(string.format("********** Experiemnt %s ***********", strExprName))

  local taExprResult = runExperiment(strExprName, isNoise, taFnnParam)
  local strResultFilename = string.format("result/fnn_nh0_nnpl4_%s.table", 
                                          taFnnParam.nHiddenLayers, taFnnParam.nNodesPerLayer, strExprName)
  torch.save(strResultFilename, taExprResult, "ascii")
end


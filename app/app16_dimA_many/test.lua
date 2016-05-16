local dataLoad = require('../../../MyCommon/dataLoad.lua')
local plotUtil = require('../../../MyCommon/plotUtil.lua')
local nnUtil = require('../../../MyCommon/nnUtil.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local archFactory = require('../../../MyCommon/archFactory.lua')
local grnnUtil = require('../../grnnUtil.lua')
local grnnArchFactory = require('../../grnnArchFactory.lua')
local mySettings = require('../../settings.lua')
require('./lCommon.lua')

local taBaseSettings = mySettings.dimA_many
local nDatasets = taBaseSettings.nRuns

function trainMLP(taTaData, taData)
  local taMlpParam = {nInputs = 2, nOutputs = 3, nNodesPerLayer = 5, nHiddenLayers = 0} 
  local nInits = 10

  local model = nnUtil.getBestTrained(taTaData, taData, nInits, archFactory.mlp, taMlpParam)

  return model
end

function trainGRNN(taTaData, taData)
  local nInits = 10
  local model = nnUtil.getBestTrained(taTaData, taData, nInits, grnnArchFactory.dimA)

  return model
end

function evaluateModel(taTest, model)
  if model == nil then 
    return
  end

  local mseTest = trainerPool.test(taTest, model)
  local fractionOut = trainerPool.getFractionOut(taTest, model, 0.15)


  return {mseTest = mseTest,
          fractionOut = fractionOut}
  end

local taFuModelers = {fnn = trainMLP, grnn = trainGRNN}

function multi_test1()
  local taPerf = {}
  for i=1, nDatasets do
    local taDatasetSettings = genRunSettings(taBaseSettings, i)
    print(taDatasetSettings)
    local taTrain, taTest = dataLoad.loadTrainTest(taDatasetSettings)
    local taTaTrain= grnnUtil.getTable(taTrain[1], taTrain[2])

    taPerf[i] = {}

    for strModelName, fuModeler in pairs(taFuModelers) do
      local model = fuModeler(taTaTrain, taTrain)
      taPerf[i][strModelName] =  evaluateModel(taTest, model)
    end

    print("Processed dataSet: " .. i)

  end


  return taPerf
end

function gen_summary(strTaPerfFilename)
  local taPerf = torch.load(strTaPerfFilename)
  local nModels = 2
  local nSize = nDatasets

  local taPerfInfo = { taFractionOut = {}, taMse = {}}

  for strModelName, fuModeler in pairs(taFuModelers) do
    local teFracOut = torch.Tensor(nSize):fill(1)
    local teMse = torch.Tensor(nSize):fill(1)

    for i=1, nSize do
      teFracOut[i] = taPerf[i][strModelName].fractionOut
      teMse[i] = taPerf[i][strModelName].mseTest
    end

    taPerfInfo.taFractionOut[strModelName]= teFracOut
    taPerfInfo.taMse[strModelName]= teMse
  end

  plotUtil.hist(taPerfInfo.taFractionOut, {nBins = 13, strFigureFilename = "summary/fractionOut.pdf", title = "Histogram for fraction of predictions with >0.15 error"})
  plotUtil.hist(taPerfInfo.taMse, {nBins = 13, min = 0, max= 0.02, strFigureFilename = "summary/mse.pdf", title = "Histogram MSE "})

end


local strTaPerfFilename = "modelPerfs.table"
--local taPerf = multi_test1()
--print(taPerf)
--torch.save(strTaPerfFilename, taPerf)
gen_summary(strTaPerfFilename)

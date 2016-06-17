local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')
local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')
local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')

require('../../FnnAdapter.lua')
require("../../FnnTrainer.lua")

local taFnnParam = { nNodesPerLayer = 4, 
                     nHiddenLayers = 1}

for exprId=4, 4 do


local strExprName = string.format("d_%d", exprId)
lfs.mkdir(string.format("figure/%s", strExprName))
  local exprSettings = lSettings.getExprSetting(strExprName)

  local teInput, taTFNames, taKONames = cDataLoad.load2dInput(exprSettings, false)
  print(taTargetNames)
  local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings, false)

  -- init params
  local taArchParam = { nHiddenLayers = taFnnParam.nHiddenLayers,
                        nNodesPerLayer = taFnnParam.nNodesPerLayer,
                        nInputs = teInput:size(2),
                        nOutputs = teTarget:size(2) }





  local taMNetTrainerParam = { 
    teInput = teInput, 
    teTarget = teTarget, 
    fuTrainer = trainerPool.trainGrnnMNetAdapter,
    taFuTrainerParams = { nMaxIteration = 20}, --200
    fuTester = testerPool.getMSE }


local mNetAdapter = FnnAdapter.new(taArchParam)

local mNetTrainer = FnnTrainer.new(taMNetTrainerParam, mNetAdapter)
local dErr = nil
dErr, mNetTrainer.mNetAdapter = mNetTrainer:trainTogether()

print("##############" .. strExprName)
print(string.format("%d) ######### MSE error:%f",arg[1], dErr))

local teOutput = mNetTrainer.mNetAdapter:getRaw():forward(teInput)

-- Non TFs
  local xId = 1
  local yId = 2

  local teOutputX = teOutput:narrow(2, xId, 1)
  local teOutputY = teOutput:narrow(2, yId, 1)

  local teTargetX = teTarget:narrow(2, xId, 1)
  local teTargetY = teTarget:narrow(2, yId, 1)


  local strExprFigureName = string.format("figure/%s/fnn_%d_new_%s_%s.png", strExprName, arg[1], taTargetNames[1], taTargetNames[yId])
  local taParam = { xlabel = taTargetNames[xId], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

  plotUtil.plot2d(teTargetX, teTargetY, taParam, teOutputX, teOutputY)


  -- TF

    for yId=1, 2 do
      local teOutputX = teOutput:narrow(2, yId, 1)
      local teTargetX = teTarget:narrow(2, yId, 1)

      local teInputX = teInput:select(2,1)


      local strExprFigureName = string.format("figure/%s/fnn_%d_new_%s_%s.png", strExprName, arg[1], taTFNames[1], taTargetNames[yId])
      local taParam = { xlabel = taTFNames[1], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

      plotUtil.plot2d(teInputX, teTargetX, taParam, teInputX, teOutputX)

    end

  end


local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')

local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')
local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')

require('./MFeedforward1Adapter.lua')
require("../../MNetTrainer.lua")
require("../common/CDataLoader.lua")


for exprId=4, 4 do


local strExpName = string.format("d_%d", exprId)
lfs.mkdir(string.format("figure/%s", strExpName))

local exprSettings = lSettings.getExprSetting(strExpName)
local dataLoader = CDataLoader.new(exprSettings, false)

local teInput, taTFNames, taKONames = dataLoader:load3dInput(exprSettings, false)
local teTarget, taTargetNames = dataLoader:loadTarget(exprSettings, false)

local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

local mNetAdapter = MFeedforward1Adapter.new(taNetParam)
local fuTrainer = trainerPool.trainGrnnMNetAdapter
local fuTester = testerPool.getMSE

local mNet = mNetAdapter:getRaw()

local taMNetTrainerParam = { teInput = teInput,
                             teTarget = teTarget,
                             fuTrainer = fuTrainer,
                             fuTester = fuTester,
                             taFuTrainerParams = { nMaxIteration = 100}
                           }

local mNetTrainer = MNetTrainer.new(taMNetTrainerParam, mNetAdapter)
local dErr = nil
mNetTrainer:trainEachUnit()
dErr, mNetTrainer.mNetAdapter = mNetTrainer:trainTogether()

dErr = fuTester(mNetTrainer.mNetAdapter:getRaw(), teInput, teTarget)
print("##############" .. strExpName)
print(string.format("%d) ######### MSE error:%f",arg[1], dErr))

for k, v in pairs(mNetTrainer.mNetAdapter.taWeights) do
  print(k, tostring(torch.repeatTensor(v, 1, 1)))
end

local teOutput = mNetTrainer.mNetAdapter:getRaw():forward(teInput)

-- Non TFs
  local xId = 1
  local yId = 2

  local teOutputX = teOutput:narrow(2, xId, 1)
  local teOutputY = teOutput:narrow(2, yId, 1)

  local teTargetX = teTarget:narrow(2, xId, 1)
  local teTargetY = teTarget:narrow(2, yId, 1)


  local strExprFigureName = string.format("figure/%s/%d_new_%s_%s.png", strExpName, arg[1], taTargetNames[1], taTargetNames[yId])
  local taParam = { xlabel = taTargetNames[xId], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

  plotUtil.plot2d(teTargetX, teTargetY, taParam, teOutputX, teOutputY)


  -- TF

    for yId=1, 2 do
      local teOutputX = teOutput:narrow(2, yId, 1)
      local teTargetX = teTarget:narrow(2, yId, 1)

      local teInputX = teInput:select(3,1)

      local strExprFigureName = string.format("figure/%s/%d_new_%s_%s.png", strExpName, arg[1], taTFNames[1], taTargetNames[yId])
      local taParam = { xlabel = taTFNames[1], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

      plotUtil.plot2d(teInputX, teTargetX, taParam, teInputX, teOutputX)

    end

  end


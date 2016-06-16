local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')
local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')
local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')

require('./MNetAdapter9s.lua')
require("../../MNetTrainer.lua")



local exprId=5
local strExpName = string.format("d_%d", exprId)
lfs.mkdir(string.format("figure/%s", strExpName))

local exprSettings = lSettings.getExprSetting(strExpName)




local teInput, taTFNames, taKONames = cDataLoad.load3dInput(exprSettings)
local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings)

local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

local mNetAdapter = MNetAdapter9s.new(taNetParam)
local fuTrainer = trainerPool.trainGrnnMNetAdapter
local fuTester = testerPool.getMSE

local mNet = mNetAdapter:getRaw()

local taMNetTrainerParam = { teInput = teInput,
                             teTarget = teTarget,
                             fuTrainer = fuTrainer,
                             fuTester = fuTester,
                             taFuTrainerParams = { nMaxIteration = 10}
                           }

local mNetTrainer = MNetTrainer.new(taMNetTrainerParam, mNetAdapter)
local dErr = nil
mNetTrainer:trainEachUnit()
dErr, mNetTrainer.mNetAdapter = mNetTrainer:trainTogether()

dErr = fuTester(mNetTrainer.mNetAdapter:getRaw(), teInput, teTarget)
print("MSE error:", dErr)

local teOutput = mNetTrainer.mNetAdapter:getRaw():forward(teInput)

local taPairsNonTF = {
  {5, 6},
  {6, 7},
  {7, 1},
  {1, 2},
  {3, 4},
  {4, 2}}
local nPairs = table.getn(taPairsNonTF)

for iPairId=1, nPairs do
  local xId = taPairsNonTF[iPairId][1]
  local yId = taPairsNonTF[iPairId][2]

  local teOutputX = teOutput:narrow(2, xId, 1)
  local teOutputY = teOutput:narrow(2, yId, 1)

  local teTargetX = teTarget:narrow(2, xId, 1)
  local teTargetY = teTarget:narrow(2, yId, 1)


  local strExprFigureName = string.format("figure/%s/new_%s_%s.png", strExpName, taTargetNames[xId], taTargetNames[yId])
  local taParam = { xlabel = taTargetNames[xId], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

  plotUtil.plot2d(teTargetX, teTargetY, taParam, teOutputX, teOutputY)
end



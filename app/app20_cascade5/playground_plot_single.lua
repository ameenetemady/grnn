local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')

local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')
local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')

require('./MCascade5Adapter.lua')
require("../../MNetTrainer.lua")
require("../common/CDataLoader.lua")

function myPrint3d(teX)
   local nD3 = teX:size(3)

   for i=1, nD3 do
      print(string.format("d3Idx: %d", i))
      print(teX:select(3, i))
   end
end


torch.manualSeed(1)
exprId=1


local strExpName = string.format("d_%d", exprId)
lfs.mkdir(string.format("figure/%s", strExpName))

local exprSettings = lSettings.getExprSetting(strExpName)
local dataLoader = CDataLoader.new(exprSettings, false, true, 1.3)

local teInput, taTFNames, taKONames = dataLoader:load3dInput(exprSettings, false)
local teTarget, taTargetNames = dataLoader:loadTarget(exprSettings, false)


----[[
local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

local mNetAdapter = MCascade5Adapter.new(taNetParam)
local fuTrainer = trainerPool.trainGrnnMNetAdapter
local fuTester = testerPool.getMSE

local mNet = mNetAdapter:getRaw()

local teOutput = mNet:forward(teInput)
--myPrint3d(teOutput)
print(teOutput)
--[[
local taMNetTrainerParam = { teInput = teInput,
                             teTarget = teTarget,
                             fuTrainer = fuTrainer,
                             fuTester = fuTester,
                             taFuTrainerParams = { 
                               nMaxIteration = 10,
                               --strOptimMethod = "SGD"
                               }
                           }

local mNetTrainer = MNetTrainer.new(taMNetTrainerParam, mNetAdapter)
local dErr = nil
dErr = mNetTrainer:trainEachUnit()
print("trainEach total error:" .. dErr)

local teOutput = mNetTrainer.mNetAdapter:getRaw():forward(teInput)

--]]

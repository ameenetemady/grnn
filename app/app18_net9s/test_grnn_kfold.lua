local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
--local grnnArchFactory = grnnArchFactory or require('../../grnnArchFactory.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')
require('../../FoldRun.lua')
require('../../KFoldRunner.lua')
require('./MNetAdapter9s.lua')


local exprSettings = lSettings.getExprSetting("d_1_small")
local teInput, taTFNames, taKONames = cDataLoad.load3dInput(exprSettings)
local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings)

local taNetParam = { taTFNames = taTFNames, taKONames = taKONames, taTargetNames = taTargetNames }

local taParam = { 
  nFolds = 5, 
  nSeeds = 1,
  teInput = teInput, 
  teTarget = teTarget, 
  mNetAdapter = MNetAdapter9s.new(taNetParam),
  fuTrainer = trainerPool.trainGrnnMNetAdapter,
  fuTester = testerPool.getMSE}

local kFoldRunner = KFoldRunner.new(taParam)
while kFoldRunner:hasMore() do
  local foldRun = kFoldRunner:getNext()
  foldRun:Run()
  print(foldRun:getSummaryTable())
end

print(kFoldRunner:getAggrSummaryTable())

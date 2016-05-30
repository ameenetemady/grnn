local testerPool = testerPool or require('../../../MyCommon/testerPool.lua')
local grnnArchFactory = grnnArchFactory or require('../../grnnArchFactory.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local lDataLoad = lDataLoad or require('./lDataLoad.lua')
require('../../FoldRun.lua')
require('../../KFoldRunner.lua')


local teInput = lDataLoad.load3dInput(lSettings)
local teTarget = lDataLoad.loadTarget(lSettings)

local taParam = { 
  nFolds = 2, 
  nSeeds = 1,
  teInput = teInput, 
  teTarget = teTarget, 
  fuArchGen = grnnArchFactory.net9s,
  fuTrainer = trainerPool.trainGrnn3d,
  fuTester = testerPool.getMSE}

local kFoldRunner = KFoldRunner.new(taParam)
while kFoldRunner:hasMore() do
  local foldRun = kFoldRunner:getNext()
  foldRun:Run()
  print(foldRun:getSummaryTable())
end

print(kFoldRunner:getAggrSummaryTable())

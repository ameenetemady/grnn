torch.manualSeed(0)

local archFactory = archFactory or require('../../../MyCommon/archFactory.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')


local exprSettings = lSettings.getExprSetting("d_1")

-- load:
local isNoise = false
local teInput, taTFNames, taKONames = cDataLoad.load2dInput(exprSettings, isNoise)
local teTarget, taTargetNames = cDataLoad.loadTarget(exprSettings, isNoise)

local nRows = teInput:size(1)
print("Number of samples: " .. nRows)

local taArchParam = { nHiddenLayers = 0,
                      nInputs = teInput:size(2),
                      nOutputs = teTarget:size(2), 
                      nNodesPerLayer = 4 }

local mlp = archFactory.mlp(taArchParam )
local teOutput = mlp:forward(teInput)

local fBest = math.huge
local fBestId = math.huge
local nMaxIter = 10

for seed=1, nMaxIter do
  torch.manualSeed(seed)
  mlp = archFactory.mlp(taArchParam)
  local f = trainerPool.trainGrnn(mlp, teInput, teTarget)
  print("MSE:" .. f .. ", seed: " .. seed)

  if f < fBest then
    fBest = f 
    fBestId = seed
  end

end
print("fBest: " .. fBest .. ", seed: " .. fBestId)

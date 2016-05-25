torch.manualSeed(0)

local archFactory = archFactory or require('../../../MyCommon/archFactory.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local lDataLoad = lDataLoad or require('./lDataLoad.lua')


local exprSettings = lSettings.getExprSetting("d_1_small")
--local exprSettings = lSettings.getExprSetting("d_1")

-- load:
local taTFs = lDataLoad.getData(exprSettings.strTFsFilePath)
local taKOs = lDataLoad.getData(exprSettings.strKOsFilePath)
local taNonTF = lDataLoad.getData(exprSettings.strNonTFsFilePath)

-- Prepare
local teInput = torch.cat(taTFs.teData, taKOs.teData, 2) -- todo: is this correct?
local teTarget = taNonTF.teData

local nRows = taKOs.teData:size(1)
print("Number of samples: " .. nRows)

local taArchParam = { nHiddenLayers = 3,
                      nInputs = teInput:size(2),
                      nOutputs = teTarget:size(2), 
                      nNodesPerLayer = 4 }

local mlp = archFactory.mlp(taArchParam )
local teOutput = mlp:forward(teInput)

local fBest = math.huge
local fBestId = math.huge
local nMaxIter = 40

for seed=1, nMaxIter do
  torch.manualSeed(seed)
  mlp = archFactory.mlp(taArchParam)
  local f = trainerPool.trainGrnn3d(mlp, teInput, teTarget)
  print("MSE:" .. f .. ", seed: " .. seed)

  if f < fBest then
    fBest = f 
    fBestId = seed
  end

end
print("fBest: " .. fBest .. ", seed: " .. fBestId)

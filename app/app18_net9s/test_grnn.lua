torch.manualSeed(0)
require('../../requireBaseUnits.lua')

local grnnArchFactory = grnnArchFactory or require('../../grnnArchFactory.lua')
local trainerPool = trainerPool or require('../../grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')


local exprSettings = lSettings.getExprSetting("d_1")
local taTFs = cDataLoad.getData(exprSettings.strTFsNoNoiseFilePath)
local taNonTF = cDataLoad.getData(exprSettings.strNonTFsNoNoiseFilePath)
--local taTFs = cDataLoad.getData(exprSettings.strTFsFilePath)
--local taNonTF = cDataLoad.getData(exprSettings.strNonTFsFilePath)
local taKOs = cDataLoad.getData(exprSettings.strKOsFilePath)

local nRows = taKOs.teData:size(1)
print("Number of samples: " .. nRows)

local nTFs = taTFs.teData:size(2)
local teTFs_3d = taTFs.teData:view(nRows, nTFs, 1)

local nKOCols = taKOs.teData:size(2)
local teKOs_3d = taKOs.teData:view(nRows, 1, nKOCols)
local teKOs_3d_expanded = teKOs_3d:expand(nRows, nTFs, nKOCols)

local teInput = torch.cat(teTFs_3d, teKOs_3d_expanded, 3)


local mNet9s = grnnArchFactory.net9s()
local teOutput = mNet9s:forward(teInput)
local teTarget = taNonTF.teData

local fBest = math.huge
local fBestId = math.huge
local nMaxIter = 5

for seed=1, nMaxIter do
  torch.manualSeed(seed)
  mNet9s = grnnArchFactory.net9s()
  local f = trainerPool.trainGrnn(mNet9s, teInput, teTarget)
  print("MSE:" .. f .. ", seed: " .. seed)

  if f < fBest then
    fBest = f 
    fBestId = seed
  end

end
print("fBest: " .. fBest .. ", seed: " .. fBestId)

-- load data from tsvs into: TFs, KO, NonTFs
-- build 3d data, to include KO record into the TF's 3rd dimension
-- KO info (included in the 3rd dimension 2:nNonTFs+1) will be available in any input's 3rd dimension as it passes through the network
--

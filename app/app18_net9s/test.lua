local grnnArchFactory = grnnArchFactory or require('../../grnnArchFactory.lua')
local lSettings = lSettings or require('./lSettings.lua')
local lDataLoad = lDataLoad or require('./lDataLoad.lua')


local exprSettings = lSettings.getExprSetting("d_1_small")
local taTFs = lDataLoad.getData(exprSettings.strTFsFilePath)
local taNonTF = lDataLoad.getData(exprSettings.strNonTFsFilePath)
local taKOs = lDataLoad.getData(exprSettings.strKOsFilePath)

local nRows = taKOs.teData:size(1)

local nTFs = taTFs.teData:size(2)
local teTFs_3d = taTFs.teData:view(nRows, nTFs, 1)

local nKOCols = taKOs.teData:size(2)
local teKOs_3d = taKOs.teData:view(nRows, 1, nKOCols)
local teKOs_3d_expanded = teKOs_3d:expand(nRows, nTFs, nKOCols)

local teInput = torch.cat(teTFs_3d, teKOs_3d_expanded, 3)


local mNet9s = grnnArchFactory.net9s()
local teOutput = mNet9s:forward(teInput)


local teTarget = teOutput + 0.01

local criterion = nn.MSECriterion()
local f = criterion:forward(teOutput, teTarget)

-- estimate df/dW
local df_do = criterion:backward(teOutput, teTarget)
print(df_do)
local gradInput = mNet9s:updateGradInput(teInput, df_do)
print(gradInput)


--[[
local n=0
for i=1+n, 10+n do
  print(teTFs_3d[i])
end
--]]

-- load data from tsvs into: TFs, KO, NonTFs
-- build 3d data, to include KO record into the TF's 3rd dimension
-- KO info (included in the 3rd dimension 2:nNonTFs+1) will be available in any input's 3rd dimension as it passes through the network
--

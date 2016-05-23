local lSettings = require('./lSettings.lua')
local lDataLoad = require('/lDataLoad.lua')

local exprSettings = lSettings.getExprSetting("d_1")
local taTFs = lDataLoad.getData(exprSettings.strTFsFilePath)
local taNonTF = lDataLoad.getData(exprSettings.strNonTFsFilePath)
local taKOs = lDataLoad.getData(exprSettings.strKOsFilePath)

local nRows = taKOs.teData:size(1)

local nTFs = taTFs.teData:size(2)
local teTFs_3d = taTFs.teData:view(nRows, nTFs, 1)

local nKOCols = taKOs.teData:size(2)
local teKOs_3d = taKOs.teData:view(nRows, 1, nKOCols)
local teKOs_3d_expanded = teKOs_3d:expand(nRows, nTFs, nKOCols)

local teTFs_3d = torch.cat(teTFs_3d, teKOs_3d_expanded, 3)

local n=0
for i=1+n, 10+n do
  print(teTFs_3d[i])
end

-- load data from tsvs into: TFs, KO, NonTFs
-- build 3d data, to include KO record into the TF's 3rd dimension
-- KO info (included in the 3rd dimension 2:nNonTFs+1) will be available in any input's 3rd dimension as it passes through the network
--

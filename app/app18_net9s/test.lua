local lSettings = require('./lSettings.lua')
local lDataLoad = require('/lDataLoad.lua')

local exprSettings = lSettings.getExprSetting("d_1")
local taTFs = lDataLoad.getTFTensor(exprSettings)
print(taTFs)

-- load data from tsvs into: TFs, KO, NonTFs
-- build 3d data, to include KO record into the TF's 3rd dimension
-- KO info (included in the 3rd dimension 2:nNonTFs+1) will be available in any input's 3rd dimension as it passes through the network
--

torch.manualSeed(0)

local grnnArchFactory = grnnArchFactory or require('../../grnnArchFactory.lua')
local trainerPool = trainerPool or require('../..//grnnTrainerPool.lua')
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('./cDataLoad.lua')


--[[
local exprSettings = lSettings.getExprSetting("d_1")
local taTFs = cDataLoad.getData(exprSettings.strTFsFilePath)
local taNonTF = cDataLoad.getData(exprSettings.strNonTFsFilePath)
local taKOs = cDataLoad.getData(exprSettings.strKOsFilePath)
--]]

local nRows = 100
local taTFs = {teData = torch.rand(nRows, 2)*10} 
local taKOs = {teData = torch.ones(nRows, 7)}

for i=1, 7 do
print("i:" .. i)
taKOs.teData = torch.ones(nRows, 7)
taKOs.teData:narrow(2, i, 1):fill(0)

local nTFs = taTFs.teData:size(2)
local teTFs_3d = taTFs.teData:view(nRows, nTFs, 1)

local nKOCols = taKOs.teData:size(2)
local teKOs_3d = taKOs.teData:view(nRows, 1, nKOCols)
local teKOs_3d_expanded = teKOs_3d:expand(nRows, nTFs, nKOCols)

local teInput = torch.cat(teTFs_3d, teKOs_3d_expanded, 3)


local mNet9s = grnnArchFactory.net9s()
local teOutput = mNet9s:forward(teInput)
--print(teOutput)
print(torch.cat(teOutput:min(1)[1], teOutput:max(1)[1], 2))

end

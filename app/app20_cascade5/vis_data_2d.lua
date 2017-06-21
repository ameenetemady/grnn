local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')

local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')

require("../common/CDataLoader.lua")


torch.manualSeed(1)
for exprId=1, 20 do


local strExpName = string.format("d_%d", exprId)
lfs.mkdir(string.format("figure/%s", strExpName))

local exprSettings = lSettings.getExprSetting(strExpName)
local dataLoader = CDataLoader.new(exprSettings, false, true, 0.000)

local teInput, taTFNames, taKONames = dataLoader:load3dInput(exprSettings, false)
local teTarget, taTargetNames = dataLoader:loadTarget(exprSettings, false)
print("nRrows: " .. teTarget:size(1))

-- TF
do
   local teTargetX = teInput:select(3, 1, 1)
   local teTargetY = teTarget:narrow(2, 1, 1)

     local strExprFigureName = string.format("figure/%s/_new_%s_%s.png", strExpName, "TF", taTargetNames[1])
     local taParam = { xlabel = "TF", ylabel = taTargetNames[1], title = "", strFigureFilename = strExprFigureName }

     plotUtil.plot2d(teTargetX, teTargetY, taParam)
end

-- Non TFs
   for i=1, 3 do
     local xId = i
     local yId = i+1

     local teTargetX = teTarget:narrow(2, xId, 1)
     local teTargetY = teTarget:narrow(2, yId, 1)


     local strExprFigureName = string.format("figure/%s/_new_%s_%s.png", strExpName, taTargetNames[xId], taTargetNames[yId])
     local taParam = { xlabel = taTargetNames[xId], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

     plotUtil.plot2d(teTargetX, teTargetY, taParam)
  end

  end


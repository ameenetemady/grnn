local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')

local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')

require("../common/CDataLoader.lua")


torch.manualSeed(1)
for exprId=1, 3 do


local strExpName = string.format("d_%d", exprId)
lfs.mkdir(string.format("figure/%s", strExpName))

local exprSettings = lSettings.getExprSetting(strExpName)
local dataLoader = CDataLoader.new(exprSettings, false, true, 0.000)

local teInput, taTFNames, taKONames = dataLoader:load3dInput(exprSettings, false)
local teTarget, taTargetNames = dataLoader:loadTarget(exprSettings, false)
print("nRrows: " .. teTarget:size(1))


-- Non TFs
  local xId = 1
  local yId = 2

  local teTargetX = teTarget:narrow(2, xId, 1)
  local teTargetY = teTarget:narrow(2, yId, 1)


  local strExprFigureName = string.format("figure/%s/%d_new_%s_%s.png", strExpName, arg[1], taTargetNames[1], taTargetNames[yId])
  local taParam = { xlabel = taTargetNames[xId], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

  plotUtil.plot2d(teTargetX, teTargetY, taParam)


  --[[ TF
    for yId=1, 2 do
      local teTargetX = teTarget:narrow(2, yId, 1)

      print( "*".. yId .."*")
      local teInputX = teInput:select(3,1)
      print(teInputX)

      local strExprFigureName = string.format("figure/%s/%d_new_%s_%s.png", strExpName, arg[1], taTFNames[1], taTargetNames[yId])
      local taParam = { xlabel = taTFNames[1], ylabel = taTargetNames[yId], title = "", strFigureFilename = strExprFigureName }

      plotUtil.plot2d(teInputX, teTargetX, taParam)

    end
--]]
  end


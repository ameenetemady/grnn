local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')
local cDataLoad = cDataLoad or require('../common/cDataLoad.lua')
local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')

for exprId=1, 1 do

  local strExpName = string.format("d_%d", exprId)
  lfs.mkdir(string.format("figure/%s", strExpName))

  local exprSettings = lSettings.getExprSetting(strExpName)
  local taNonTF = cDataLoad.getData(exprSettings.strNonTFsNoNoiseFilePath)
  local taTF = cDataLoad.getData(exprSettings.strTFsNoNoiseFilePath)

  print("** " .. strExpName .. " **" )
	print(exprSettings)


  -- NonTFs
--  local xId = 1 
--  local yId = 2
--
--  local teX = taNonTF.teData:narrow(2, xId, 1)
--  local teY = taNonTF.teData:narrow(2, yId, 1)
--
--  local strExprFigureName = string.format("figure/%s/%s_%s.png", strExpName, taNonTF.taGenes[xId], taNonTF.taGenes[yId])
--  local taParam = { xlabel = taNonTF.taGenes[xId], ylabel = taNonTF.taGenes[yId], title = "", strFigureFilename = strExprFigureName }
--
--  plotUtil.plot2d(teX, teY, taParam)
--
--
--  -- TFs
--  for yId=1, 2 do
--    local teX = taTF.teData:narrow(2, 1, 1)
--    local teY = taNonTF.teData:narrow(2, yId, 1)
--    local strExprFigureName = string.format("figure/%s/%s_%s.png", strExpName, taTF.taGenes[1], taNonTF.taGenes[yId])
--    local taParam = { xlabel = taTF.taGenes[1], ylabel = taNonTF.taGenes[yId], title = "", strFigureFilename = strExprFigureName }
--
--    plotUtil.plot2d(teX, teY, taParam)
--  end

  -- 3d
  local teX = taTF.teData:narrow(2, 1, 1)
  local teY = taNonTF.teData:narrow(2, 1, 1)
  local teZ = taNonTF.teData:narrow(2, 2, 1)

--	print(taTF.teData)
--	print(taNonTF.teData)

--  local strExprFigureName = string.format("figure/%s/%s_%s_%s.png", strExpName, taTF.taGenes[1], taNonTF.taGenes[1], taNonTF.taGenes[2])
  local taParam = { xlabel = taTF.taGenes[1], ylabel = taNonTF.taGenes[1], zlabel = taNonTF.taGenes[2], 
  title = "", strFigureFilename = strExprFigureName }

  plotUtil.plot3d(teX, teY, teZ, taParam)

end
--]]

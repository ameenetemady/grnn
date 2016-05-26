local lSettings = lSettings or require('./lSettings.lua')
local lDataLoad = lDataLoad or require('./lDataLoad.lua')
local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')

for exprId=1, 20 do

  local strExpName = string.format("d_%d", exprId)
  local exprSettings = lSettings.getExprSetting(strExpName)
  local taTFs = lDataLoad.getData(exprSettings.strTFsNoNoiseFilePath)
  local taNonTF = lDataLoad.getData(exprSettings.strNonTFsNoNoiseFilePath)
  local taKOs = lDataLoad.getData(exprSettings.strKOsFilePath)

  local xId = 6
  local yId = 7

  local teX = taNonTF.teData:narrow(2, xId, 1)
  local teY = taNonTF.teData:narrow(2, yId, 1)

  local strExprFigureName = string.format("figure/%s_%s_%s.png", strExpName, taNonTF.taGenes[xId], taNonTF.taGenes[yId])
  local taParam = { xlabel = taNonTF.taGenes[xId], ylabel = taNonTF.taGenes[yId], title = "", strFigureFilename = strExprFigureName }

  plotUtil.plot2d(teX, teY, taParam)

end

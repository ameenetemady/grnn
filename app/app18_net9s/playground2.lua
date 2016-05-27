local lfs = lfs or require 'lfs'
local lSettings = lSettings or require('./lSettings.lua')
local lDataLoad = lDataLoad or require('./lDataLoad.lua')
local plotUtil = plotUtil or require('../../../MyCommon/plotUtil.lua')

local taPairsNonTF = {
  {5, 6},
  {6, 7},
  {7, 1},
  {1, 2},
  {3, 4},
  {4, 2}}

local nPairs = table.getn(taPairsNonTF)
print(nPairs)

for exprId=1, 20 do

  local strExpName = string.format("d_%d", exprId)
  lfs.mkdir(string.format("figure/%s", strExpName))

  local exprSettings = lSettings.getExprSetting(strExpName)
  local taNonTF = lDataLoad.getData(exprSettings.strNonTFsNoNoiseFilePath)
  print(taNonTF.taGenes)

  print("** " .. strExpName .. " **" )
  for iPairId=1, nPairs do
    local xId = taPairsNonTF[iPairId][1]
    local yId = taPairsNonTF[iPairId][2]

    print(string.format("%d:%d", xId, yId ))

    local teX = taNonTF.teData:narrow(2, xId, 1)
    local teY = taNonTF.teData:narrow(2, yId, 1)

    local strExprFigureName = string.format("figure/%s/%s_%s.png", strExpName, taNonTF.taGenes[xId], taNonTF.taGenes[yId])
    local taParam = { xlabel = taNonTF.taGenes[xId], ylabel = taNonTF.taGenes[yId], title = "", strFigureFilename = strExprFigureName }

    plotUtil.plot2d(teX, teY, taParam)

  end

end
--]]

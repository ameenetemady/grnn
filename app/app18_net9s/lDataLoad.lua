local dataLoad = dataLoad or require('../../../MyCommon/dataLoad.lua')
local lDataLoad = {}

do
  function lDataLoad.getData(strFilePath)
    local taGenes = dataLoad.getHeader(strFilePath)
    local taLoadParam = { strFilename = strFilePath, nCols = table.getn(taGenes), taCols = taGenes, isHeader = true }
    local teData = dataLoad.loadTensorFromTsv(taLoadParam)

    return { taGenes = taGenes, teData = teData }
  end

  return lDataLoad
end

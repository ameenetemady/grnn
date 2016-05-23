local dataLoad = dataLoad or require('../../../MyCommon/dataLoad.lua')
local lDataLoad = {}

do
  function lDataLoad.getTFTensor(lSettings)
    local taGenes = dataLoad.getHeader(lSettings.strTFsFilePath)

    local taCols = {}
    local nGenes = 0
    for k, v in pairs(taTFs) do
      taCols[v]=true
      nCols = nCols + 1
    end

    
    print(taHeader)

  end

  return lDataLoad
end

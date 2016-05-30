local dataLoad = dataLoad or require('../../../MyCommon/dataLoad.lua')
local lDataLoad = {}

do
  function lDataLoad.getData(strFilePath)
    local taGenes = dataLoad.getHeader(strFilePath)
    local taLoadParam = { strFilename = strFilePath, nCols = table.getn(taGenes), taCols = taGenes, isHeader = true }
    local teData = dataLoad.loadTensorFromTsv(taLoadParam)

    return { taGenes = taGenes, teData = teData }
  end

  function lDataLoad.load3dInput(exprSettings)
    local taTFs = lDataLoad.getData(exprSettings.strTFsNoNoiseFilePath)
    local taKOs = lDataLoad.getData(exprSettings.strKOsFilePath)
    local nRows = taKOs.teData:size(1)

    local nTFs = taTFs.teData:size(2)
    local teTFs_3d = taTFs.teData:view(nRows, nTFs, 1)

    local nKOCols = taKOs.teData:size(2)
    local teKOs_3d = taKOs.teData:view(nRows, 1, nKOCols)
    local teKOs_3d_expanded = teKOs_3d:expand(nRows, nTFs, nKOCols)

    local teInput = torch.cat(teTFs_3d, teKOs_3d_expanded, 3)

    return teInput
  end

  function lDataLoad.loadTarget(exprSettings)

    local taNonTF = lDataLoad.getData(exprSettings.strNonTFsNoNoiseFilePath)

    return taNonTF.teData
  end

  return lDataLoad
end

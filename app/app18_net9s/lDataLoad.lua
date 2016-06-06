local dataLoad = dataLoad or require('../../../MyCommon/dataLoad.lua')
local lDataLoad = {}

do
  function lDataLoad.getData(strFilePath)
    print(strFilePath)

    local taGenes = dataLoad.getHeader(strFilePath)
    local taLoadParam = { strFilename = strFilePath, nCols = table.getn(taGenes), taCols = taGenes, isHeader = true }
    local teData = dataLoad.loadTensorFromTsv(taLoadParam)

    return { taGenes = taGenes, teData = teData }
  end

  function lDataLoad.load3dInput(exprSettings, isNoise)
    isNoise = isNoise or false

    local strTFFilepath = isNoise and exprSettings.strTFsFilePath or exprSettings.strTFsNoNoiseFilePath
    local strKOFilepath = exprSettings.strKOsFilePath

    local taTFs = lDataLoad.getData(strTFFilepath)
    local taKOs = lDataLoad.getData(strKOFilepath)
    local nRows = taKOs.teData:size(1)

    local nTFs = taTFs.teData:size(2)
    local teTFs_3d = taTFs.teData:view(nRows, nTFs, 1)

    local nKOCols = taKOs.teData:size(2)
    local teKOs_3d = taKOs.teData:view(nRows, 1, nKOCols)
    local teKOs_3d_expanded = teKOs_3d:expand(nRows, nTFs, nKOCols)

    local teInput = torch.cat(teTFs_3d, teKOs_3d_expanded, 3)

    return teInput, taTFs.taGenes, taKOs.taGenes
  end

  function lDataLoad.load2dInput(exprSettings, isNoise)
    isNoise = isNoise or false

    local strTFFilepath = isNoise and exprSettings.strTFsFilePath or exprSettings.strTFsNoNoiseFilePath
    local strKOFilepath = exprSettings.strKOsFilePath

    local taTF = lDataLoad.getData(strTFFilepath)
    local taKO = lDataLoad.getData(strKOFilepath)

    local teInput = torch.cat(taTF.teData, taKO.teData, 2)

    return teInput, taTF, taKO
  end

  function lDataLoad.loadTarget(exprSettings, isNoise)
    isNoise = isNoise or false

    local strNonTFFilepath = isNoise and exprSettings.strNonTFsFilePath or exprSettings.strNonTFsNoNoiseFilePath
    local taNonTF = lDataLoad.getData(strNonTFFilepath)

    return taNonTF.teData, taNonTF.taGenes
  end

  return lDataLoad
end

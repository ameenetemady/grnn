local cSettings = {}

do
  function cSettings.getExprSetting(strExpName, strAppBase)
    local taExprSetting = {}
    taExprSetting.strKOsFilePath = string.format("%s/%s/processed_KO.tsv", strAppBase, strExpName)
    taExprSetting.strTFsFilePath = string.format("%s/%s/processed_TFs.tsv", strAppBase, strExpName)
    taExprSetting.strTFsNoNoiseFilePath = string.format("%s/%s/processed_TFsNoNoise.tsv", strAppBase, strExpName)
    taExprSetting.strNonTFsFilePath = string.format("%s/%s/processed_NonTFs.tsv", strAppBase, strExpName)
    taExprSetting.strNonTFsNoNoiseFilePath = string.format("%s/%s/processed_NonTFsNoNoise.tsv", strAppBase, strExpName)

    return taExprSetting
  end

  return cSettings
end



local lSettings = {}

do
  local strBaseGnwPath = "/home/ameen/gnw"
  local strAppBase = string.format("%s/app18_net9s", strBaseGnwPath)

  lSettings.taExpr = {"d_1" , "d_2"}

  function lSettings.getExprSetting(strExpName)
    local taExprSetting = {}
    taExprSetting.strKOsFilePath = string.format("%s/%s/processed_KO.tsv", strAppBase, strExpName)
    taExprSetting.strTFsFilePath = string.format("%s/%s/processed_TFs.tsv", strAppBase, strExpName)
    taExprSetting.strNonTFsFilePath = string.format("%s/%s/processed_NonTFs.tsv", strAppBase, strExpName)

    return taExprSetting
  end

  return lSettings
end



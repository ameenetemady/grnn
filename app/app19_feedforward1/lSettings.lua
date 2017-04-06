local cSettings = cSettings or require("../common/cSettings.lua")
local lSettings = {}

do
  local strBaseGnwPath = "/Users/ameen/mygithub/gnw"
  local strAppBase = string.format("%s/app19_feedforward1", strBaseGnwPath)

  function lSettings.getExprSetting(strExpName)
    return cSettings.getExprSetting(strExpName, strAppBase)
  end

  return lSettings
end



require('./MNetAdapter9s.lua')
local cSettings = cSettings or require("../common/cSettings.lua")
local lSettings = {}

do
  local strBaseGnwPath = string.format("%s/mygithub/gnw", os.getenv("HOME"))
  local strAppBase = string.format("%s/app18_net9s", strBaseGnwPath)

  function lSettings.getExprSetting(strExpName)
    return cSettings.getExprSetting(strExpName, strAppBase)
  end

  function lSettings.getNewMNet(taNetParam)
    return MNetAdapter9s.new(taNetParam)
  end

  return lSettings
end



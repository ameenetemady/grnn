require('./MNetAdapter9s.lua')
local cSettings = cSettings or require("../common/cSettings.lua")
local lSettings = {}

do
  function lSettings.getExprSetting(strExpName)
    local strBaseGnwPath = string.format("%s/mygithub/gnw", os.getenv("HOME"))
    local strAppBase = string.format("%s/app18_net9s", strBaseGnwPath)
    return cSettings.getExprSetting(strExpName, strAppBase)
  end

  function lSettings.getNewMNet(taNetParam)
    return MNetAdapter9s.new(taNetParam)
  end

  return lSettings
end



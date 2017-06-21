require('./MCascade5Adapter.lua')
local cSettings = cSettings or require("../common/cSettings.lua")
local lSettings = {}

do
  local strBaseGnwPath = string.format("%s/mygithub/gnw", os.getenv("HOME"))
  local strAppBase = string.format("%s/app20_cascade5", strBaseGnwPath)

  function lSettings.getExprSetting(strExpName)
    return cSettings.getExprSetting(strExpName, strAppBase)
  end

  function lSettings.getNewMNet(taNetParam)
    return MCascade5Adapter.new(taNetParam)
  end

  return lSettings
end



require('./MFeedforward1Adapter.lua')
local cSettings = cSettings or require("../common/cSettings.lua")
local lSettings = {}

do
  local strBaseGnwPath = string.format("%s/mygithub/gnw", os.getenv("HOME"))
  local strAppBase = string.format("%s/app19_feedforward1", strBaseGnwPath)

  function lSettings.getExprSetting(strExpName)
    return cSettings.getExprSetting(strExpName, strAppBase)
  end

  function lSettings.getNewMNet(taNetParam)
    return MFeedforward1Adapter.new(taNetParam)
  end

  return lSettings
end



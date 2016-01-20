local mySettings = require("../../mySettings.lua")
local dataUtil = require("../../regulondb/dataUtil.lua")

do
  local dataLoad = {}

  function dataLoad.loadUnique(taMeta)
    local strFilename = "/media/ameen/MyDrive/Research/Data/transcriptome.expand.v7pm.proc3_uniq.luadata"
    local taData = torch.load(strFilename)

    local nTargetCols = #taMeta.targetFilter
    local nRows = taData.teY:size(1)

    local teYSelected = torch.Tensor(nRows, nTargetCols):fill(0)

    local taGeneDic = dataUtil.loadBNumberDic(mySettings.strFilenameGeneDic)
    for k, v in pairs(taMeta.targetFilter) do
      local strBNumber = taGeneDic[v]
      local idExtract =  taData.taHeader.taY["m.".. strBNumber]

      teYSelected:select(2, tonumber(k)):copy(taData.teY:select(2, idExtract))
    end

    local taDataNew =  { taCondKey = taData.taCondKey, teY = teYSelected}

    return taDataNew
  end

  return dataLoad
end


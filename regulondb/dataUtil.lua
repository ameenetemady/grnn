local csv = require("csv")
local myUtil = require("../../MyCommon/util.lua")

do
  local dataUtil = {}

  function dataUtil.loadBNumberDic(strFilename)
      local taLoadParams = {header=false}
      local f = csv.open(strFilename, taLoadParams)

      local taNameToBNumber = {}

      for fields in f:lines() do
        local strGeneName = fields[7]
        local strBNumber = fields[3]
        taNameToBNumber[strGeneName] = strBNumber
      end

      return taNameToBNumber
  end

  return dataUtil
end

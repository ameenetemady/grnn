local settings = require('../../settings.lua')
local csv = require("csv")
local myUtil = require('../../../MyCommon/util.lua')

local dataLoad = {}

do
  function dataLoad.loadFromTsv(strFilename, nCols)
    local taLoadParams = {header=false}
    local f = csv.open(strFilename, taLoadParams)

    local taData= {}
    for fields in f:lines() do
      local teRow = torch.Tensor(nCols)
      for i=1, nCols do
        teRow[i] = tonumber(fields[i]) 
      end

      table.insert(taData, teRow)
    end

    local teData = myUtil.getTensorFromTableOfTensors(taData)

    return teData
  end

  function  dataLoad.loadInput()
    local nInputCols = settings.cascade4_big.nInputCols
    local teInput = dataLoad.loadFromTsv(settings.cascade4_big.inputFilename, nInputCols)
    teInput:narrow(2, 2, nInputCols -1):add(1) -- so that "-1" becomes "0" and "0" becomes "1"
    return teInput
  end

  function dataLoad.loadTarget()
    local nTargetCols = settings.cascade4_big.nTargetCols
    local teTarget = dataLoad.loadFromTsv(settings.cascade4_big.targetFilename, nTargetCols)
    return teTarget

  end

--  print(dataLoad.loadInput())
--  print(dataLoad.loadTarget())

  return dataLoad
end


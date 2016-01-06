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
    local nInputCols = settings.cascade4.nInputCols
    local teInput = dataLoad.loadFromTsv(settings.cascade4.inputFilename, nInputCols)
    teInput:narrow(2, 2, nInputCols -1):add(1) -- so that "-1" becomes "0" and "0" becomes "1"
    return teInput
  end

  function dataLoad.loadTarget()
    local nTargetCols = settings.cascade4.nTargetCols
    local teTarget = dataLoad.loadFromTsv(settings.cascade4.targetFilename, nTargetCols)
    return teTarget

  end

  function dataLoad.isMatch(a, b, noise)

    local bPlusNoise = b + noise
    local bMinusNoise = b - noise
    local isALess = torch.lt(a, bPlusNoise)
    local isAMore = torch.gt(a, bMinusNoise)
    return torch.all(isALess) and torch.all(isAMore)
  end

  function dataLoad.loadTrainTest()
    local teInput = dataLoad.loadInput()
    local teTarget = dataLoad.loadTarget()
    local taInputTest = {}
    local taInputTrain = {}
    local taTargetTrain = {}
    local taTargetTest = {}
    local noise = 0.1

    local teInutSetting = teInput:sub(1, -1, 2, -1)
    local nGenes = teInutSetting:size(2)

    for i=1, teInput:size(1) do
      if teInutSetting[i]:sum() == nGenes or 
         dataLoad.isMatch(teInput[i], torch.Tensor({0, 0, 1, 0, 1 }), noise) or
         dataLoad.isMatch(teInput[i], torch.Tensor({0, 0, 0, 1, 0 }), noise) then

        table.insert(taInputTrain, teInput[i])
        table.insert(taTargetTrain, teTarget[i])
      else
        table.insert(taInputTest, teInput[i])
        table.insert(taTargetTest, teTarget[i])
      end
    end

    local taTrain = { myUtil.getTensorFromTableOfTensors(taInputTrain), myUtil.getTensorFromTableOfTensors(taTargetTrain) }
    local taTest = { myUtil.getTensorFromTableOfTensors(taInputTest), myUtil.getTensorFromTableOfTensors(taTargetTest) }

    return taTrain, taTest
  end

  return dataLoad
end


local settings = require('../../settings.lua')
local csv = require("csv")
local myUtil = require('../../../MyCommon/util.lua')

local dataLoad = {}

do
  function dataLoad.loadFromTsv(strFilename, nCols)
    local taLoadParams = {header=false, separator="\t"}
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
    local nInputCols = settings.feedforward1.nInputCols
    local teInput = dataLoad.loadFromTsv(settings.feedforward1.inputFilename, nInputCols)
    return teInput
  end

  function dataLoad.loadTarget()
    local nTargetCols = settings.feedforward1.nTargetCols
    local teTarget = dataLoad.loadFromTsv(settings.feedforward1.targetFilename, nTargetCols)
    return teTarget

  end

  function dataLoad.loadTrainTest()
    local teInput = dataLoad.loadInput()
    local teTarget = dataLoad.loadTarget()

    local nSize = teInput:size(1)

    local teIdx = torch.linspace(1, nSize, nSize)
    local trainMask = torch.mod(teIdx, 2):eq(torch.zeros(nSize))
    local testMask = torch.mod(teIdx, 2):ne(torch.zeros(nSize))

    local teTrain_input = teInput:maskedSelect(torch.ByteTensor(nSize, 1):copy(trainMask))
    local tmp = torch.ByteTensor(nSize, 2):copy(torch.cat(trainMask, trainMask, 2))
    local teTrain_target = teTarget:maskedSelect(tmp)
    teTrain_input:resize(trainMask:sum(), 1)
    teTrain_target:resize(trainMask:sum(), 2)
    print(teTrain_target:size())
    local taTrain = {teTrain_input, teTrain_target}

    
    local teTest_input = teInput:maskedSelect(testMask)
    local teTest_target = teTarget:maskedSelect(torch.cat(testMask, testMask, 2))
    teTest_input:resize(testMask:sum(), 1)
    teTest_target:resize(testMask:sum(), 2)
    print(teTest_target:size())
    local taTest = {teTest_input, teTest_target}

    return taTrain, taTest
  end

  return dataLoad
end

dataLoad.loadTrainTest()

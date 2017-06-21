local myUtil = require('../MyCommon/util.lua')

function getFromFile(strFilename, taColnames)
   local taContent = torch.load(strFilename, 'ascii')
   local taData = {}
   for k, taV in pairs(taContent) do
      local taCols = {}
      for i, strColname in pairs(taColnames) do
         taCols[i] = taV[strColname]
      end

      table.insert(taData, taCols)
   end

   local teData = torch.Tensor(taData)
   return torch.totable(torch.median(teData, 1):squeeze())
end

function getAggregated(taArgs, taColnames)
   local taRes = {}
   local nArgs = table.getn(taArgs)
   local strLabel = arg[1]
   for i=2, nArgs do
      local taFileData = getFromFile(taArgs[i], taColnames)
      table.insert(taFileData, strLabel)
      table.insert(taRes, taFileData)
   end

   return taRes
end

local taColnames = { "nRows", "taTestResult" } 
local taAggr = getAggregated(arg, taColnames)

local csvString = myUtil.getCsvStringFromTable(taAggr)
print(csvString)

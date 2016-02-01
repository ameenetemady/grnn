require 'gnuplot'
local mySettings = require("../../mySettings.lua")
local myUtil = require("../../../MyCommon/util.lua")
local dataLoad = require("./dataLoad.lua")

local dataLoad_test = {}


function dataLoad_test.loadAndSaveTo_test1()
  local taParam = { taGenes = { hns = true, rcsB = true }, nGenes = 2 }
--  local taParam = { taGenes = { lacI= true, lacZ= true }, nGenes = 2 }
  dataLoad.loadAndSaveTo(taParam, mySettings.strFilename_md3_exprData_p1)
end

function dataLoad_test.analyze_p1()
  local taData = torch.load(mySettings.strFilename_md3_exprData_p1)

--  --[[
  local taV = {}
  for key, taVal in pairs(taData) do
    taV[key] = torch.Tensor(taVal)
  end

--gnuplot.hist(taV.lacI, 10)
gnuplot.raw('set xlabel "hns"')
gnuplot.raw('set ylabel "rcsB"')
gnuplot.plot({'data', taV.hns, taV.rcsB, 'points pt 2 ps 0.4'})
--[[
print("hns")
print(taV.hns:mean())
print(taV.hns:min())
print(taV.hns:max())
print(taV.hns:std())
print("rcsB")
print(taV.rcsB:mean())
print(taV.rcsB:min())
print(taV.rcsB:std())
--]]


--  print(taV.lacZ)

--]]
end

function dataLoad_test.all()
--  dataLoad_test.loadAndSaveTo_test1()
  dataLoad_test.analyze_p1()

end

dataLoad_test.all()

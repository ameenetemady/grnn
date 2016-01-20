require 'gnuplot'
local dataLoad = require("./dataLoad.lua")
local info = require("../../regulondb/info.lua")

--local plotUtil = require("../../../MyCommon/plotUtil.lua")

-- for cascade myGroup1:
-- 1) load data
-- 2) collapse unique conditions (median)
-- 3) extract expression levels of g1, g2, g3 
-- 4) plot g1 vs. g2 and  g2 vs. g3

local taMeta = { targetFilter = info.taMyGeneGroups[3] , strNamingStyle = "geneName" }
local taData = dataLoad.loadUnique(taMeta) -- taData.teTarget will contain only above genes
gnuplot.raw('set xlabel "' .. taMeta.targetFilter[2] .. '"')
gnuplot.raw('set ylabel "' .. taMeta.targetFilter[3] .. '"')
gnuplot.plot({'data', taData.teY:select(2,2), taData.teY:select(2,3), 'points pt 2 ps 0.4'})

--plotUtil.plot(taData.teTarget[1], taData.teTarget[2])
--plotUtil.plot(taData.teTarget[2], taData.teTarget[3])

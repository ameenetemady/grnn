require 'gnuplot'
local dataLoad = require('../../../MyCommon/dataLoad.lua')
local plotUtil = require('../../../MyCommon/plotUtil.lua')
local mySettings = require('../../settings.lua')
require('./lCommon.lua')

local taBaseSettings = mySettings.feedforward1_many
local nRuns =  taBaseSettings.nRuns
local strFigurePrefix = "figure/raw_"

for i=1, nRuns do
  local taRunSettings = genRunSettings(taBaseSettings, i)
  
  local teInput = dataLoad.loadTensorFromTsv(taRunSettings.taInput)
  local teTarget = dataLoad.loadTensorFromTsv(taRunSettings.taTarget)

  local taParam = {xlabel = "g1", 
                   ylabel = "g2", 
                   title = "Run: " .. i, 
                   strFigureFilename = string.format("%s%d.png", strFigurePrefix, i)}

  plotUtil.plot2d(teInput, teTarget:select(2, 2), taParam)
end

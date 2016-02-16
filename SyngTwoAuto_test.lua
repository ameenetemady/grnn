require 'nn'
require 'gnuplot'

local myUtil = require('../MyCommon/util.lua')
local trainerPool = require('../MyCommon/trainerPool.lua')
local grnnUtil = require('./grnnUtil.lua')
local syngTwoAuto = require('./SyngTwoAuto.lua')

local SyngTwoAuto_test = {}

function SyngTwoAuto_test.AutoSyngTwoAuto_test1()
  torch.manualSeed(1)

  local synthWeight = torch.Tensor({0, 1, 3, 0.5, 0.5, 1, 3, 1, 3})
  local nSize = 500

  local taData = genData1(synthWeight, nSize)


 -- --[[
  
  local weight = torch.Tensor({1.5, 0, 0, 0, 0, 2, 3, 2, 3})
  local mlp = syngTwoAuto.new(weight)

  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)


  print("expected parameters")
  print(synthWeight)
  --]]

end

function genData1(synthWeight, nSize)
  local teX = torch.rand(nSize, 2)*2
  local mlp = syngTwoAuto.new(synthWeight)
  local teY = mlp:forward(teX):reshape(nSize, 1)


--  gnuplot.plot({'x1', teX:select(2,1):squeeze(), teY:squeeze(), 'points pt 2 ps 0.4'},
--                {'x2', teX:select(2,2):squeeze(), teY:squeeze(), 'points pt 2 ps 0.2 lc rgb "red"'})

  print(teY:size())

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { teX[i], teY[i] }
    table.insert(taData, taRow)
  end

  return taData
end

function SyngTwoAuto_test.all()
  SyngTwoAuto_test.AutoSyngTwoAuto_test1()
end

SyngTwoAuto_test.all()

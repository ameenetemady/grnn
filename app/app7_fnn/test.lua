require 'nn'
require 'gnuplot'
local myUtil = require('../../../MyCommon/util.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../../grnnUtil.lua')

function plot1()
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()


  gnuplot.plot({'1', teInput:select(2,2):squeeze(), teTarget:squeeze(), 'points pt 2 ps 0.4'})
end

function test2(nHiddenLayers, nNodesPerLayer)
  torch.manualSeed(1)
--  local teInput = dataLoad.loadInput()
--  local teTarget = dataLoad.loadTarget()
--  local taData = grnnUtil.getTable(teInput, teTarget)


  local taTrain, taTest = dataLoad.loadTrainTest()
  local taData = grnnUtil.getTable(taTrain[1], taTrain[2])

  -- 2) Generate Model
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(2, nNodesPerLayer))
  mlp:add(nn.Sigmoid())

  for i=1, nHiddenLayers do
    mlp:add(nn.Linear(nNodesPerLayer, nNodesPerLayer))
    mlp:add(nn.Sigmoid())
  end

  mlp:add(nn.Linear(nNodesPerLayer, 1))

  -- 3) train Model / load Model
  trainerPool.full_CG(taData, mlp)
 
--  local teOutput = mlp:forward(teInput)
--  gnuplot.plot({'1', teTarget:squeeze(), teOutput:squeeze(), 'points pt 2 ps 0.4'})


  local teOutput = mlp:forward(taTest[1])
  gnuplot.plot({'1', taTest[2]:squeeze(), teOutput:squeeze(), 'points pt 2 ps 0.4'})

end

function plot2()
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()
  local taData = grnnUtil.getTable(teInput, teTarget)

--  local weight = torch.Tensor({0.3587919,0.0386136,0.9787060,0.3409350,0.9021380,0.5875384,-7.5971354,0.3412197,-9.1355756})
  local weight = torch.Tensor({-0.5278220,-0.8153228,-0.2068385,-0.6274796,-0.2241785,-0.3088785,0.3394921,-0.2064651,0.8710781})
  local mlp = syngTwoAuto.new(weight)
  trainerPool.full_CG(taData, mlp)

  local teOutput = mlp:forward(teInput)
  gnuplot.plot({'1', teTarget:squeeze(), teOutput:squeeze(), 'points pt 2 ps 0.4'})

end

--plot1()

--test1()
--plot2()
test2(0, 3)

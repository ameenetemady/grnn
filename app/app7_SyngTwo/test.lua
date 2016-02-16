require 'nn'
require 'gnuplot'
local myUtil = require('../../../MyCommon/util.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../../grnnUtil.lua')
local syngTwoAuto = require('../../SyngTwoAuto.lua')

function plot1()
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()


  gnuplot.plot({'1', teInput:select(2,2):squeeze(), teTarget:squeeze(), 'points pt 2 ps 0.4'})
end

function test1()
  torch.manualSeed(1)
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()
  local taData = grnnUtil.getTable(teInput, teTarget)

--  local weight = torch.Tensor({-0.5408456,-0.9452248,0.0688278,0.3409350,0.8279240,-0.1653904,-0.0855904,0.1173797,-0.1386029})
  local weight = torch.Tensor({0.3587919,0.0386136,0.9787060,0.3409350,0.9021380,0.5875384,-7.5971354,0.3412197,-9.1355756})
--  local weight = torch.Tensor({0, 0, 0, 0, 0, 0, 0, 0, 0})
--  for i=1,10 do
--    print("*** i=" .. i .. " ***")
--    local weight = torch.rand(9)*2 - 1
--    print(myUtil.getCsvStringFrom1dTensor(weight))
    local mlp = syngTwoAuto.new(weight)
    trainerPool.full_CG(taData, mlp)

    local paramOptim, __ = mlp:parameters()
    print(myUtil.getCsvStringFrom1dTensor(paramOptim[1]))
--  end

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
plot2()

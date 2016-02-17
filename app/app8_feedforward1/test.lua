require 'nn'
require 'gnuplot'
local myUtil = require('../../../MyCommon/util.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../../grnnUtil.lua')
local syngTwoAuto = require('../../SyngTwoAuto.lua')
local syngOneAuto = require('../../SyngOneAuto.lua')

function plot1()
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()

  gnuplot.plot({'1', teInput:squeeze(), teTarget:select(2, 1):squeeze(), 'points pt 2 ps 0.4'},
               {'2', teInput:squeeze(), teTarget:select(2, 2):squeeze(), 'points pt 2 ps 0.4'})
end

function feedforwardFactory(param)
  local mlp_g86 = nn.Concat(2)
  mlp_g86:add(nn.Identity())
  mlp_g86:add(syngOneAuto.new(param.g6w))

  local mlp_g867 = nn.Concat(2)
  mlp_g867:add(nn.Identity())
  mlp_g867:add(syngTwoAuto.new(param.g7w))

  local main = nn.Sequential()
  main:add(mlp_g86)
  main:add(mlp_g867)
  main:add(nn.Narrow(2, 2, 2))
  
  return main
end

function test1()
  torch.manualSeed(1)

--  local teInput = dataLoad.loadInput()
--  local teTarget = dataLoad.loadTarget()
--  local taData = grnnUtil.getTable(teInput, teTarget)

  local taTrain, taTest = dataLoad.loadTrainTest()
  local taData = grnnUtil.getTable(taTrain[1], taTrain[2])


  for i=1,10 do
    print("*** i=" .. i .. " ***")
    local param = { g6w = torch.rand(1, 4)*2-1,
                    g7w = torch.rand(9)*2-1}

    local mlp = feedforwardFactory(param)

    print(myUtil.getCsvStringFrom2dTensor(param.g6w))
    print(myUtil.getCsvStringFrom1dTensor(param.g7w))

    trainerPool.full_CG(taData, mlp)
    local testErr = trainerPool.test(taTest, mlp)
     print("testError: " .. testErr)


    local paramOptim, __ = mlp:getParameters()
    print(myUtil.getCsvStringFrom1dTensor(paramOptim))
  end


 

  --[[
  local param = { g6w = torch.Tensor({{1, 1, 1, 1}}),
                  g7w = torch.Tensor({1.5, 0, 0, 0, 0, 2, 3, 2, 3})}

  local mlp = feedforwardFactory(param)

  trainerPool.full_CG(taData, mlp)

  local teOutput = mlp:forward(teInput)
  gnuplot.plot({'1', teTarget:select(2, 1, 1):squeeze(), teOutput:select(2, 1, 1):squeeze(), 'points pt 2 ps 0.4'})
  --]]

end

function test2()
-- 1.0063844,-0.9400020,-18.2620732,0.1743622,0.2879459,2.7897872,-0.0112311,-0.0855904,1.1829785,0.4917303,-5.2789840,0.6467680,-14.2722647
  torch.manualSeed(1)

  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()
  local taData = grnnUtil.getTable(teInput, teTarget)

  local param = { g6w = torch.Tensor({{1.0063844,-0.9400020,-18.2620732,0.1743622}}),
                  g7w = torch.Tensor({0.2879459,2.7897872,-0.0112311,-0.0855904,1.1829785,0.4917303,-5.2789840,0.6467680,-14.2722647})}

  local mlp = feedforwardFactory(param)

  print(myUtil.getCsvStringFrom2dTensor(param.g6w))
  print(myUtil.getCsvStringFrom1dTensor(param.g7w))

--  trainerPool.full_CG(taData, mlp)
  local testErr = trainerPool.test({teInput, teTarget}, mlp)
  local teOutput = mlp:forward(teInput)
  gnuplot.plot({'1', teTarget:select(2, 1):squeeze(), teOutput:select(2, 1):squeeze(), 'points pt 2 ps 0.4'},
               {'2', teTarget:select(2, 2):squeeze(), teOutput:select(2, 2):squeeze(), 'points pt 2 ps 0.4'})
  --gnuplot.plot({'1', teTarget:select(2, 2):squeeze(), teOutput:select(2, 2):squeeze(), 'points pt 2 ps 0.4'})
  print("testError: " .. testErr)

end

--plot1()
test1()
--test2()

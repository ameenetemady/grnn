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

end

function test2(strFigureFilename)
  torch.manualSeed(1)

  local taTrain, taTest = dataLoad.loadTrainTest()

  local teInput = taTest[1]
  local teTarget= taTest[2]

  local param = { g6w = torch.Tensor({{0.8085739,-0.8835572,-27.3594966,0.1615991}}),
                  g7w = torch.Tensor({-0.7585765,1.4642636,-0.7310677,-0.0855904,1.2089421,0.0014355,-4.5913864,0.6653003,-6.1370726})}

  local mlp = feedforwardFactory(param)

  print(myUtil.getCsvStringFrom2dTensor(param.g6w))
  print(myUtil.getCsvStringFrom1dTensor(param.g7w))

--  trainerPool.full_CG(taData, mlp)
  local testErr = trainerPool.test({teInput, teTarget}, mlp)
  local teOutput = mlp:forward(teInput)


  gnuplot.figure(1)
  gnuplot.raw('set terminal pdf')
  gnuplot.raw('set output "' .. strFigureFilename .. '"')
  gnuplot.raw('set xtics out nomirror; set ytics out nomirror; set border 3;set key reverse; set grid')
  gnuplot.raw('set xtics 0.25')
  gnuplot.raw('set ytics 0.25')

  gnuplot.xlabel("Predicted normalized mRNA level")
  gnuplot.ylabel("Observed normalized level")
--  gnuplot.title("Feedforward loop (incoherent type 4)")
  gnuplot.axis({0,1.05,0,1.05})
  gnuplot.movelegend('left', 'top')

  gnuplot.raw('set style circle radius graph 0.005')
  local yy = myUtil.getFilledCurve_identity(0, 1.1, 0.10)
  gnuplot.plot({yy, 'filledcurves fs transparent solid 0.3 noborder'},
               {'G2', teOutput:select(2, 1):squeeze(), teTarget:select(2, 1):squeeze(), 'circles fs transparent solid 0.6 noborder'},
               {'G3', teOutput:select(2, 2):squeeze(), teTarget:select(2, 2):squeeze(), 'circles fs transparent solid 0.6 noborder lc rgb "red"'})


  print("testError: " .. testErr)

end

--plot1()
--test1()
test2("feedforward1.pdf")

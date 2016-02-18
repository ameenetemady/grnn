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

  local mlp_g56 = nn.Concat(2)
  mlp_g56:add(nn.Identity)
  mlp_g56:add(syngOneAuto.new(param.g6w))

  local mlp_g5 = syngOneAuto.new(param.g5w)

  local main = nn.Sequential()
  main:add(mlp_g5)
  main:add(mlp_g56)

  return main
end

function test1()
  torch.manualSeed(1)

  local taTrain, taTest = dataLoad.loadTrainTest()
  local taData = grnnUtil.getTable(taTrain[1], taTrain[2])

  local taMinErr = { testErr = 99999999, trainErr = 9999999, param = nil, id = nil }

  for i=1,20 do
    print("*** i=" .. i .. " ***")
    local param = { g5w = torch.rand(1, 4)*2-1,
                    g6w = torch.rand(1, 4)*2-1}

    local mlp = feedforwardFactory(param)

    print(myUtil.getCsvStringFrom2dTensor(param.g5w))
    print(myUtil.getCsvStringFrom2dTensor(param.g6w))

    trainerPool.full_CG(taData, mlp)
    local trainErr = trainerPool.test(taTrain, mlp)
    local testErr = trainerPool.test(taTest, mlp)
     print("testError: " .. testErr)


    local paramOptim, __ = mlp:getParameters()
    print(myUtil.getCsvStringFrom1dTensor(paramOptim))

    if trainErr < taMinErr.trainErr then
      taMinErr.trainErr = trainErr
      taMinErr.testErr = testErr
      taMinErr.param = paramOptim
      taMinErr.id = i
    end
  end

  print(taMinErr)


end

function test2(strFigureFilename)
  local taTrain, taTest = dataLoad.loadTrainTest()

  local teInput = taTest[1]
  local teTarget= taTest[2]

--,  
  local param = { g5w = torch.Tensor({{-0.0629082,0.1962875,2.7342565,-0.9251734}}),
                  g6w = torch.Tensor({{-0.8252813,1.0768720,-1.2838694,0.4477465}})}

  local mlp = feedforwardFactory(param)
  local testErr = trainerPool.test(taTest, mlp)

  local teOutput = mlp:forward(teInput)

  gnuplot.figure(1)
  gnuplot.raw('set terminal pdf')
  gnuplot.raw('set output "' .. strFigureFilename .. '"')
  gnuplot.raw('set xtics out nomirror; set ytics out nomirror; set border 3;set key reverse; set grid')
  gnuplot.raw('set xtics 0.25')
  gnuplot.raw('set ytics 0.25')

  gnuplot.xlabel("prediction")
  gnuplot.ylabel("target")
  gnuplot.title("cascade of activator-respressor")
  gnuplot.axis({0,1.05,0,1.05})
  gnuplot.movelegend('left', 'top')

  gnuplot.raw('set style circle radius graph 0.005')
  gnuplot.plot({'G3', teOutput:select(2, 1):squeeze(), teTarget:select(2, 1):squeeze(), 'circles fs transparent solid 0.6 noborder'},
               {'G4', teOutput:select(2, 2):squeeze(), teTarget:select(2, 2):squeeze(), 'circles fs transparent solid 0.6 noborder lc rgb "red"'})


  print("testError: " .. testErr)

end

--plot1()

--test1()
test2("cascadeB.pdf")

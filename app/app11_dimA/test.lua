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

  gnuplot.plot({'1-2', teInput:select(2, 1):squeeze(), teTarget:select(2, 1):squeeze(), 'points pt 2 ps 0.4'},
               {'1-3', teInput:select(2, 1):squeeze(), teTarget:select(2, 2):squeeze(), 'points pt 2 ps 0.4'},
               {'5-3', teInput:select(2, 2):squeeze(), teTarget:select(2, 2):squeeze(), 'points pt 2 ps 0.4'},
               {'5-10', teInput:select(2, 2):squeeze(), teTarget:select(2, 3):squeeze(), 'points pt 2 ps 0.4'})
end

local strModelFilename = "dimA_mlp.model"

function feedforwardFactory(param)

  local mlp_g2 = nn.Sequential()
  mlp_g2:add(nn.Narrow(2, 1, 1))
  mlp_g2:add(syngOneAuto.new(param.g2w))

  local mlp_g3 = syngTwoAuto.new(param.g3w)

  local mlp_g10 = nn.Sequential()
  mlp_g10:add(nn.Narrow(2, 2, 1))
  mlp_g10:add(syngOneAuto.new(param.g10w))

  local main = nn.Concat(2)
  main:add(mlp_g2)
  main:add(mlp_g3)
  main:add(mlp_g10)

  return main
end

function test1()
  torch.manualSeed(1)

  local taTrain, taTest = dataLoad.loadTrainTest()
  local taData = grnnUtil.getTable(taTrain[1], taTrain[2])

  local taMinErr = { testErr = 99999999, trainErr = 9999999, param = nil, id = nil }

  for i=1,20 do
    print("*** i=" .. i .. " ***")
    local param = { g2w = torch.rand(1, 4)*2-1,
                    g3w = torch.rand(9)*2-1,
                    g10w = torch.rand(1, 4)*2-1}

    local mlp = feedforwardFactory(param)

    trainerPool.full_CG(taData, mlp)
    local trainErr = trainerPool.test(taTrain, mlp)
    local testErr = trainerPool.test(taTest, mlp)
     print("testError: " .. testErr)


    local paramOptim, __ = mlp:getParameters()

    if trainErr < taMinErr.trainErr then
      taMinErr.trainErr = trainErr
      taMinErr.testErr = testErr
      taMinErr.param = paramOptim
      taMinErr.id = i
      print("saving model " .. i .. " into " .. strModelFilename )
      torch.save(strModelFilename, mlp)
    end
  end


  print(taMinErr)
  print(myUtil.getCsvStringFrom1dTensor(taMinErr.param))

end

function test2(strFigureFilename)
  local taTrain, taTest = dataLoad.loadTrainTest()

  local teInput = taTest[1]
  local teTarget= taTest[2]


  local mlp = torch.load(strModelFilename)

  local teOutput = mlp:forward(teInput)

  gnuplot.figure(1)
  gnuplot.raw('set terminal pdf')
  gnuplot.raw('set output "' .. strFigureFilename .. '"')
  gnuplot.raw('set xtics out nomirror; set ytics out nomirror; set border 3;set key reverse; set grid')
  gnuplot.raw('set xtics 0.25')
  gnuplot.raw('set ytics 0.25')

  gnuplot.xlabel("Predicted normalized mRNA level")
  gnuplot.ylabel("Observed normalized mRNA level")
--  gnuplot.title("DIM circuit")
  gnuplot.axis({0,1.05,0,1.05})
  gnuplot.movelegend('left', 'top')

  gnuplot.raw('set style circle radius graph 0.005')
  gnuplot.plot({'G3', teOutput:select(2, 1):squeeze(), teTarget:select(2, 1):squeeze(), 'circles fs transparent solid 0.6 noborder lc rgb "red"'},
               {'G4', teOutput:select(2, 2):squeeze(), teTarget:select(2, 2):squeeze(), 'circles fs transparent solid 0.6 noborder lc rgb "green"'},
               {'G5', teOutput:select(2, 3):squeeze(), teTarget:select(2, 3):squeeze(), 'circles fs transparent solid 0.6 noborder lc rgb "blue"'})


  local testErr = trainerPool.test(taTest, mlp)
  print("testError: " .. testErr)

end

--plot1()
--test1()
test2("dimA.pdf")



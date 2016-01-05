require 'nn'
require('../../ConditionalFunUnit.lua')
require('../../Hill.lua')
require('../../GLogistic.lua')
require 'gnuplot'

local myUtil = require('../../../MyCommon/util.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../../grnnUtil.lua')
local strModelFileTest1 = "test1.model"

function plot1()
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()

  local mask = torch.eq(teInput:narrow(2,2, 1), 1)
  local teX = teInput:narrow(2, 1, 1):maskedSelect(mask)
  local teY = teTarget:narrow(2, 1, 1):maskedSelect(mask)

  gnuplot.plot({'1', teX, teY, 'points pt 2 ps 0.4'})
end

function plot2(i)
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()

  local mask = torch.eq(teInput:narrow(2,i+1, 1), 1)
  local teX = teTarget:narrow(2, i-1, 1):maskedSelect(mask)
  local teY = teTarget:narrow(2, i, 1):maskedSelect(mask)

  gnuplot.plot({'1', teX, teY, 'points pt 2 ps 0.4'})
end

function getCascadeInputTarget(geneID)
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()
  local mask = torch.Tensor()
  local teX = torch.Tensor()

  if geneID == 1 then
    mask = torch.eq(teInput:narrow(2,2, 1), 1)
    teX = teInput:narrow(2, 1, 1):maskedSelect(mask)
  else
    mask = torch.eq(teInput:narrow(2,geneID+1, 1), 1)
    teX = teTarget:narrow(2, geneID-1, 1):maskedSelect(mask)
  end

  local teY = teTarget:narrow(2, geneID, 1):maskedSelect(mask)

  return teX, teY
end

function plotModelOverData(geneID)
  local mlp = torch.load(strModelFileTest1)
  local glogistic = grnnUtil.getSubModel(mlp, geneID)

  local teX, teY = getCascadeInputTarget(geneID)
  gnuplot.plot({'1', teX, teY, 'points pt 2 ps 0.4'})
  local teModelX = torch.linspace(-.2, 1.2, 100)
  local teModelY = glogistic:forward(teModelX)
  gnuplot.plot({'Model', teModelX, teModelY, 'lines ls 1 lc rgb "red"'},
               {'data', teX, teY, 'points pt 2 ps 0.4 lc rgb "blue"'} )

end

function test1()
  local nGenes = 4
  -- 1) Load data
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()

  -- 2) Generate Model
  local initModelWeights = Cascade_getWeights_initModel()
  local fuFun = function(geneID)
    local weight = initModelWeights[geneID] 
    return  nn.GLogistic(weight)
  end

  local mlp = MultiLayer_ConditionalFunUnit(fuFun, nGenes)

  local taData = grnnUtil.getTable(teInput, teTarget)

  -- 3) train Model
  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)

  -- 4) report
  torch.save(strModelFileTest1 , mlp)

  local teOutput = mlp:forward(teInput):squeeze()
  print(teOutput)
end

function  Cascade_getWeights_initModel()
  local weight = torch.Tensor({{1.0, 1.0, 1, 1},
                              {1.0, 1.0, 1, 1},
                              {1.0, 1.0, 1, 1},
                              {1.0, 1.0, 1, 1}})
  return weight
end



--test1()
--for i=1, 1 do
--  plotModelOverData(i)
  plotModelOverData(4)
--end
--plot1()
--plot2(4)

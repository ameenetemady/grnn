require 'nn'
require 'gnuplot'
require('../../ConditionalFunUnit.lua')
require('../../Hill.lua')
require('../../GLogistic.lua')

local myUtil = require('../../../MyCommon/util.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../../grnnUtil.lua')

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

  print(teTarget)
  local taData = grnnUtil.getTable(teInput, teTarget)

  -- 3) train Model
  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)

  print(mlp:forward(teInput):squeeze())

end

function  Cascade_getWeights_initModel()
  local weight = torch.Tensor({{1.0, 1.0, 1, 0.1},
                              {1.0, 1.0, 1, 0.1},
                              {1.0, 1.0, 1, 0.1},
                              {1.0, 1.0, 1, 0.1}})
  return weight
end

test1()
--plot1()
--plot2(2)


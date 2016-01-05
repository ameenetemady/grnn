require 'nn'
require 'gnuplot'

local myUtil = require('../../../MyCommon/util.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../../grnnUtil.lua')

function test(nHiddenLayers)
  local nGenes = 4
  -- 1) Load data
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()

  -- 2) Generate Model
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(nGenes + 1, nGenes))
  mlp:add(nn.Sigmoid())

  for i=1, nHiddenLayers do
    mlp:add(nn.Linear(nGenes, nGenes))
    mlp:add(nn.Sigmoid())
  end


  local taData = grnnUtil.getTable(teInput, teTarget)

  -- 3) train Model
  trainerPool.full_CG(taData, mlp)

  local teOutput = mlp:forward(teInput):squeeze()
  print(teOutput)
end

--test(0)
--test(1)
--test(2)
test(3)

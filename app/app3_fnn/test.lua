require 'nn'
require 'gnuplot'

local myUtil = require('../../../MyCommon/util.lua')
local trainerPool = require('../../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../../grnnUtil.lua')

local strModelFileTest1 = "test1.model"
local strModelFileTest2 = "test2.model"


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

function test2(nHiddenLayers)
  local nGenes = 4
  -- Load data
  local taTrain, taTest = dataLoad.loadTrainTest()
  local taData = grnnUtil.getTable(taTrain[1], taTrain[2])
 
  -- 2) Generate Model
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(nGenes + 1, nGenes))
  mlp:add(nn.Sigmoid())

  for i=1, nHiddenLayers do
    mlp:add(nn.Linear(nGenes, nGenes))
    mlp:add(nn.Sigmoid())
  end


  -- 3) train Model / load Model
  trainerPool.full_CG(taData, mlp)
  torch.save(strModelFileTest2 , mlp)
--  mlp = torch.load(strModelFileTest2)

  -- 4) test Model
  local testErr = trainerPool.test(taTest, mlp)
  print("testError: " .. testErr)

end


--test1(0)
--test1(1)
--test1(2)
--test1(3)

test2(0)

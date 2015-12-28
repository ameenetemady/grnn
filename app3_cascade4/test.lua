require 'nn'
require('../ConditionalFunUnit.lua')
require('../Hill.lua')

local myUtil = require('../../MyCommon/util.lua')
local trainerPool = require('../../MyCommon/trainerPool.lua')
local dataLoad = require('./dataLoad.lua')
local grnnUtil = require('../grnnUtil.lua')

function test1()
  local nGenes = 4
  -- 1) Load data
  local teInput = dataLoad.loadInput()
  local teTarget = dataLoad.loadTarget()

  -- 2) Generate Model
  local initModelWeights = Cascade_getWeights_initModel()
  local fuFun = function(geneID)
    local weight = initModelWeights[geneID] 
    return  nn.Hill(weight)
  end

  local mlp = MultiLayer_ConditionalFunUnit(fuFun, nGenes)

  local taData = grnnUtil.getTable(teInput, teTarget)

  -- 3) train Model
  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)

end

function  Cascade_getWeights_initModel()
  local weight = torch.Tensor({{1.0, 2.5, 4, 1.17},
                              {1.0, 2.5, 4, 1.17},
                              {1.0, 2.5, 4, 1.17},
                              {1.0, 2.5, 4, 1.17}})
  return weight
end



test1()

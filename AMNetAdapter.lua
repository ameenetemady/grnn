local AMNetAdapter = AMNetAdapter or torch.class("AMNetAdapter")
local myUtil = myUtil or require('../MyCommon/util.lua')

--just for testing:
function AMNetAdapter:test_addToWeights(dAdd)
  local taWeights = self:pri_getModelWeights()

  local taWeightsNew = {}
  for k, v in pairs(taWeights) do
    taWeightsNew[k] = torch.add(v:clone(), dAdd)
  end

  self.mNet, self.taFu = self.getNewMNet(taWeightsNew)
end

function AMNetAdapter:pri_getModelWeights()
  local taWeights = {}
  for k, v in pairs(self.taFu) do
    taWeights[k] = v.mGx:parameters()
  end

  return taWeights
end

function AMNetAdapter:reload()
  self.mNet, self.taFu = self.getNewMNet(self.taWeights)
end

function AMNetAdapter:setWeight(strGene, teWeight)
  if self.taWeights == nil then
    self.taWeights = {}
  end

  self.taWeights[strGene] = teWeight
end

function AMNetAdapter:getRaw()
  return self.mNet
end

-- ***************************
-- ****** Static Methods *****
-- ***************************
function AMNetAdapter.pri_cloneParams(taParam)
  return taParam
end

function AMNetAdapter.pri_cloneWeights(taWeights)
  if taWeights == nil then
    return nil
  end

  local taWeightsClone = {}
  for k, v in pairs(taWeights) do
    myUtil.updateTable(taWeightsClone[k], v)
  end

  return taWeightsClone
end



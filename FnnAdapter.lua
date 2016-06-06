local archFactory = archFactory or require('../MyCommon/archFactory.lua')
local FnnAdapter = FnnAdapter or torch.class("FnnAdapter")

function FnnAdapter:__init(taParam, mNet)
  self.taParam = taParam
  self.mNet = mNet or archFactory.mlp(taParam)
end

function FnnAdapter:getRaw()
  return self.mNet
end

function FnnAdapter:clone()
  local mNetClone = self.mNet:clone()
  return self.new(self.taParam, mNetClone)
end

function FnnAdapter:cloneNoWeight()
  return self.new(self.taParam)
end

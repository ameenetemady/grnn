local cDataLoad = cDataLoad or require('./cDataLoad.lua')

CDataLoader = torch.class("CDataLoader")

function CDataLoader:__init(exprSettings, isNoise, isBalanceTF)
  self.exprSettings = exprSettings
  self.isNoise = isNoise
  self.isBalanceTF = isBalanceTF
end

function CDataLoader:pri_getSelected(teData)
  if self.isBalanceTF ~= true then
    return teData
  end

  if self.teSelectedIdx == nil then --ToDo: do this ###################******************###########
    self.teSelectedIdx = nil
  end

end


function CDataLoader:load3dInput()
  local teData, taTFNames, taNonTFNames = cDataLoad.load3dInput(self.exprSettings, self.isNoise)

  return self:pri_getSelected(teData), taTFNames, taNonTFNames
end

function CDataLoader:load2dInput()
  return cDataLoad.load2dInput(self.exprSettings, self.isNoise)
end

function CDataLoader:loadTarget()
  return cDataLoad.loadTarget(self.exprSettings, self.isNoise)
end

local cDataLoad = cDataLoad or require('./cDataLoad.lua')
local dataLoad = dataLoad or require('../../../MyCommon/dataLoad.lua')

CDataLoader = torch.class("CDataLoader")

function CDataLoader:__init(exprSettings, isNoise, isBalanceTF, dMinDist)
  self.exprSettings = exprSettings
  self.isNoise = isNoise
  self.isBalanceTF = isBalanceTF
  self.dMinDist = dMinDist or 6
end

function CDataLoader:pri_loadAllCat()
    local strTFFilepath = self.isNoise and self.exprSettings.strTFsFilePath or self.exprSettings.strTFsNoNoiseFilePath
    local strKOFilepath = self.exprSettings.strKOsFilePath
    local strNonTFFilepath = self.isNoise and self.exprSettings.strNonTFsFilePath or self.exprSettings.strNonTFsNoNoiseFilePath

    local taTFs = cDataLoad.getData(strTFFilepath)
    local taKOs = cDataLoad.getData(strKOFilepath)
    local taNonTFs = cDataLoad.getData(strNonTFFilepath)

    return torch.cat({taTFs.teData, taNonTFs.teData, taKOs.teData}, 2)
end

function CDataLoader:pri_getBalancedIdx(teDataCat)
    local nRowsTotal = teDataCat:size(1)
    
    local taIdx = {}
    table.insert(taIdx, 1)
    local nLastAddedId = 1
    for i=2, nRowsTotal do
      local dDist = torch.dist(teDataCat:select(1, nLastAddedId), 
                                teDataCat:select(1, i))

      if dDist > self.dMinDist then
        table.insert(taIdx, i)
        nLastAddedId = i
      end
    end

    return torch.LongTensor(taIdx)
end

function CDataLoader:pri_getSelectedGivenIdx(teData, teIdx)
  local teMask = torch.ByteTensor(teData:size(1)):fill(0)
  teMask:indexFill(1, teIdx, 1)

  return dataLoad.getMaskedSelect(teData, teMask)
end



function CDataLoader:pri_getSelected(teData)
  if self.isBalanceTF ~= true then
    return teData
  end

  if self.teSelectedIdx == nil then
    local teDataCat = self:pri_loadAllCat()
    self.teSelectedIdx = self:pri_getBalancedIdx(teDataCat)
  end

  return self:pri_getSelectedGivenIdx(teData, self.teSelectedIdx)
end


function CDataLoader:load3dInput()
  local teData, taTFNames, taNonTFNames = cDataLoad.load3dInput(self.exprSettings, self.isNoise)

  return self:pri_getSelected(teData), taTFNames, taNonTFNames
end

function CDataLoader:load2dInput()
  local teData, taTFNames, taKONames = cDataLoad.load2dInput(self.exprSettings, self.isNoise)

  return self:pri_getSelected(teData), taTFNames, taNonTFNames
end

function CDataLoader:loadTarget()
  local teData, taNonTFNames = cDataLoad.loadTarget(self.exprSettings, self.isNoise)

  return self:pri_getSelected(teData), taNonTFNames
end

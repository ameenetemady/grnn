if not isRequireBaseUnits then
  require 'nn'
  require('./torchNew/Squeeze.lua')
  require('./torchNew/Unsqueeze.lua')
  require('./CMulNoParamBatch.lua')
  require('./ClonableUnit.lua')
  require('./AMNetAdapter.lua')
  isRequireBaseUnits = true
end

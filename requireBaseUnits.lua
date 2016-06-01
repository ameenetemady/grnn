if not isRequireBaseUnits then
  require 'nn'
  require('./torchNew/Squeeze.lua')
  require('./torchNew/Unsqueeze.lua')
  require('./CMulNoParamBatch.lua')
  require('./ClonableUnit.lua')
  isRequireBaseUnits = true
end

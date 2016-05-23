require 'nn'
require('./addedFramework.lua')

local grnnArchUnits = {}

do
  function grnnArchUnits.dGx(nfArgs, fu)
    local mRes = nn.Sequential()
    mRes:add(nn.Narrow(3, 1, nfArgs))
    mRes:add(nn.Squeeze(3, 3))
    mRes:add(fu())

    return mRes
  end

  function grnnArchUnits.cGx(nfArgs, fu)

  end

  return grnnArchUnits
end

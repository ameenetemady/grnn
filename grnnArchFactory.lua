local syngTwoAuto = require('./SyngTwoAuto.lua')
local syngOneAuto = require('./SyngOneAuto.lua')

do
  local archFactory = {}

  function archFactory.feedforward(param)
    param = param or {}

    local layer_SyngOne = nn.Concat(2)
    layer_SyngOne:add(nn.Identity())
    layer_SyngOne:add(syngOneAuto.new(param.wSyngOne))

    local layer_SyngTwo = nn.Concat(2)
    layer_SyngTwo:add(nn.Identity())
    layer_SyngTwo:add(syngTwoAuto.new(param.wSyngTwo))

    local main = nn.Sequential()
    main:add(layer_SyngOne)
    main:add(layer_SyngTwo)
    main:add(nn.Narrow(2, 2, 2))
    
    return main
  end

  return archFactory
end

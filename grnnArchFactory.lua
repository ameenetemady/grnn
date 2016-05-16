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

  function archFactory.dimA(param)
    param = param or {}


    local mlp_g2 = nn.Sequential()
    mlp_g2:add(nn.Narrow(2, 1, 1))
    mlp_g2:add(syngOneAuto.new(param.g2w))

    local mlp_g3 = syngTwoAuto.new(param.g3w)

    local mlp_g10 = nn.Sequential()
    mlp_g10:add(nn.Narrow(2, 2, 1))
    mlp_g10:add(syngOneAuto.new(param.g10w))

    local main = nn.Concat(2)
    main:add(mlp_g2)
    main:add(mlp_g3)
    main:add(mlp_g10)

    return main

  end

  return archFactory
end

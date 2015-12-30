require 'nn'

local grnnUtil = {}
local myUtil = require('../MyCommon/util.lua')

do
  function grnnUtil.getTable(teX, teY)
    local nSize = teX:size(1)
    local taData = { n = nSize}
    myUtil.pri_addSize(taData)

    for i=1, nSize do
      if teY:dim() == 1 then
        table.insert(taData, { teX[i], torch.Tensor(1):fill(teY[i]) })
      else
        table.insert(taData, { teX[i], teY[i]})
      end
    end

    return taData
  end

  function grnnUtil.logParams(model)

    local parameters, gradParams = model:getParameters()
    print("parameters:")
    print(parameters)
  end

  function grnnUtil.getRandInput(nSize, nGenes)
    local nWidth = (nGenes or 1) + 1

    local teX = torch.rand(nSize, nWidth)
    teX:select(2, 1):mul(10)

    for i=2, nWidth do
      teX:select(2, i):round()
    end

    return teX
  end

  function grnnUtil.getSeqConModule(m1, m2)
    local seq = nn.Sequential()
    seq:add(m1)

    local con = nn.Concat(2)
    con:add(nn.Identity())
    con:add(m2)

    seq:add(con)

    return seq
  end

  return grnnUtil

end



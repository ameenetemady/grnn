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

  return grnnUtil

end



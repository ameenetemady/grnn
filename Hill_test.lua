require 'nn'
require('./Hill.lua')
require 'gnuplot'

local myUtil = require('../ANN/common/util.lua')
local trainerPool = require('./trainerPool.lua')

local Hill_test = {}

function Hill_test.hill_getOutput_test1()
  local input = torch.Tensor({1, 2, 3})
  local weight = torch.Tensor({1, 2, 3, 2})
  local output = hill_getOutput(input, weight)

  print(output)
end

function Hill_test.forward_test1()
  local weight = torch.Tensor({1, 2, 3, 2})
  local hill = nn.Hill(weight)

  local input = torch.Tensor({1, 2, 3})
  local output = hill:forward(input)

  print(output)

end

function Hill_test.updateGradInput_test1()

  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({1, 2, 3, 2})
  local hill = nn.Hill(weight)

  local input = torch.Tensor({1, 2, 3})
  local target = torch.Tensor({2.8, 2.3, 2})

  local output = hill:forward(input)
  print(output)

  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = hill:updateGradInput(input, df_do)


  print(target)
  print(gradInput)

end


function  Hill_test.accGradParameters_test1()
  local criterion = nn.MSECriterion()
  local weight = torch.Tensor({1, 2, 3, 2})
  local hill = nn.Hill(weight)

  local input = torch.Tensor({1, 2, 3})
  local target = torch.Tensor({2.8, 2.3, 2})

  local output = hill:forward(input)

  local f = criterion:forward(output, target)

  -- estimate df/dW
  local df_do = criterion:backward(output, target)
  local gradInput = hill:updateGradInput(input, df_do)

  local _, gradParams = hill:getParameters()
  print(gradParams)
  hill:accGradParameters(input, df_do, 1)
  _, gradParams = hill:getParameters()
  print(gradParams)
end

-- Train b, a, k parameters using single hill
function  Hill_test.accGradParameters_test2() 
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight = torch.Tensor({1, 2, 3, 2})
  local nSize = 100 

  local taData = Hill_test.genHillData1(synthWeight, nSize)
  
  local initWeight = torch.Tensor({2, 10, 1, 2}) -- modify b from 1 to 2
  local hill = nn.Hill(initWeight)

  Hill_test.logParams(hill)
  trainerPool.full_CG(taData, hill)
  Hill_test.logParams(hill)

  print("expected parameters")
  print(synthWeight)

end

-- Train b, a, k parameters using two sequential hills
function  Hill_test.accGradParameters_test3() 
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight1 = torch.Tensor({1, 2, 3, 2})
  local synthWeight2 = torch.Tensor({1, 2, 3, 2})
  local nSize = 1000

  local taData, synthModel = Hill_test.genHillData2(synthWeight1, synthWeight2, nSize)
  
  local initWeight1 = torch.Tensor({1, 2, 3, 2}) -- modify b from 1 to 2
  local initWeight2 = torch.Tensor({1, 3, 3, 2}) -- modify b from 1 to 2
  local hill1 = nn.Hill(initWeight1)
  local hill2 = nn.Hill(initWeight2)
  local model = nn.Sequential()
  model:add(hill1)
  model:add(hill2)

  Hill_test.logParams(model)
  trainerPool.full_CG(taData, model)
  Hill_test.logParams(model)

  print("expected parameters")
  local synthParams, _ = synthModel:getParameters()
  print(synthParams)

end

-- train with multiple outputs
function Hill_test.accGradParameters_test4()
  torch.manualSeed(1)
  local criterion = nn.MSECriterion()
  local synthWeight1 = torch.Tensor({1, 2, 3, 2})
  local synthWeight2 = torch.Tensor({5, 4, 6, 2})
  local nSize = 100

  local taData, synthModel = Hill_test.genHillData3(synthWeight1, synthWeight2, nSize)


  local weight1 = torch.Tensor({1.5, 2.2, 3, 2})
  local weight2 = torch.Tensor({0.1, 2.5, 3, 2})
  local hill1 = nn.Hill(weight1)
  local hill2 = nn.Hill(weight2)

  local model = Hill_test.getSeqConModule(hill1, hill2)


  Hill_test.logParams(model)
  trainerPool.full_CG(taData, model)
  Hill_test.logParams(model)

  print("expected parameters")
  local synthParams, _ = synthModel:getParameters()
  print(synthParams)

end

function Hill_test.getSeqConModule(m1, m2)
  local seq = nn.Sequential()
  seq:add(m1)

  local con = nn.Concat(2)
  con:add(nn.Identity())
  con:add(m2)

  seq:add(con)

  return seq
end

function Hill_test.genHillData3(weight1, weight2, nSize)
  local teX = torch.rand(nSize,1)*100
  local hill1 = nn.Hill(weight1)
  local hill2 = nn.Hill(weight2)

  local seq = Hill_test.getSeqConModule(hill1, hill2)
  local teY= seq:forward(teX)

  print(teY)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { torch.Tensor(1):copy(teX[i]), teY[i] }
    table.insert(taData, taRow)
  end

  return taData,seq 
--  gnuplot.plot('synthetic', teX, output, 'points pt 2 ps 0.4')
--  print(output)
end


function Hill_test.logParams(model)

  local parameters, gradParams = model:getParameters()
  print("parameters:")
  print(parameters)
end

function Hill_test.genHillData2(weight1, weight2, nSize)
  local teX = torch.rand(nSize)*100
  local hill1 = nn.Hill(weight1)
  local hill2 = nn.Hill(weight2)

  local model = nn.Sequential()
  model:add(hill1)
  model:add(hill2)
  local teY= model:forward(teX)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { torch.Tensor(1):fill(teX[i]), torch.Tensor(1):fill(teY[i]) }
    table.insert(taData, taRow)
  end

  return taData, model
--  gnuplot.plot('synthetic', teX, output, 'points pt 2 ps 0.4')
--  print(output)
end


function Hill_test.genHillData1(weight, nSize)
  local teX = torch.rand(nSize)*5
  local hill = nn.Hill(weight)
  local teY= hill:forward(teX)

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { torch.Tensor(1):fill(teX[i]), torch.Tensor(1):fill(teY[i]) }
    table.insert(taData, taRow)
  end

  return taData
--  gnuplot.plot('synthetic', teX, output, 'points pt 2 ps 0.4')
--  print(output)
end

function  Hill_test.all()
--    Hill_test.E2EA()
--  Hill_test.hill_getOutput_test1()
--  Hill_test.forward_test1()
--  Hill_test.updateGradInput_test1()
--  Hill_test.accGradParameters_test1()
--  Hill_test.accGradParameters_test2()
--  Hill_test.accGradParameters_test3()
  Hill_test.accGradParameters_test4()
end

Hill_test.all()

require 'nn'
autograd = require 'autograd'
require('./SyngTwo.lua')
require 'gnuplot'

local myUtil = require('../MyCommon/util.lua')
local trainerPool = require('../MyCommon/trainerPool.lua')
local grnnUtil = require('./grnnUtil.lua')


local SyngTwo_test = {}

function SyngTwo_test.SyngTwo_getOutput_test1()
  local input = torch.Tensor({ {1, 0}, {0, 1}, {1, 1} })
  local weight = torch.Tensor({1, 2, 3, 4, 2})

  local output =  SyngTwo_getOutput(input, weight)

  local expected = torch.Tensor({0.0833, 0.1667, 0.1792})
  assert( torch.sum(expected - output) < 0.0001, "output far from expected!")
  print("PASS:" .. debug.getinfo(1, "n").name);
end

function SyngTwo_test.SyngTwo_updateOutput_test1()
  local input = torch.Tensor({ {1, 0}, {0, 1}, {1, 1} })
  local weight = torch.Tensor({1, 2, 3, 4, 2})

  local mlp = nn.SyngTwo(weight)
  local output = mlp:forward(input)

  local expected = torch.Tensor({0.0833, 0.1667, 0.1792})
  assert( torch.sum(expected - output) < 0.0001, "output far from expected!")
  print("PASS:" .. debug.getinfo(1, "n").name);

end

function SyngTwo_test.SyngTwo_updateGradInput_test1()
  local input = torch.Tensor({ {1, 0}, {0, 1}, {2, 1} })
  local weight = torch.Tensor({1, 2, 3, 4, 2})

  local mlp = nn.SyngTwo(weight)
  local output = mlp:forward(input)
  local gradOutput = torch.Tensor({1, -1, 10})

  local gradInput = mlp:updateGradInput(input, gradOutput)

  local expected_3 = torch.Tensor({0.0413, 0.5785})
  local actual_3 = gradInput:narrow(1, 3, 1)
  assert( torch.sum(expected_3 - actual_3) < 0.0001, "output far from expected!")
  print("PASS:" .. debug.getinfo(1, "n").name);
end

local fuAutoSyngTwoFull = function(input, w, bias)
  local x1 = torch.select(input, 2, 1)
  local x2 = torch.select(input, 2, 2)

  local y1 = torch.exp(torch.mul(torch.add(torch.mul(x1, -1), w[6]), w[7]))
  local y2 = torch.exp(torch.mul(torch.add(torch.mul(x2, -1), w[8]), w[9]))

  local comb = torch.mul(torch.cmul(y1, y2), w[5])
  local top_comb = torch.mul(comb, w[6])
  local top_y1 = torch.mul(y1, w[2])
  local top_y2 = torch.mul(y2, w[3])

  local top = torch.add(torch.add(torch.add(top_comb, top_y2), top_y1), w[1])

  local but = torch.add(torch.add(torch.add(comb, y2), y1), 1)
  
  local output = torch.cdiv(top, but)


  return output
end

function SyngTwo_test.AutoSyngTwo_test1()
  torch.manualSeed(1)

  local synthWeight = torch.Tensor({0, 1, 3, 0.5, 0.5, 1, 3, 1, 3})
  local nSize = 500

  local taData = genData1(synthWeight, nSize)


 -- --[[
  
  local weight = torch.Tensor({1.5, 0, 0, 0, 0, 2, 3, 2, 3})
  local mlp = autograd.nn.AutoModule('AutoSyngTwoFull')(fuAutoSyngTwoFull, weight:clone())

  grnnUtil.logParams(mlp)
  trainerPool.full_CG(taData, mlp)
  grnnUtil.logParams(mlp)


  print("expected parameters")
  print(synthWeight)
  --]]

end

function genData1(synthWeight, nSize)
  local teX = torch.rand(nSize, 2)*2
  local mlp = autograd.nn.AutoModule('AutoSyngTwoFull')(fuAutoSyngTwoFull, synthWeight:clone())
  local teY = mlp:forward(teX):reshape(nSize, 1)


--  gnuplot.plot({'x1', teX:select(2,1):squeeze(), teY:squeeze(), 'points pt 2 ps 0.4'},
--                {'x2', teX:select(2,2):squeeze(), teY:squeeze(), 'points pt 2 ps 0.2 lc rgb "red"'})

  print(teY:size())

  local taData = { n = nSize}
  myUtil.pri_addSize(taData)

  for i=1, nSize do
    local taRow = { teX[i], teY[i] }
    table.insert(taData, taRow)
  end

  return taData
end


function SyngTwo_test.all()
-- SyngTwo_test.SyngTwo_getOutput_test1()
-- SyngTwo_test.SyngTwo_updateOutput_test1()
-- SyngTwo_test.SyngTwo_updateGradInput_test1()
  SyngTwo_test.AutoSyngTwo_test1()
end

SyngTwo_test.all()

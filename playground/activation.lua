require 'gnuplot'
require 'nn'
require('../Hill.lua')

local activation = {}

function activation.hill_test1()
  local input = torch.linspace(-10, 10, 200)
  local output1 = hill_getOutput(input, torch.Tensor({0, 1, 1, 2}))
  local output2 = hill_getOutput(input, torch.Tensor({0, 1, 1, 2}))

  gnuplot.plot({'1', input, output1, 'points pt 2 ps 0.4'}, 
               {'2', input, output2, 'points pt 2 ps 0.2 lc rgb "red"'})
end

function myLogistic_extract_params(weight)
  return weight[1], weight[2], weight[3], weight[4]
end


function activation.myLogistic(input, weight)
  local output = input:clone()

  local a, b, c, d = myLogistic_extract_params(weight)
  --[[ if repressor: 
  --    a =>0: defines the max
  --    b <=0: defines the slope
  --    c: defines the shift \in R
  --    d: defines the minimum

  --]]
  output:apply(
    function(x)
      local denominator = 1 + math.exp(b*(c-x))
      assert( denominator ~= 0, "(x/k) cannot be -1")
      local y =  a/denominator + d
      return y
    end)

  return output
end

function activation.myLogistic_test1()
  local input = torch.linspace(-20, 20, 200)
  local output1 = activation.myLogistic(input, torch.Tensor({10, -0.2, -10, 1}))
  local output2 = activation.myLogistic(input, torch.Tensor({1, -2, 4, 1}))

  gnuplot.plot({'1', input, output1, 'points pt 2 ps 0.4'}, 
               {'2', input, output2, 'points pt 2 ps 0.2 lc rgb "red"'})
end


function activation.logisticNew(teInput, taWeights)
  local teX1 = teInput:select(2, 1)
  local teX2 = teInput:select(2, 2)

  local teH1 = torch.exp(torch.mul(teX1, taWeights.p1))
  local teH2 = torch.exp(torch.mul(teX2, taWeights.p2))

  local teTop = torch.add(torch.add(torch.mul(teH1, taWeights.t1), 
                                    torch.mul(teH2, taWeights.t2)),
                        taWeights.t0)

  local teBut = torch.add(torch.add(torch.mul(teH1, taWeights.b1),
                                    torch.mul(teH2, taWeights.b2)),
                          1)

  return torch.cdiv(teTop,teBut)
end

function activation.logisticNew_test1_x1Zero()
  local nRows = 200
  local taWeights = { t0=0, t1=1, t2=1, b1=1, b2=1.5, p1=1, p2=-1 }

  local input1 = torch.rand(nRows, 2)*10 - 5 
  input1:select(2, 1):fill(0)
  local output1 = activation.logisticNew(input1, taWeights)

  input2 = torch.rand(nRows, 2)*10 - 5 
  input2:select(2, 2):fill(0)
  output2 = activation.logisticNew(input2, taWeights)

  gnuplot.plot({'1', input1:select(2, 2), output1, 'with points pt 2 ps 0.4'}, 
               {'2', input2:select(2, 1), output2, 'with points pt 2 ps 0.2 lc rgb "red"'})

  print("plotted!?")
end

function  activation.all()
--activation.hill_test1()
--activation.myLogistic_test1()
--  activation.logisticNew_test1_x1Zero()
end

activation.all()

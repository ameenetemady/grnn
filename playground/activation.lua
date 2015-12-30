require 'nn'
require('../Hill.lua')
require 'gnuplot'

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

function  activation.all()
--activation.hill_test1()
activation.myLogistic_test1()
end

activation.all()

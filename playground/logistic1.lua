require 'gnuplot'
require 'nn'
local grad = require 'autograd'

local logistic = {}

function logistic.logisticNew(teInput, taWeights)
  local teX1 = teInput:narrow(2, 1, 1)
  local teX2 = teInput:narrow(2, 2, 1)

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

function logistic.plotZeros(taWeights, plotId)
  local plotId = plotId or 1
  local nRows = 200

  local input1 = torch.rand(nRows, 2)*10 - 5 
  input1:select(2, 1):fill(0)
  local output1 = logistic.logisticNew(input1, taWeights)

  input2 = torch.rand(nRows, 2)*10 - 5 
  input2:select(2, 2):fill(0)
  output2 = logistic.logisticNew(input2, taWeights)

  gnuplot.figure(plotId)
  gnuplot.title(string.format("%d", plotId))
  gnuplot.plot({'1', input1:select(2, 2), output1, 'with points pt 2 ps 0.4'}, 
               {'2', input2:select(2, 1), output2, 'with points pt 2 ps 0.2 lc rgb "red"'})

end

function logistic.test1()
  -- generate data:
  local nRows = 200
  local taWeights = { t0=0, t1=1, t2=1, b1=1, b2=1.5, p1=1, p2=-1 }
  local teX = torch.rand(nRows, 2)*10 - 5 
  local teY = logistic.logisticNew(teX, taWeights)

  -- fit:
  -- a) initialize p1, p2
  local p1=1.5; local p2=-1.5

  -- goal: argMin |Ax-B|
  -- b) construct A
  local teX1 = teX:select(2, 1)
  local teX2 = teX:select(2, 2)

  local teH1 = torch.exp(torch.mul(teX1, p1))
  local teH2 = torch.exp(torch.mul(teX2, p2))

  local teA = torch.cat({torch.ones(nRows, 1),
                        teH1,
                        torch.mul(torch.cmul(teY, teH1), -1),
                        teH2,
                        torch.mul(torch.cmul(teY, teH2), -1)})

  -- c) construct B
  local teB = torch.Tensor(nRows, 1)
  teB:select(2, 1):copy(teY)

  local teWeights = torch.gels(teB, teA):squeeze()
  local taWeightsEst ={ t0 = teWeights[1], t1 = teWeights[2], t2 = teWeights[4], b1 = teWeights[3], b2 = teWeights[5], p1=p1, p2=p2 }
  print("orig")
  print(taWeights)
  
  logistic.plotZeros(taWeights, 1)

  print("est")
  print(taWeightsEst)
  logistic.plotZeros(taWeightsEst, 2)

end

function logistic.test2_autograd()
  local params = { p=torch.Tensor({1.0,-1.1 }) }

  fuLoss = function(params, teW, teX, teY)
    local nRows = torch.size(teY, 1)

    -- Loss function: |Aw-B|
    
    -- a) construct A
    local teX1 = torch.narrow(teX, 2, 1, 1)
    local teX2 = torch.narrow(teX, 2, 2, 1)

    local teH1 = torch.exp(torch.mul(teX1, params.p[1]))
    local teH2 = torch.exp(torch.mul(teX2, params.p[2]))

    local teA = torch.cat({torch.ones(nRows, 1),
                          teH1,
                          torch.mul(torch.cmul(teY, teH1), -1),
                          teH2,
                          torch.mul(torch.cmul(teY, teH2), -1)})

    -- b) construct B
    local teB = torch.Tensor(nRows, 1)
    torch.select(teB, 2, 1):copy(teY)


    local teResBase = torch.add(torch.bmm(torch.view(teA, 1, teA:size(1), teA:size(2)), 
                                          torch.view(teW, 1, teW:size(1), teW:size(2))),
                               torch.mul(teB, -1))

--    teResBase = torch.view(teResBase, nRows, 1)

    local dRes = torch.bmm(torch.transpose(teResBase, 2, 3), teResBase)
    return dRes[1][1][1]
  end

  local nRows = 200
  local taWeights = { t0=0, t1=1, t2=1, b1=1, b2=1.5, p1=1, p2=-1 }
  local teX = torch.rand(nRows, 2)*10 - 5 
  local teY = logistic.logisticNew(teX, taWeights)

--  local dLoss = fuLoss(params, taWeights, teX, teY)
--  print(dLoss)

  local fuGradLoss = grad(fuLoss,{
    debugHook = function(debugger, msg)
      debugger.showDot()
      print(msg)
      os.exit(0)
    end
  })

  local teW = torch.Tensor({{taWeights.t0}, 
                            {taWeights.t1},
                            {taWeights.b1},
                            {taWeights.t2},
                            {taWeights.b2}})
  local teGradParams, loss = fuGradLoss(params, teW, teX, teY)

  print(teGradParams.p)
  print(loss)
end

function  logistic.all()
  --logistic.logisticNew_test1_x1Zero()
  --logistic.test1()
  logistic.test2_autograd()
end

logistic.all()

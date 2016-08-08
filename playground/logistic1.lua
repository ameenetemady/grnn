require 'gnuplot'
require 'nn'
require 'optim'
local myUtil = require('../../MyCommon/util.lua')

local grad = require 'autograd'

local logistic = {}

-- { t0 = teWeights[1], t1 = teWeights[2], t2 = teWeights[4], b1 = teWeights[3], b2 = teWeights[5], p1=teP[1], p2=teP[2] }
function logistic.logisticNewTe(teInput, teWeights)
  local t0 = teWeights[1]; 
  local t1 = teWeights[2]; local t2 = teWeights[4]
  local b1 = teWeights[3]; local b2 = teWeights[5]
  local p1 = teWeights[6]; local p2 = teWeights[7]


  local teX1 = torch.narrow(teInput, 2, 1, 1)
  local teX2 = torch.narrow(teInput, 2, 2, 1)

  local teH1 = torch.exp(torch.mul(teX1, p1))
  local teH2 = torch.exp(torch.mul(teX2, p2))

  local teTop = torch.add(torch.add(torch.mul(teH1, t1), 
                                    torch.mul(teH2, t2)),
                        t0)

  local teBut = torch.add(torch.add(torch.mul(teH1, b1),
                                    torch.mul(teH2, b2)),
                          1)

  return torch.cdiv(teTop,teBut)
end

function logistic.logisticNew(teInput, taWeights)
  local teX1 = torch.narrow(teInput, 2, 1, 1)
  local teX2 = torch.narrow(teInput, 2, 2, 1)

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

function logistic.plotZeros(taWeights, plotId, xmin, xmax, ymin, ymax)
  local plotId = plotId or 1
  local nRows = 200

  local input1 = torch.rand(nRows, 2)*5 
  input1:select(2, 1):fill(0)
  local output1 = logistic.logisticNew(input1, taWeights)

  input2 = torch.rand(nRows, 2)*5 
  input2:select(2, 2):fill(0)
  output2 = logistic.logisticNew(input2, taWeights)

  gnuplot.figure(plotId)
  gnuplot.title(string.format("%d", plotId))
  gnuplot.plot({'1', input1:select(2, 2), output1, 'with points pt 2 ps 0.4'}, 
               {'2', input2:select(2, 1), output2, 'with points pt 2 ps 0.2 lc rgb "red"'})


  xmin = xmin or ''
  xmax = xmax or ''
  ymin = ymin or ''
  ymax = ymax or ''
  gnuplot.axis({xmin, xmax, ymin, ymax})

  return nil, nil, math.min(output1:min(), output2:min()), math.max(output1:max(), output2:max())

end

function logistic.getOptimGels(teX, teY, teP)
  -- goal: argMin |Ax-B|
  -- b) construct A
  local teX1 = teX:select(2, 1)
  local teX2 = teX:select(2, 2)

  local teH1 = torch.exp(torch.mul(teX1, teP[1]))
  local teH2 = torch.exp(torch.mul(teX2, teP[2]))

  local nRows = torch.size(teY, 1)
  local teA = torch.cat({torch.ones(nRows, 1),
                        teH1,
                        torch.mul(torch.cmul(teY, teH1), -1),
                        teH2,
                        torch.mul(torch.cmul(teY, teH2), -1)})

  -- c) construct B
  local teB = torch.Tensor(nRows, 1)
  teB:select(2, 1):copy(teY)

  local teWeights = torch.gels(teB, teA):squeeze()
  local taWeightsEst ={ t0 = teWeights[1], t1 = teWeights[2], t2 = teWeights[4], b1 = teWeights[3], b2 = teWeights[5], p1=teP[1], p2=teP[2] }

  return taWeightsEst
end

function logistic.test1()
  -- generate data:
  local nRows = 200
  local taWeights = { t0=0, t1=1, t2=1, b1=1, b2=1.5, p1=1, p2=-1 }
  local teX = torch.rand(nRows, 2)*5 
  local teY = logistic.logisticNew(teX, taWeights)

  -- fit:
  -- a) initialize p1, p2
  local p1=1.2; local p2=-1.2
  local taWeightsEst = logistic.getOptimGels(teX, teY, torch.Tensor({p1, p2}))

  print("orig")
  print(taWeights)
  
  logistic.plotZeros(taWeights, 1)

  print("est")
  print(taWeightsEst)
  logistic.plotZeros(taWeightsEst, 2)

end

function fuLoss(myParams1, teW, teX, teY)

    local nRows = torch.size(teY, 1)

    -- Loss function: |Aw-B|
    
    -- a) construct A
    local teX1 = torch.narrow(teX, 2, 1, 1)
    local teX2 = torch.narrow(teX, 2, 2, 1)

    local teH1 = torch.exp(torch.mul(teX1, myParams1[1]))
    local teH2 = torch.exp(torch.mul(teX2, myParams1[2]))

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

    local dRes = torch.bmm(torch.transpose(teResBase, 2, 3), teResBase)[1][1][1]

    --adding regularization here:
    --todo: try this with fuLoss2 instead:
    local dLambda= 0.005
    dRes = dRes + dLambda * torch.sum(torch.pow(myParams1, 2)) 

    return dRes
  end

function fuLoss2(teP, teW, teX, teY)
-- ={ t0 = teW[1], t1 = teW[2], t2 = teW[4], b1 = teW[3], b2 = teW[5], p1=teP[1], p2=teP[2] }
  local teWeights = torch.cat(torch.select(teW, 2, 1), teP) 
  local nRows = torch.size(teY, 1)
  local teYRes = torch.add(logistic.logisticNewTe(teX, teWeights),
                           torch.mul(teY, -1))
  local teYRes3d = torch.view( teYRes, 1, nRows, 1)
--  local teYPred = torch.view(logistic.logisticNewTe(teX, teWeights), 1, nRows, 1)
  
  print("teYPred:size()")
  print(teYRes3d:size())

  local dRes = torch.bmm(torch.transpose(teYRes3d, 2, 3), teYRes3d)[1][1][1]

  local dLambda= 0.005
  dRes = dRes + dLambda * torch.sum(torch.pow(teP, 2)) 
  print("********:")
  print(dRes.value)

  return dRes
end


function logistic.test2_autograd()
  torch.manualSeed(1)

  local nRows = 200
  local taWeights = { t0=0, t1=1, t2=1, b1=1, b2=1.5, p1=1, p2=-1 }
  local teX = torch.rand(nRows, 2)*5 
  local teY = logistic.logisticNew(teX, taWeights)
  local teW = torch.Tensor({{taWeights.t0}, 
                            {taWeights.t1},
                            {taWeights.b1},
                            {taWeights.t2},
                            {taWeights.b2}})

  local myParams =  torch.Tensor({1.1,-1.9})

  local fuGradLoss = grad(fuLoss2)
  local teGradParams, loss = fuGradLoss(myParams, teW, teX, teY)
  print(teGradParams)
  print(loss)
end

function logistic.fuForOptim(teX, teY, teInitParam)
  -- a) assume const teInitParam (i.e. p1, p2) and get teW
 local taWeights = logistic.getOptimGels(teX, teY, teInitParam)
 local teW = torch.Tensor({{taWeights.t0}, 
                           {taWeights.t1},
                           {taWeights.b1},
                           {taWeights.t2},
                           {taWeights.b2}})
 
  -- b) assume const teW and calculate grad for p1, p2
  local fuGradLoss = grad(fuLoss)
  local teGradParams, loss = fuGradLoss(teInitParam, teW, teX, teY)

  return loss, teGradParams, taWeights
end

function  logistic.test3_autogradOptim()
  torch.manualSeed(1)

  local nRows = 20
  local taWeights = { t0=0.0, t1=1.5, t2=1, b1=1.5, b2=1.0, p1=-1.0, p2=1.0 }
  local teX = torch.rand(nRows, 2)*5 
  local teY = logistic.logisticNew(teX, taWeights)
  local teW = torch.Tensor({{taWeights.t0}, 
                           {taWeights.t1},
                           {taWeights.b1},
                           {taWeights.t2},
                           {taWeights.b2}})

  --local teInitParam =  torch.Tensor({1.5,-1.5})
  local teInitParam =  torch.Tensor({-1.5,1.5})

  --[[
  local taInitWeight = myUtil.shallowCopy(taWeights)
  taInitWeight.p1 = teInitParam[1]
  taInitWeight.p2 = teInitParam[2]
  local teYInitPred = logistic.logisticNew(teX, taInitWeight)
  print("teYInitPred:")
  print(teYInitPred)
  --]]

  local taWeightOptim = nil

  local fuEval = function(teParam)
    local loss, teGradParams, taCurrWeights = logistic.fuForOptim(teX, teY, teParam)
--    print(taCurrWeights)
    taWeightOptim= myUtil.shallowCopy(taCurrWeights)
--    print(taCurrWeights)
--    print(loss .. ", ")
    return loss, teGradParams
  end

  local teParamOptim, lossOptim = optim.cg(fuEval, teInitParam, {maxIter = 25})

--  print("taWeightOptim")
--  print(taWeightOptim)


  -- verify actual vs. prediction
  local teYPred = logistic.logisticNew(teX, taWeightOptim)
  print(torch.add(teY, -teYPred))


  local xmin, xmax, ymin, ymax = logistic.plotZeros(taWeights, 1)--,teX:min(), teX:max(), teY:min(), teY:max())
  logistic.plotZeros(taWeightOptim, 2, xmin, xmax, ymin, ymax)--,teX:min(), teX:max(), teY:min(), teY:max())

  print("teParamOptim:")
  print(teParamOptim)
  print(lossOptim)



--[[
  local initLoss = fuLoss(teInitParam, teW, teX, teY)
  print("initLoss:" ..  initLoss)
  local loss, teGradParams = logistic.fuForOptim(teX, teY, teInitParam)
  print("newLoss:" .. loss)
  print(teGradParams)
  --]]


end
function  logistic.all()
  --logistic.logisticNew_test1_x1Zero()
--logistic.test1()
--  logistic.test2_autograd()
  logistic.test3_autogradOptim()
end

logistic.all()


--]]

--[[

  local outerFn = function(myParams2, teW, teX, teY)
    local grad = fuGradLoss(myParams2, teW, teX, teY)
    return torch.sum(grad)
  end

  local fuGradGradLoss = grad(outerFn)
  local teGradGradParams = fuGradGradLoss(myParams, teW, teX, teY)
  print(teGradGradParams)
  --]]



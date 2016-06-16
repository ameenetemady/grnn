require 'nn'
require 'optim'
local myUtil = require('../MyCommon/util.lua')

do
  local trainerPool = {}

  function trainerPool.getDefaultTrainParams(nRows, strOptimMethod)

    local taTrainParam = {  --batchSize = 9, 
                            batchSize = math.floor(nRows),
                            criterion = nn.MSECriterion(),
                            nMaxIteration = 100,
                            coefL1 = 0.0,
                            coefL2 = 0.0,
                            strOptimMethod = strOptimMethod or "CG",
                            isLog = true,
                            taOptimParams = {}
                          }

    if taTrainParam.strOptimMethod == "SGD" then
      taTrainParam.taOptimParams = { 
        learningRate = 0.1,
        learningRateDecay = 0.9995,
        momentum = 0.1 }
      taTrainParam.fuOptim = optim.sgd
  
    elseif taTrainParam.strOptimMethod == "LBFGS" then
      taTrainParam.taOptimParams = { 
        maxIter = 100,
        lineSearch = optim.lswolfe }
      taTrainParam.fuOptim = optim.lbfgs

    elseif taTrainParam.strOptimMethod == "CG" then
      taTrainParam.taOptimParams = {
        maxIter = 20 }
      taTrainParam.fuOptim = optim.cg

    else
      error("invalid operation")
    end

    return taTrainParam
  end

  function trainerPool.pri_trainGrnn_SingleRound(mNet, teInput, teTarget, taTrainParam)
    parameters, gradParameters = mNet:getParameters()
    local criterion = taTrainParam.criterion
    local overallErr = 0
    local nRows = teInput:size(1)

    for t = 1,nRows, taTrainParam.batchSize do
      -- create batches
      --myUtil.log("batch first item:" .. t, true, taTrainParam.isLog)
      local nCurrBatchSize = math.min(taTrainParam.batchSize, nRows - t + 1)
      local teBatchX = teInput:narrow(1, t, nCurrBatchSize)
      local teBatchY = teTarget:narrow(1, t, nCurrBatchSize)

      local fuEval = function(x)
        collectgarbage()

        -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- evaluate function for the complete mini batch
        local teBatchPredY = mNet:forward(teBatchX)
        local f = criterion:forward(teBatchPredY, teBatchY)

        -- estimate df/dW
        local df_do = criterion:backward(teBatchPredY, teBatchY)
        mNet:backward(teBatchX, df_do)

       -- penalties (L1 and L2):
        if taTrainParam.coefL1 ~= 0 or taTrainParam.coefL2 ~= 0 then
          -- locals:
           local norm,sign= torch.norm,torch.sign
 
          -- Loss:
          f = f + taTrainParam.coefL1 * norm(parameters,1)
          f = f + taTrainParam.coefL2 * norm(parameters,2)^2/2

          -- Gradients:
          gradParameters:add( sign(parameters):mul(taTrainParam.coefL1) + parameters:clone():mul(taTrainParam.coefL2) )
        end
        
        overallErr = overallErr + f

        return f, gradParameters
      end --fuEval

      taTrainParam.fuOptim(fuEval, parameters, taTrainParam.taOptimParams)
    end

    return trainerPool.getErr(mNet, teInput, teTarget, taTrainParam)
  end

  function trainerPool.pri_serialize_deserialize(obj)
    local seriObj = torch.serialize(obj)
    local deSeriObj = torch.deserialize(seriObj)

    return deSeriObj
  end

  function trainerPool.getErr(mNet, teInput, teTarget, taTrainParam)
    local criterion = taTrainParam.criterion

    local teOutput = mNet:forward(teInput)
    local criterion = nn.MSECriterion()
    local fErr = criterion:forward(teOutput, teTarget)

    return fErr
  end

  function trainerPool.runCVExperiment(fuGetArch, teInput, teTarget, nFolds)
    --Todo: implement

  end

  function trainerPool.trainGrnn(mNet, teInput, teTarget)
    local criterion = nn.MSECriterion()
    local taTrainParam = trainerPool.getDefaultTrainParams(teInput:size(1),"CG" )

    local errPrev = math.huge
    local errCurr = math.huge

    for i=1, taTrainParam.nMaxIteration do
      errCurr = trainerPool.pri_trainGrnn_SingleRound(mNet, teInput, teTarget, taTrainParam)

      if errPrev <= errCurr or myUtil.isNan(errCurr)  then
        print("** early stop **")
        return errPrev
      elseif errCurr ~= nil then
        local message = errCurr < errPrev and "<" or "!>"
        myUtil.log(message, false, taTrainParam.isLog)
        errPrev = errCurr
      else
        error("invalid value for errCurr!")
      end

    end

    return errCurr
  end

  function trainerPool.trainGrnnMNetAdapter(mNetAdapter, teInput, teTarget, taTrainerParamsOverride)
    local criterion = nn.MSECriterion()
    local taTrainParam = trainerPool.getDefaultTrainParams(teInput:size(1),"CG" )
    myUtil.updateTable(taTrainParam, taTrainerParamsOverride)

    local errPrev = math.huge
    local mNetAdapterPrev = nil
    local errCurr = math.huge

    for i=1, taTrainParam.nMaxIteration do
      local mNetRaw = mNetAdapter:getRaw()
      errCurr = trainerPool.pri_trainGrnn_SingleRound(mNetRaw, teInput, teTarget, taTrainParam)

      if errPrev <= errCurr or myUtil.isNan(errCurr)  then
        print("** early stop **")
        return errPrev, mNetAdapterPrev
      elseif errCurr ~= nil then
        local message = errCurr < errPrev and "<" or "!>"
        myUtil.log(message, false, taTrainParam.isLog)
        errPrev = errCurr
        mNetAdapterPrev = mNetAdapter:clone()
      else
        error("invalid value for errCurr!")
      end

    end

    return errCurr, mNetAdapter
  end

  return trainerPool
end

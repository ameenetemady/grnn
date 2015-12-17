-- a pool of nn trainers

require 'nn'
require 'optim'
local myUtil = require('../ANN/common/util.lua')

do

  local trainerPool = {}



  function trainerPool.forSumOfBitsTheta1_a(taData, mlp)
    local criterion = nn.MSECriterion()
    local trainer = nn.StochasticGradient(mlp, criterion)
    trainer.maxIteration = 100
    trainer.learningRate = 0.005
    trainer.learningRateDecay = 0.995
--    trainer.shuffleIndices = true
    trainer:train(taData)
  end

  function trainerPool.forSumOfBitsTheta2_a(taData, mlp)
    local criterion = nn.MSECriterion()
    local trainer = nn.StochasticGradient(mlp, criterion)
    trainer.maxIteration = 2000
    trainer.learningRate = 2.0
    trainer.learningRateDecay = 0.995
--    trainer.shuffleIndices = true
    trainer.verbose = true
    trainer:train(taData)
  end

function trainerPool.getDefaultTrainParams(nSize)
  return  {   batchSize =  math.floor(nSize),
                            maxIteration = 1000,
                            coefL1 = 0.0,
                            coefL2 = 0.0,
                            strOptimMethod = "CG",

                            -- SGD Params:
                            taSgdParams = {
                              learningRate = 0.1,
                              learningRateDecay = 0.9995,
                              momentum = 0.1,
--                              dampening = 0,
--                              nesterov = true 
                              },

                            -- LBFGS params
                            taLbfgsParams = {
                              maxIter = 1000,
                              lineSearch = optim.lswolfe },
                             
                            -- CG params : Try "Conjugate Gradient"
                            taCgParams = {
                              maxIter = 20}
                            }
end

  function trainerPool.train_MiniBatch(taData, model, criterion, taTrainParams)

    parameters, gradParameters = model:getParameters()

    local overallErr = 0
    local nCols = taData[1][1]:size(1)

    for t = 1,taData:size(), taTrainParams.batchSize do
 
      -- create mini batch
      local currBatchSize = math.min(taTrainParams.batchSize, taData:size() - t + 1)
--      io.write( t .. ", " )
--      io.flush()
      local teBatchX = torch.zeros(currBatchSize, nCols)
      local teBatchY = torch.zeros(currBatchSize)
      local k = 1
      for i = t, math.min(t + taTrainParams.batchSize - 1, taData:size()) do
        teBatchX[k] = taData[i][1]
        teBatchY[k] = taData[i][2]:squeeze()
        k = k + 1
      end

      local fuEval = function(x)

        -- just in case:
        collectgarbage()

        -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- evaluate function for the complete mini batch
        local teBatchPredY = model:forward(teBatchX)
        local f = criterion:forward(teBatchPredY, teBatchY)

        -- estimate df/dW
        local df_do = criterion:backward(teBatchPredY, teBatchY)
        model:backward(teBatchX, df_do)

       -- penalties (L1 and L2):
        if taTrainParams.coefL1 ~= 0 or taTrainParams.coefL2 ~= 0 then
          -- locals:
           local norm,sign= torch.norm,torch.sign
 
          -- Loss:
          f = f + taTrainParams.coefL1 * norm(parameters,1)
          f = f + taTrainParams.coefL2 * norm(parameters,2)^2/2


          -- Gradients:
          gradParameters:add( sign(parameters):mul(taTrainParams.coefL1) + parameters:clone():mul(taTrainParams.coefL2) )
        end
        
        overallErr = overallErr + f
--        io.write(f .. ",")
--        io.flush()

        return f, gradParameters
      end --fuEval

      if taTrainParams.strOptimMethod == 'SGD' then
        optim.sgd(fuEval, parameters, taTrainParams.taSgdParams)

      elseif taTrainParams.strOptimMethod == "LBFGS" then
        optim.lbfgs(fuEval, parameters, taTrainParams.taLbfgsParams)

      elseif taTrainParams.strOptimMethod == "CG" then
        optim.cg(fuEval, parameters, taTrainParams.taCgParams)

      else
        error('unknown strOptimMethod:' .. taTrainParams.strOptimMethod)
      end

      -- io.write(", " ..  overallErr/t)
      --io.flush()

    end

    return overallErr/(taData:size()/taTrainParams.batchSize)
    -- ToDo: calculate accurate overall training loss (instead of above)



  end

  function trainerPool.train_MiniBatch_Outer(taTrainData, model, criterion, taTrainParams)
--    local taX = taTrainData[1]
--    local taY = taTrainData[2]
    local err = 0
    local lastErr = math.random()
    for i=1, taTrainParams.maxIteration do
      err = trainerPool.train_MiniBatch(taTrainData, model, criterion, taTrainParams)

      if lastErr == err then
        print("** early stop **")
        break
      else
--        print("\r\n=======#" .. i .. ":" .. err .. "=======")
        lastErr = err
      end

    end

    print("err: " .. err)

  end


  function trainerPool.full_CG(taData, mlp)
    local criterion = nn.MSECriterion()
    
    local taTrainParams = trainerPool.getDefaultTrainParams(taData:size())
    trainerPool.train_MiniBatch_Outer(taData, mlp, criterion, taTrainParams)
--print(taData)


  end

  return trainerPool
end

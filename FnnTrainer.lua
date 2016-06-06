local FnnTrainer = FnnTrainer or torch.class("FnnTrainer")

function FnnTrainer:__init(taParam, mNetAdapter)
  self.taParam = taParam
  self.mNetAdapter = mNetAdapter
end

function FnnTrainer:trainTogether()
    local dTrainErr
    dTrainErr, self.mNetAdapter = self.taParam.fuTrainer(self.mNetAdapter:clone(), 
                                                         self.taParam.teInput, self.taParam.teTarget, self.taParam.taFuTrainerParams)
    return dTrainErr, self.mNetAdapter
end

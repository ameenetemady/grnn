local aggr_result = require('../../aggr_result.lua')
local lfs = require 'lfs'

local taSet = { 
----[[
  { strFormat= "result/grnn_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 10 },
  { strFormat= "result/grnn_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 10 },
  { strFormat= "result/fnn_nh0_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 10 },
  { strFormat= "result/fnn_nh0_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 10 }

}

for k, v in pairs(taSet) do
  local taSummary = aggr_result.getAggrSummary(v)
  print("*****" .. string.format(v.strFormat, "*") .. ":")
  print(taSummary)
end

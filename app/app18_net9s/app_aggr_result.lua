local aggr_result = require('../../aggr_result.lua')
local lfs = require 'lfs'

local taSet = { 
----[[
  { strFormat= "result/grnn_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/grnn_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh0_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh1_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh2_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh3_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh4_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh0_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh1_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh2_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh3_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 },
  { strFormat= "result/fnn_nh4_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100 }


}

for k, v in pairs(taSet) do
  local taSummary = aggr_result.getAggrSummary(v)
  
  print( 
        string.format("%.4f", taSummary.taTestResult.mean) .. "±" ..
        string.format("%.4f", taSummary.taTestResult.std) .. "|" ..
        string.format(v.strFormat, "*"))
--  print(taSummary)
end

local aggr_result = require('../../aggr_result.lua')
local lfs = require 'lfs'

local taSet = { 
----[[
  { strFormat= "result/grnn_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 20 },
    { strFormat= "result/fnn_nh0_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 20},
		
  { strFormat= "result/fnn_nh1_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 20},

  { strFormat= "result/fnn_nh2_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 20},
  --[[ { strFormat= "result/fnn_nh3_nnpl4_d_%s.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 100},
		--]]

  { strFormat= "result/grnn_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 20 },
  { strFormat= "result/fnn_nh0_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 20 },
  { strFormat= "result/fnn_nh1_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 20 },
  { strFormat= "result/fnn_nh2_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 10 },

		--[[
  { strFormat= "result/fnn_nh3_nnpl4_d_%s_noise.table",
    taFields = { "dTrainErr", "taTestResult"},
    nMaxId = 10 },
    --]]


}

for k, v in pairs(taSet) do
  local taSummary = aggr_result.getAggrSummary(v)
  
  print( 
        string.format("%.4f", taSummary.taTestResult.median) .. "±" ..
        string.format("%.4f", taSummary.taTestResult.std) .. "|" ..
        string.format(v.strFormat, "*"))
  --print(taSummary)
end

----[[
local taBenchMark = { 
          taGroups = {
            { name = "grnn",
              strFilePattern = "result/grnn_d_%s.table"},
            { name = "grnn",
              strFilePattern = "result/grnn_d_%s_noise.table"},
            { name = "fnn0",
              strFilePattern = "result/fnn_nh0_nnpl4_d_%s.table"},
            { name = "fnn0",
              strFilePattern = "result/fnn_nh0_nnpl4_d_%s_noise.table"},
            { name = "fnn1",
              strFilePattern = "result/fnn_nh1_nnpl4_d_%s.table"},
            { name = "fnn1",
              strFilePattern = "result/fnn_nh1_nnpl4_d_%s_noise.table"},
            { name = "fnn2",
              strFilePattern = "result/fnn_nh1_nnpl4_d_%s.table"},
            { name = "fnn2",
              strFilePattern = "result/fnn_nh1_nnpl4_d_%s_noise.table"},
          },
          nMinId = 1,
          nMaxId = 20,
          strMetricProperty = "taTestResult",
      }
--aggr_result.printFullSummary(taBenchMark)
aggr_result.printFullSummaryForFigure(taBenchMark)
--]]

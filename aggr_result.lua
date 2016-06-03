local lfs = require 'lfs'

do
  local aggr_result = {}

  function aggr_result.pri_AppendData(taAggrData, taCurrData)
    for kCurr, vCurr in pairs(taCurrData) do
      for kAggr, vAggr in pairs(taAggrData) do
        table.insert(vAggr, vCurr[kAggr])
      end
    end

  end

  function aggr_result.pri_SetHeader(taAggrData, taFields)
    for k, v in pairs(taFields) do
      taAggrData[v] = {}
    end
  end

  function aggr_result.pri_GetSummary(taAggrData)
    local taSummary = {}
    for k, taV in pairs(taAggrData) do
      local teData = torch.Tensor(taV)
      taSummary[k] = { mean = teData:mean(), std = teData:std() }
    end

    return taSummary
  end

  function aggr_result.getAggrSummary(taInfo)
    local taAggrData = {}
    aggr_result.pri_SetHeader(taAggrData, taInfo.taFields)

    for i=1, taInfo.nMaxId do
      local strFilename = string.format(taInfo.strFormat, i)
      local taCurrData = torch.load(strFilename, "ascii")
      aggr_result.pri_AppendData(taAggrData, taCurrData)
    end

    local taSummary = aggr_result.pri_GetSummary(taAggrData)

    return taSummary
  end

  return aggr_result
end

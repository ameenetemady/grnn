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
      taSummary[k] = { median = torch.median(teData):squeeze(), std = teData:std() }
    end

    return taSummary
  end

  function aggr_result.getAggrSummary(taInfo)
    local taAggrData = {}
    aggr_result.pri_SetHeader(taAggrData, taInfo.taFields)
    local nMinId = taInfo.nMinId or 1

    for i=nMinId, taInfo.nMaxId do
      local strFilename = string.format(taInfo.strFormat, i)
      local taCurrData = torch.load(strFilename, "ascii")
      aggr_result.pri_AppendData(taAggrData, taCurrData)
    end

    local taSummary = aggr_result.pri_GetSummary(taAggrData)

    return taSummary
  end
  
  function aggr_result.printGroupSummary(taGroup, nMinId, nMaxId, strMetricProperty)
    print("********" .. taGroup.name .. "********" )
    for i=nMinId, nMaxId do
      print("d_" .. i)
      local strFilename = string.format(taGroup.strFilePattern, i)
      local taCurrData = torch.load(strFilename, "ascii")
      for k, v in pairs(taCurrData) do
        local dValue = v[strMetricProperty]
        if dValue > 1 then
          print("!!" .. dValue .. "!!")
        else
          print(dValue)
        end

        
      end
      

    end
  end
  
  function aggr_result.printFullSummary(taBenchMark)
    for k, v in pairs(taBenchMark.taGroups) do
        aggr_result.printGroupSummary(v, 
                                      taBenchMark.nMinId, taBenchMark.nMaxId, 
                                      taBenchMark.strMetricProperty)
    end
  
  end

  return aggr_result
end



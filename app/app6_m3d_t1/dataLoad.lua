local mySettings = require("../../mySettings.lua")
local csv = require("csv")

do
  local dataLoad = {}

  function dataLoad.loadAndSaveTo(taParam, strFilenameSave)
    local strFilename = mySettings.strFilename_md3_exprData_p0
    local taLoadParams = {header=false, separator = "\t,_"}
    local f = csv.open(strFilename, taLoadParams)

    local i = 0
    local nMaxLines = 99999999999
    local taRawSelected = {}
    local nGenesRead = 0
    local nCols = 0

    for fields in f:lines() do
      i = i + 1
      if i> nMaxLines or nGenesRead == taParam.nGenes then
        print("done on i=" .. i)
        break
      end

      local strGeneName = fields[1]
      if taParam.taGenes[strGeneName] then
        taRawSelected[strGeneName] = {}
        nCols = #fields - 3
        print("reading " .. #fields)
        for fieldId=4, #fields do
          taRawSelected[strGeneName][fieldId-3] = fields[fieldId]
        end

        nGenesRead = nGenesRead + 1
      end

    end

    -- remove all NaN
    local taColsToRemove = {}
    for strGeneName, taRow in pairs(taRawSelected) do
      for colId=nCols,1, -1 do
        if taRawSelected[strGeneName][colId] == "NaN" then
          taColsToRemove[colId] = true
        else
          taRawSelected[strGeneName][colId] = tonumber(taRawSelected[strGeneName][colId])
        end
      end
    end

    local taColIdsToRemove = {}
    for k, v in pairs(taColsToRemove) do
      table.insert(taColIdsToRemove, k)
    end
    table.sort(taColIdsToRemove)

    print(taColIdsToRemove)
    for idx=#taColIdsToRemove, 1, -1 do
      for strGeneName, taRow in pairs(taRawSelected) do
        table.remove(taRawSelected[strGeneName], taColIdsToRemove[idx])
      end
    end

    --[[
    for strGeneName, taRow in pairs(taRawSelected) do
      for colId, val in pairs (taColsToRemove) do
        taRawSelected[strGeneName][colId] = nil
      end
    end

    --]]


--    print(taColsToRemove)
--    print(taRawSelected)

    torch.save(strFilenameSave, taRawSelected)

  end

  return dataLoad

end



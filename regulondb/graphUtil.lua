local csv = require("csv")
require("../../MyCommon/Stack.lua")

do
  local graphUtil = {}

  function graphUtil.load_tf_gene()
      local taLoadParams = {header=false}
      local strFilename = "processed/network_tf_gene_noHeader.txt" 
      local f = csv.open(strFilename, taLoadParams)

      local taAllTF = {}

      for fields in f:lines() do

        local strTFName = fields[1]:gsub("^%u", string.lower) -- convert tf_name to gene_name
        local taCurrTF = taAllTF[strTFName]
        if taCurrTF == nil then
          taCurrTF = {taConnection = {}}
          taAllTF[strTFName] = taCurrTF
        end
        local strGeneName = fields[2]
        local strEffect = fields[3]
        local strConfidence = fields[5]

        local taConnection = { strGeneName, strEffect, strConfidence }

        table.insert(taAllTF[strTFName].taConnection, taConnection)

      end

    return taAllTF
  end

  function graphUtil.DfsSweep(taGraph, visitedNodes)
    local stackFinish = Stack()
    table.foreach(taGraph, 
      function(k, v)
        if visitedNodes[k] == nil then
          graphUtil.Dfs(k, taGraph, visitedNodes, stackFinish)
        end
      end)

      return stackFinish
  end

  function graphUtil.getTranspose(taGraph)
    local taGraphT = {}

    for ok, ov in pairs(taGraph) do
      for ik, iv in pairs(ov.taConnection) do
        local strDst = iv[1]
        local strSrc = ok
        
        local taCurr = taGraphT[strDst]
        if taCurr == nil then
          taCurr = { taConnection = {}}
          taGraphT[strDst] = taCurr
        end

        table.insert(taCurr.taConnection, {strSrc, iv[2], iv[3]})

      end
    end

    return taGraphT
  end

  function graphUtil.getStrongConnected(taGraph)
    local visitedNodes = {}
    local stackFinish = graphUtil.DfsSweep(taGraph, visitedNodes)

    local taGraphT = graphUtil.getTranspose(taGraph)

    visitedNodes = {}
    taStrongConnected = {}
    while not stackFinish:isEmpty() do
      local strCurr = stackFinish:pop()
      local stackFinsihCurr = Stack()
      graphUtil.Dfs(strCurr, taGraphT, visitedNodes, stackFinsihCurr)

      if not stackFinsihCurr:isEmpty() then
        table.insert(taStrongConnected, stackFinsihCurr._stack)
      end
    end

    return taStrongConnected
  end

  function graphUtil.Dfs(strRoot, taGraph, visitedNodes, stackFinish)
    -- 1) initialize
    local s = Stack()
    s:push(strRoot)

    while not s:isEmpty() do
      -- 2) visit
      local strCurr = s:pop()
      local taNodeInfo = taGraph[strCurr]

      if (taNodeInfo == nil and visitedNodes[strCurr] == nil)  or visitedNodes[strCurr] == 1 then
        visitedNodes[strCurr] = 2
        if stackFinish ~= nil then
          stackFinish:push(strCurr)
        end

      elseif visitedNodes[strCurr] ~= 2 then
        visitedNodes[strCurr] = 1
        s:push(strCurr)

        table.foreach(taNodeInfo.taConnection, function(k, v)
          local strNeighborName = v[1]
          if visitedNodes[strNeighborName] == nil then
--            io.write(strNeighborName .. "|")
            s:push(strNeighborName)
          end
        end)
--        print("")

      end --elseif

    end --while

  end -- function

  function  graphUtil.printGraph_flat(taGraph)
    if taGraph == nil then
      return
    end

     for key,value in pairs(taGraph) do
       io.write(key .. ":")
       for key, value in pairs(value.taConnection) do
         local v2 = value[2] or ""
         local v3 = value[3] or ""
         io.write(value[1] .. "," .. v2 .. "," .. v3 .. "|" )
       end
       print("")
     end
  end
  
-- graphUtil.load_tf_gene()
  return graphUtil

end


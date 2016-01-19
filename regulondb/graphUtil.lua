local csv = require("csv")
local myUtil = require("../../MyCommon/util.lua")
require("../../MyCommon/Stack.lua")

do
  local graphUtil = {}

  function graphUtil.load_tf_gene(strFilename)
      local taLoadParams = {header=false}
      local strFilename = "processed/" .. strFilename -- network_tf_gene_noHeader.txt" 
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

  -- desc: run DFS for all nodes that have outgoing links
  -- return: return stackFinish accumulated during DFS searches
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

  -- return: the transpose of taGraph
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

  -- desc: using kosaraju's algorithm
  -- return: a table which consists of groups of strongly connected nodes
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

    -- ** extra post processing: **
    -- put ScId for each node
    local taScInfo = { taScIds = {}, taScSize = {} }
    for id, taSc in pairs(taStrongConnected) do
      for __, strNode in pairs(taSc) do

        -- update taScIds
        taScInfo.taScIds[strNode] = id


        -- update taScSize
        taScInfo.taScSize[id] = taScInfo.taScSize[id] or 0
        taScInfo.taScSize[id] = taScInfo.taScSize[id] + 1
      end
    end

    return taStrongConnected, taScInfo
  end

  -- desc: Depth First Search(DFS) starting from strRoot. Mark the visitedNodes. Once node is done(black) push it to stackFinish
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

        table.foreach(taNodeInfo.taConnection, 
          function(k, v)
            local strNeighborName = v[1]
            if visitedNodes[strNeighborName] == nil then
              s:push(strNeighborName)
            end
          end)

      end --elseif

    end --while

  end -- function

  -- remove the edges which are part of a loop
  function graphUtil.removeEdgesFromSameSC(strNodeName, taNodeInfo, taScInfo)
    local taScIds = taScInfo.taScIds

    for key, taNeighbor in pairs(taNodeInfo.taConnection) do
      local strNeighborName = taNeighbor[1]
      if taScIds[strNodeName] == taScIds[strNeighborName] then -- sameSC
        table.remove(taNodeInfo.taConnection, key)
      end --if
    end--for
  end

  function graphUtil.removeEdgesFromScToCycle(strNodeName, taNodeInfo, taScInfo)

    local taScSize = taScInfo.taScSize
    local taScIds = taScInfo.taScIds

    for key, taNeighbor in pairs(taNodeInfo.taConnection) do
      local strNeighborName = taNeighbor[1]
      if taScSize[taScIds[strNeighborName]] > 1 then
        table.remove(taNodeInfo.taConnection, key)
      end --if
    end--for

  end

  function graphUtil.addNodeMetaInfo(taGraph)

    local taNIn = {}
    -- Calc 
    for strNodeName,taNodeInfo in pairs(taGraph) do
      for __, taNeighbor in pairs(taNodeInfo.taConnection) do
        local strNeighborName = taNeighbor[1]

        local nIn = taNIn[strNeighborName] or 0
        taNIn[strNeighborName] = nIn + 1
      end
    end

    -- Fill
    for key, value in pairs(taNIn) do
      local taNodeInfo = taGraph[key]
      if taNodeInfo == nil then
        taNodeInfo = { taConnection = {} }
        taGraph[key] = taNodeInfo
      end

      taNodeInfo.nIn = value
    end


  end

  function graphUtil.removeEdgesIf(taGraph, fRemove)
    for strNodeName, taNodeInfo in pairs(taGraph) do
      for key, taConnection in pairs(taNodeInfo.taConnection) do 
        local strNeighborName = taConnection[1]
        local taNeighborInfo = taGraph[strNeighborName]
        if fRemove(strNeighborName, taNeighborInfo) then
          taNodeInfo.taConnection[key] = nil
        end
      end
    end

  end

  function graphUtil.getCascadeSubgraphs(taGraph)
    local taACyclicGraph = graphUtil.getACyclicSubgraphs(taGraph)
    graphUtil.addNodeMetaInfo(taACyclicGraph)

    graphUtil.removeEdgesIf(taACyclicGraph, 
      function(strNodeName, taNodeInfo)
        return taNodeInfo ~= nil and taNodeInfo.nIn ~= nil and taNodeInfo.nIn > 1
      end)

   return taACyclicGraph
  end

  function graphUtil.removeSelfLinks(taGraph)
     for strNodeName, taNodeInfo in pairs(taGraph) do
       for ikey, taConnection in pairs(taNodeInfo.taConnection) do
         local strNeighborName = taConnection[1]
         if strNeighborName == strNodeName then
           taNodeInfo.taConnection[ikey] = nil
         end
       end
     end

  end

  -- return: a new graph, where edges that:
  -- A) participate in a cycle(loop) are removed
  -- B) points a node from a StrongComponent(SC) to another SC which has a loop
  function graphUtil.getACyclicSubgraphs(taGraph)
    local taStrongConnected, taScInfo= graphUtil.getStrongConnected(taGraph)
    local taAsyclicSubgraphs = {}

    local taGraphCopy = myUtil.getDeepCopy(taGraph)
    
    for strNodeName, taNodeInfo in pairs(taGraphCopy) do 
      graphUtil.removeEdgesFromSameSC(strNodeName, taNodeInfo, taScInfo) --A)
      graphUtil.removeEdgesFromScToCycle(strNode, taNodeInfo, taScInfo) -- B)
    end

    return taGraphCopy
  end

  -- desc: print the graph
  function  graphUtil.printGraph_flat(taGraph, isIncludeLeafNode)
    if taGraph == nil then
      return
    end

     for key,value in pairs(taGraph) do
       if  (value.taConnection ~= nil and #value.taConnection ~= 0) or isIncludeLeafNode then
         io.write(key .. ":")
         for ikey, ivalue in pairs(value.taConnection) do
           local v1 = ivalue[1] or ""
           local v2 = ivalue[2] or ""
           local v3 = ivalue[3] or ""
           io.write(v1 .. "," .. v2 .. "," .. v3 .. "|" )
         end
         print("")
       end
     end
  end
  
  return graphUtil

end


local graphUtil = require('./graphUtil.lua')


local graphUtil_test = {}
local taTestGraphA = {
    a = { taConnection = { {"b"}, {"c"}, {"f"} }
        },
    b = { taConnection = { {"d"}, {"c"} }
        },
    d = { taConnection = { {"a"}, {"c"} }
        },
    e = { taConnection = { {"g"}, {"c"} }
        },
    f = { taConnection = { {"c"}, {"a"} }
        },
    g = { taConnection = { {"d"}, {"e"} }
        }
   }

local taTestGraphB = {
    a = { taConnection = { {"b"} }
        },
    b = { taConnection = { {"c"}, {"d"} }
        },
    c = { taConnection = { {"a"}}
        },
    d = { taConnection = { {"f"}, {"e"} }
        },
    e = { taConnection = { {"f"}}
        },
    f = { taConnection = { {"g"}}
        },
    g = { taConnection = { {"h"}}
        },
    h = { taConnection = { {"g"}}
        },
   }

local taTestGraphC = {
    a = { taConnection = { {"b"} }
        },
    b = { taConnection = { {"c"}, {"d"} }
        },
    c = { taConnection = { {"b"}, {"d"} }
        },
    d = { taConnection = { {"e"}}
        },
    e = { taConnection = { {"f"}}
        }
   }

local strFilename = "network_tf_gene_noHeader.txt"

function graphUtil_test.load_tf_gene_test1()
  local taTFAll = graphUtil.load_tf_gene(strFilename)
  graphUtil.printGraph_flat(taTFAll)
end

function graphUtil_test.printGraph_flat_test1()
  graphUtil.printGraph_flat(taTestGraphA)
end

function graphUtil_test.Dfs_test1()
  local visitedNodes = {}
  local root = "a"
  graphUtil.Dfs(root, taTestGraphA, visitedNodes)
  print(visitedNodes)
end

function graphUtil_test.Dfs_test2()
  local visitedNodes = {}
  local root = "a"
  local s = Stack()
  graphUtil.Dfs(root, taTestGraphA, visitedNodes, s)
  print(s)
end


function graphUtil_test.DfsSweep_test1()
  local visitedNodes = {}
  graphUtil.DfsSweep(taTestGraphA, visitedNodes)
  print(visitedNodes)
end

function graphUtil_test.getStrongConnected_test1()
  local taStrongConnected = graphUtil.getStrongConnected(taTestGraphA)
  print(taStrongConnected)

end

function graphUtil_test.getStrongConnected_test2()
  local taTFAll = graphUtil.load_tf_gene(strFilename)
  local taStrongConnected = graphUtil.getStrongConnected(taTFAll)
  print(taStrongConnected)

end


function graphUtil_test.getTranspose_test1()
  local taGraphT = graphUtil.getTranspose(taTestGraphA)
  graphUtil.printGraph_flat(taGraphT)
end


function graphUtil_test.getACyclicSubgraphs_test1A()
  local taTrimmedGraph = graphUtil.getACyclicSubgraphs(taTestGraphA)
  graphUtil.printGraph_flat(taTrimmedGraph)
 
end

function graphUtil_test.getACyclicSubgraphs_test1B()
  local taTrimmedGraph = graphUtil.getACyclicSubgraphs(taTestGraphB)
  graphUtil.printGraph_flat(taTrimmedGraph)
 
end

function graphUtil_test.getACyclicSubgraphs_test1C()
  local taTrimmedGraph = graphUtil.getACyclicSubgraphs(taTestGraphC)
  graphUtil.printGraph_flat(taTrimmedGraph)
 
end


function graphUtil_test.getACyclicSubgraphs_test2()
  local taTFAll = graphUtil.load_tf_gene(strFilename)
  local taTrimmedGraph = graphUtil.getACyclicSubgraphs(taTFAll)
  graphUtil.printGraph_flat(taTrimmedGraph)
 
end

function graphUtil_test.addNodeMetaInfo_test1()
  graphUtil.addNodeMetaInfo(taTestGraphA)
  print(taTestGraphA)

end

function graphUtil_test.getCascadeSubgraphs_test1B()
  local taTrimmedGraph = graphUtil.getCascadeSubgraphs(taTestGraphB)
  graphUtil.printGraph_flat(taTrimmedGraph, false)
end

function graphUtil_test.getCascadeSubgraphs_test1C()
  local taTrimmedGraph = graphUtil.getCascadeSubgraphs(taTestGraphC)
  graphUtil.printGraph_flat(taTrimmedGraph, false)
end


function graphUtil_test.getCascadeSubgraphs_test2()
  strFilename = "network_tf_All_noHeader.txt" 
  local taTFAll = graphUtil.load_tf_gene(strFilename)
  local taTrimmedGraph = graphUtil.getCascadeSubgraphs(taTFAll)
  graphUtil.printGraph_flat(taTrimmedGraph, false)
end

function graphUtil_test.getCascadeSubgraphs_test3()
  strFilename = "network_tf_All_noHeader.txt" 
  local taTFAll = graphUtil.load_tf_gene(strFilename)
  graphUtil.removeSelfLinks(taTFAll)
  local taTrimmedGraph = graphUtil.getCascadeSubgraphs(taTFAll)

  local taLongCascade = {}
  for strNodeName, taNodeInfo in pairs(taTrimmedGraph) do
    for key, taConnection in pairs(taNodeInfo.taConnection) do
      local strNeighborName = taConnection[1]
      local taNeighborNodeInfo = taTrimmedGraph[strNeighborName]
      if strNodeName ~= strNeighborName and taNeighborNodeInfo ~= nil and taNeighborNodeInfo.taConnection ~= nil and #taNeighborNodeInfo.taConnection ~= 0 then
        taLongCascade[strNodeName] = strNeighborName
      end
    end
  end

  print(taLongCascade)
  graphUtil.printGraph_flat(taTrimmedGraph)

end


function graphUtil_test.all()
--  graphUtil_test.load_tf_gene_test1()
--  graphUtil_test.printGraph_flat_test1()
--  graphUtil_test.Dfs_test1()
--  graphUtil_test.Dfs_test2()
--  graphUtil_test.DfsSweep_test1()
--  graphUtil_test.getTranspose_test1()
--  graphUtil_test.getStrongConnected_test1()
--  graphUtil_test.getStrongConnected_test2()
--  graphUtil_test.getACyclicSubgraphs_test1A()
--  graphUtil_test.getACyclicSubgraphs_test1B()
--  graphUtil_test.getACyclicSubgraphs_test1C()
--  graphUtil_test.getACyclicSubgraphs_test2()
--  graphUtil_test.addNodeMetaInfo_test1()
--  graphUtil_test.getCascadeSubgraphs_test1B()
--  graphUtil_test.getCascadeSubgraphs_test1C()
--  graphUtil_test.getCascadeSubgraphs_test2()
  graphUtil_test.getCascadeSubgraphs_test3()
end

graphUtil_test.all()

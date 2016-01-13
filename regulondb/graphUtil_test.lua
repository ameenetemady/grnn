local graphUtil = require('./graphUtil.lua')


local graphUtil_test = {}
local taTestGraph = {
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

function graphUtil_test.load_tf_gene_test1()
  local taTFAll = graphUtil.load_tf_gene()
  graphUtil.printGraph_flat(taTFAll)
end

function graphUtil_test.printGraph_flat_test1()
  graphUtil.printGraph_flat(taTestGraph)
end

function graphUtil_test.Dfs_test1()
  local visitedNodes = {}
  local root = "a"
  graphUtil.Dfs(root, taTestGraph, visitedNodes)
  print(visitedNodes)
end

function graphUtil_test.Dfs_test2()
  local visitedNodes = {}
  local root = "a"
  local s = Stack()
  graphUtil.Dfs(root, taTestGraph, visitedNodes, s)
  print(s)
end


function graphUtil_test.DfsSweep_test1()
  local visitedNodes = {}
  graphUtil.DfsSweep(taTestGraph, visitedNodes)
  print(visitedNodes)
end

function graphUtil_test.getStrongConnected_test1()
  local taStrongConnected = graphUtil.getStrongConnected(taTestGraph)
  print(taStrongConnected)

end

function graphUtil_test.getStrongConnected_test2()
  local taTFAll = graphUtil.load_tf_gene()
  local taStrongConnected = graphUtil.getStrongConnected(taTFAll)
  print(taStrongConnected)

end


function graphUtil_test.getTranspose_test1()
  local taGraphT = graphUtil.getTranspose(taTestGraph)
  graphUtil.printGraph_flat(taGraphT)
end

function graphUtil_test.all()
--  graphUtil_test.load_tf_gene_test1()
--  graphUtil_test.printGraph_flat_test1()
--  graphUtil_test.Dfs_test1()
--  graphUtil_test.Dfs_test2()
--  graphUtil_test.DfsSweep_test1()
--  graphUtil_test.getTranspose_test1()
--  graphUtil_test.getStrongConnected_test1()
  graphUtil_test.getStrongConnected_test2()
end

graphUtil_test.all()

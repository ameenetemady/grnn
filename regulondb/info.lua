do
  local info = {}
  info.taSigmaInfo = { Sigma19 = "fecI",
                    Sigma24 = "rpoE",
                    Sigma28 = "fliA",
                    Sigma32 = "rpoH",
                    Sigma38 = "rpoS",
                    Sigma54 = "rpoN",
                    Sigma70 = "rpoD" }

  info.taMyGeneGroups = { { "cpxR", "csgD", "yhbS"}, -- (-, -) (Sigma38, Sigma38, ?)
                          { "cpxR", "csgD", "yhbT"}, -- (-, -) (Sigma38, Sigma38, ?)
                          { "cpxR", "csgD", "yccT"}, -- (-, +) (Sigma38, Sigma38, ?)
                          { "cpxR", "csgD", "yccJ"}, -- (-, +) (Sigma38, Sigma38, Sigma38)
                          { "rpoS", "cpxR"}, -- 5
                          { "rpoS", "csgD"}, -- 6
                          { "rpoS", "yccJ"}, -- 7
                          { "hns", "rcsB"}, -- 8
                       }


  return info
end

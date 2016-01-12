local csv = require("csv")

local graph_gen = {}

function graph_gen.load_tf_gene()
    local taLoadParams = {header=false}
    local strFilename = "processed/network_tf_gene_noHeader.txt" 
    local f = csv.open(strFilename, taLoadParams)

    local taAllTF = {}

    for fields in f:lines() do

      local strTFName = fields[1]
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
--      print(fields[3])
      --[[
      for key,value in pairs(fields) do
        io.write(key .. ":" .. value .. "|")
      end
      --]]

    end

    for key,value in pairs(taAllTF) do
      io.write(key .. ":")
      for key, value in pairs(value.taConnection) do
        io.write(value[1] .. "," .. value[2] .. "," .. value[3] .. "|" )
      end
      print("")
    end

end

graph_gen.load_tf_gene()

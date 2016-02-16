require('../../MyCommon/PermutationGenerator.lua')
local myUtil = require('../../MyCommon/util.lua')

local gnw_multifactorial = {}

function gnw_multifactorial.cascade5()
  local permutGen = PermutationGenerator({0, 0, 0, 0, 0}, {100, 1, 1, 1, 1})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end


  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:narrow(2, 1, 1):mul(0.01)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll)

  print(strRes)
end

function gnw_multifactorial.SyngTwo()
  local permutGen = PermutationGenerator({0, 0, 0}, {10, 10, 0})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end

  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:narrow(2, 1, 2):mul(-0.1)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll, "\t")

  print(strRes)

end

function gnw_multifactorial.all()
--  gnw_multifactorial.cascade5()
  gnw_multifactorial.SyngTwo()
end

gnw_multifactorial.all()

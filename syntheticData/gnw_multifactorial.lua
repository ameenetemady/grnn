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

function gnw_multifactorial.feedforward1()
  local permutGen = PermutationGenerator({0, 0, 0}, {0, 0, 100})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end

  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:narrow(2, 3, 1):mul(-0.01)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll, "\t")

  print(strRes)
end

function gnw_multifactorial.cascadeA()
  local permutGen = PermutationGenerator({0, 0, 0}, {0, 0, 100})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end

  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:narrow(2, 3, 1):mul(-0.01)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll, "\t")

  print(strRes)
end

function gnw_multifactorial.dimA()
  local permutGen = PermutationGenerator({0, 0, 0, 0, 0}, {10, 0, 0, 10, 0})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end

  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:mul(-0.1)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll, "\t")

  print(strRes)
end

function gnw_multifactorial.net9s()
  local permutGen = PermutationGenerator({0, 0, 0, 0, 0, 0, -10, 0, -10}, {0, 0, 0, 0, 0, 0, 10, 0, 10})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end

  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:mul(-0.1)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll, "\t")

  print(strRes)
end

function gnw_multifactorial.net10s()
  local permutGen = PermutationGenerator({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 10, 0, 0, 10})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end

  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:mul(-0.1)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll, "\t")

  print(strRes)
end

function gnw_multifactorial.net9sb()
  local permutGen = PermutationGenerator({0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 20, 0, 0, 20, 0})
  local tePerm = permutGen:getNext()
  local taAll = {}

  while tePerm ~= nil do
   table.insert(taAll, tePerm:clone())
   tePerm = permutGen:getNext()
  end

  local teAll = myUtil.getTensorFromTableOfTensors(taAll)
  teAll:mul(-0.05)
  local strRes = myUtil.getCsvStringFrom2dTensor(teAll, "\t")

  print(strRes)

end


function gnw_multifactorial.all()
--  gnw_multifactorial.cascade5()
--  gnw_multifactorial.SyngTwo()
--  gnw_multifactorial.feedforward1()
--  gnw_multifactorial.cascadeA()
--  gnw_multifactorial.dimA()
--  gnw_multifactorial.net9s()
--  gnw_multifactorial.net10s()
  gnw_multifactorial.net9sb()
end

gnw_multifactorial.all()

require 'nn'
require('./CMulNoParam.lua')

local MyMul10, parent = torch.class('nn.MyMul10', 'nn.Module')

function MyMul10:__init()
  parent.__init(self)

end


function MyMul10:updateOutput(input)
  self.output = torch.mul(input, 0.5)
  return self.output
end



function ConditionalFunUnit_branchConst()
  local branchA = nn.Sequential()
  branchA:add(nn.Identity())
  branchA:add(nn.Identity())
--  branchA:add(nn.Padding(1, 1)) -- added just to increase dimention

  local branchB = nn.Sequential()
  branchB:add(nn.MulConstant(-1))
  branchB:add(nn.AddConstant(1))
--  branchB:add(nn.Padding(1, 1)) -- added just to increase dimention

  local branchConcatAB = nn.Parallel(2, 1)

  branchConcatAB:add(branchA)
  branchConcatAB:add(branchB)

  local seqMul = nn.Sequential()
  seqMul:add(branchConcatAB)
  seqMul:add(nn.View(2, -1, 1))
  seqMul:add(nn.SplitTable(1))
  seqMul:add(nn.CMulNoParam())

  return seqMul
end


function ConditionalFunUnit_branchTrainable(fuLearnableModuleFactory)
  local branchA = nn.Sequential()
  branchA:add(fuLearnableModuleFactory())
--  branchA:add(nn.Padding(1, 1)) -- added just to increase dimention

  local branchB = nn.Sequential()
  branchB:add(nn.Identity())
--  branchB:add(nn.Padding(1, 1)) -- added just to increase dimention

  local branchConcatAB = nn.Parallel(2, 1)

  branchConcatAB:add(branchA)
  branchConcatAB:add(branchB)

  local seqMul = nn.Sequential()
  seqMul:add(branchConcatAB)
  seqMul:add(nn.View(2, -1, 1))
  seqMul:add(nn.SplitTable(1))
  seqMul:add(nn.CMulNoParam())
  seqMul:add(nn.Identity())

  return seqMul


end

function ConditionalFunUnit(fuLearnableModuleFactory)
  local branchConst = ConditionalFunUnit_branchConst()
  local branchTrainable = ConditionalFunUnit_branchTrainable(fuLearnableModuleFactory)


  local concatMain = nn.Concat(2)
  concatMain:add(branchConst)
  concatMain:add(branchTrainable)

  local main = nn.Sequential()
  main:add(concatMain)
  main:add(nn.Sum(2))


 return main 

end

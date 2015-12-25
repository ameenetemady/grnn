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


function ConditionalFunUnit_branchTrainable(fuLearnableModuleFactory, param)
  local branchA = nn.Sequential()
  branchA:add(fuLearnableModuleFactory(param))
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

function ConditionalFunUnit(fuLearnableModuleFactory, param)
  local branchConst = ConditionalFunUnit_branchConst()
  local branchTrainable = ConditionalFunUnit_branchTrainable(fuLearnableModuleFactory, param)


  local concatMain = nn.Concat(2)
  concatMain:add(branchConst)
  concatMain:add(branchTrainable)

  local main = nn.Sequential()
  main:add(concatMain)
  main:add(nn.Sum(2))


 return main 

end


function OneLayer_ConditionalFunUnit(fuLearnableModuleFactory, nGenes, geneID)
  local mainP = nn.Concat(2)

  -- before Ca
  if geneID ~= 1 then
    local beforeCa = nn.Sequential()
    beforeCa:add( nn.Narrow(2, 1, geneID - 1))
    beforeCa:add( nn.Identity())
    
    mainP:add(beforeCa)
  end

  -- for Ca
  local caItself = nn.Sequential()
  caItself:add(nn.Narrow(2, geneID, 2))
  local caUnit = ConditionalFunUnit(fuLearnableModuleFactory, geneID)
  caItself:add(caUnit)

  if geneID == nGenes then
    caItself:add(nn.Replicate(1, 2)) -- replicate (ensure two dimentional), "nn.View" returns "non-contigues" error, using nn.Replicate to workaround it
--    caItself:add(nn.View(-1, 1 )) -- ensure two dimentional
  else
    caItself:add(nn.Replicate(2, 2)) -- replicate
  end
  mainP:add(caItself)


  -- after Ca
  if geneID ~= nGenes then
    local afterCa = nn.Sequential()
    afterCa:add(nn.Narrow(2, geneID + 2, nGenes - geneID))
    afterCa:add(nn.Identity())
    mainP:add(afterCa)
  end

  return mainP

end

function MultiLayer_ConditionalFunUnit(fuLearnableModuleFactory, nGenes)
  local mainSeq = nn.Sequential()

  for i=1, nGenes do
    local currLayer = OneLayer_ConditionalFunUnit(fuLearnableModuleFactory, nGenes, i)
    mainSeq:add(currLayer)

  end

  return mainSeq
end

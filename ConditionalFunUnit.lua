require 'nn'

local MyMul10, parent = torch.class('nn.MyMul10', 'nn.Module')


function MyMul10:__init()
  parent.__init(self)

end


function MyMul10:updateOutput(input)
  return 10*input
end


function ConditionalFunUnit_branchConst()
  local branchPassThroughA = nn.Sequential()
  branchPassThroughA:add(nn.Identity())
  branchPassThroughA:add(nn.Identity())
  branchPassThroughA:add(nn.Padding(1, 1)) -- added just to increase dimention

  local branchPassThroughB = nn.Sequential()
  branchPassThroughB:add(nn.MulConstant(-1))
  branchPassThroughB:add(nn.AddConstant(1))
  branchPassThroughB:add(nn.Padding(1, 1)) -- added just to increase dimention

  local branchPassThroughConcatAB = nn.Parallel(2, 1)

  branchPassThroughConcatAB:add(branchPassThroughA)
  branchPassThroughConcatAB:add(branchPassThroughB)

  local seqMul = nn.Sequential()
  seqMul:add(branchPassThroughConcatAB)
  seqMul:add(nn.View(2, 1))
  seqMul:add(nn.SplitTable(1))
  seqMul:add(nn.MM(true, false))

  return seqMul
end


function ConditionalFunUnit_branchTrainable()

end

function ConditionalFunUnit(fuLearnableModuleFactory)
  return ConditionalFunUnit_branchConst()

end

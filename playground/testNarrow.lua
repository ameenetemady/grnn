require 'nn'

local nSamples = 5
local nGenes = 7

teInput = torch.Tensor(nSamples, 2, 1):fill(1)
teKO = torch.linspace(1, nGenes, nGenes)
teKOTemplate= torch.Tensor(nSamples, 1, nGenes):fill(1)
for i=1, nSamples do
  teKOTemplate[i][1] = teKO*(10^(i-1)) 
end




local teKOTemplateExpanded = torch.expand(teKOTemplate, nSamples, 2, nGenes)


teConcat = torch.cat(teInput, teKOTemplateExpanded, 3)

mlpMain = nn.Sequential()
mlpMain:add(nn.Narrow(2, 1, 1))

teOutput = mlpMain:forward(teConcat)

print(teOutput)


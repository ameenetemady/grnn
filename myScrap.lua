require 'nn'

c = nn.Parallel(2, 1)

c:add(nn.AddConstant(10))
c:add(nn.MulConstant(10))


teX = torch.Tensor({{11, 22}})
output = c:forward(teX)

print(output)



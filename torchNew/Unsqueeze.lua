local Unsqueeze, parent = torch.class('nn.Unsqueeze', 'nn.Module')



function nnUtils_addSingletonDimension(...)
  local view, t, dim
  if select('#',...) < 3 then
    t, dim = select(1,...)
  else
    view, t, dim = select(1,...)
    assert(torch.isTensor(view),
    "output tensor expected, got " .. type(view))
  end

  assert(torch.isTensor(t), "input tensor expected")
  dim = dim or 1
  assert(dim > 0 and dim <= (t:dim() + 1), "invalid dimension: " .. dim
  .. '. Tensor is of ' .. t:dim() .. ' dimensions.')

  view = view or t.new()
  local size = torch.LongStorage(t:dim() + 1)
  local stride = torch.LongStorage(t:dim() + 1)

  for d = 1, dim - 1 do
    size[d] = t:size(d)
    stride[d] = t:stride(d)
  end
  size[dim] = 1
  stride[dim] = 1
  for d = dim + 1, t:dim() + 1 do
    size[d] = t:size(d - 1)
    stride[d] = t:stride(d - 1)
  end

  view:set(t:storage(), t:storageOffset(), size, stride)
  return view
end





local function _assertTensor(t)
  assert(torch.isTensor(t), "This module only works on tensor")
end

function Unsqueeze:__init(pos, numInputDims)
  parent.__init(self)
  self.pos = pos or error('the position to insert singleton dim not specified')
  self:setNumInputDims(numInputDims)
end

function Unsqueeze:setNumInputDims(numInputDims)
  self.numInputDims = numInputDims
  return self
end

function Unsqueeze:updateOutput(input)
  _assertTensor(input)
  local actualPos = self:_getActualPosition(input)
  nnUtils_addSingletonDimension(self.output, input, actualPos)
  return self.output
end

function Unsqueeze:updateGradInput(input, gradOutput)
  _assertTensor(input)
  _assertTensor(gradOutput)
  assert(input:nElement() == gradOutput:nElement())

  self.gradInput:view(gradOutput, input:size())
  return self.gradInput
end

function Unsqueeze:__tostring__()
  return torch.type(self)..'(dim ' .. self.pos .. ')'
end

function Unsqueeze:_getActualPosition(input)
  -- get valid dimesion offset for batchMode (if any)
  local inputDim = input:dim() -- data batch dim
  self.numInputDims = self.numInputDims or inputDim -- feature map dim
  local offsetDim = inputDim - self.numInputDims
  assert(offsetDim >= 0, "input feature map dim (numInputDims) must be <= input:dim()")

  -- the actual position; clearer error message for batchMode (if any)
  local actualPos = self.pos + offsetDim
  assert(actualPos >= 1 and actualPos <= (inputDim + 1),
    ("Invalid position: %d. input:dim() is %d, input feature map dim (numInputDims) is %d.")
    :format(self.pos, inputDim, self.numInputDims)
  )
  return actualPos
end

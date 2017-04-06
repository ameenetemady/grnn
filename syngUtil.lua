local syngUtil = {}

do
	-- Description: returns two new tensors corresponding to teInputSlice, teTargetSclice only
	-- 						  keeping the masked ones per teKOSclice
  function syngUtil.getPresent(teInputSlice, teTargetSclice, teKOSlice)
    local tePresent = torch.ByteTensor(teKOSlice:size()):copy(teKOSlice)

    local teTargetPresent = syngUtil.getMasked(teTargetSclice, tePresent)
    local teInputPresent = syngUtil.getMasked(teInputSlice, tePresent)

    return teInputPresent, teTargetPresent
  end

	-- Description: retuns new tensor keeping only the masked elements on the rows specified by teMask
  function syngUtil.getMasked(teX, teMask)
    local teRes = torch.Tensor(teMask:sum(), teX:size(2))

    for colId=1, teX:size(2) do
      teRes:narrow(2, colId, 1):copy( teX:narrow(2, colId, 1):maskedSelect(teMask) )
    end

    return teRes
  end

  return syngUtil
end

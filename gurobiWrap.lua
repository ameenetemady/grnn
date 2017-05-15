local gurobi = require 'gurobi'
do
	local gurobiWrap = {}

	-- Description, solves Aw = y constrained with  nLStart, nLLength to define the weights (w) to be nonnegative
	function gurobiWrap.gelsNonNegative(teA, teY, nLStart, nLLength)
		local env = gurobi.loadenv("")

		local nD = teA:size(2)
		local nM = teA:size(1)

		local teC = torch.ones(nD + nM * 2)
		teC:narrow(1, 1, nD):fill(0)

		local teG = torch.cat({teA, torch.eye(nM), -torch.eye(nM)}, 2)

		local dVeryNegative=-1e99
		local teLB = torch.Tensor(teC:size()):fill(dVeryNegative)

		if nLStart and nLLength then -- if constraints provided
			teLB:narrow(1, nLStart, nLLength):fill(0)
		end

		teLB:narrow(1, nD + 1, nM * 2):fill(0)

		local model = gurobi.newmodel(env, "", teC, teLB)
		gurobi.addconstrs(model, teG, 'EQ', teY)

		-- solve
		local status, teW = gurobi.solve(model)

		local teWCopy = teW:narrow(1, 1, nD):clone()
		gurobi.free(env, model)

		return status, teWCopy
	end

	return gurobiWrap

end

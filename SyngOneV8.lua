local archFactory = require('./grnnArchFactory.lua')
local SyngOneV8 = {}

do

	local nInputs, nHidden = 1, 2

	function SyngOneV8.new(taMParameters)
		local mNew = archFactory.piecewiseLinear1(nInputs, nHidden, taMParameters)

		return mNew
	end

	function SyngOneV8.getInitWeights()
		return nil
	end

	return SyngOneV8
end

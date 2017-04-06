local archFactory = require('./grnnArchFactory.lua')
local SyngTwoV8 = {}

do

	local nInputs, nHidden = 2, 3

	function SyngTwoV8.new(taMParameters)
		local mNew = archFactory.piecewiseLinear1(nInputs, nHidden, taMParameters)

		return mNew
	end

	function SyngTwoV8.getInitWeights()
		return nil
	end

	return SyngTwoV8
end

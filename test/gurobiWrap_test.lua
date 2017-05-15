gurobiW = require('../gurobiWrap.lua')

local tester = torch.Tester()
local gurobiWTest = torch.TestSuite()
local eps = 1e-5

function genData1(teWInit)
		local nD = teWInit:size(1)
		local nRow = 10
		teA = torch.rand(nRow, nD)
	--	print(teA)
		local teY = torch.mv(teA, teWInit) 
		local teError = torch.rand(nRow)  * 0.1
		return teA, teY + teError
end

function gurobiWTest.t1()
	local teWInit = torch.Tensor{-1, 1, 10}
	local teA, teY = genData1(teWInit)
	local nLStart = 1
	local nLLength = 2

	local status, teW = gurobiW.gelsNonNegative(teA, teY, nLStart, nLLength)

	if status == 3 then
		print("status:" .. status .. " !!!! error !!!! ")
	elseif status == 2 then
		print("status:" .. status )
	end

	print(status)
	print(teW)

end

tester:add(gurobiWTest)
tester:run()

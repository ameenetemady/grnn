local dataUtil = require("./dataUtil.lua")
local mySettings = require("../mySettings.lua")
local dataUtil_test = {}

function dataUtil_test.getBNumber_test1()
 local strGeneName = "pptA" 
 local taNameToBNumber = dataUtil.loadBNumberDic(mySettings.strFilenameGeneDic)
 print(strGeneName .. ":" .. taNameToBNumber[strGeneName])
end

function dataUtil_test.all()
  dataUtil_test.getBNumber_test1()

end


dataUtil_test.all()

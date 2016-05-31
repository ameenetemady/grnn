require('./MNet9s.lua')

local MNet9s_test = {}

function MNet9s_test.trainTogether_test1()
  local taParam = {}
  local mNetA = MNet9s.new(taParam)
  local taTestResA = mNetA:test(taData)
  print("mNetA test result:")
  print(taTestResA)

  local mNetB = mNetA:clone()
  mNetB:trainTogether(taData)
  local taTestResB = mNetB:test(taData)

  print("mNetB test result:")
  print(taTestResB)
end

function MNet9s_test.trainEach_test1()
  local taParam = {}
  local mNet = MNet9s.new(taParam)

  mNet:trainEach()

  local taTestRes = mNet:test(taData)
  print("mNet test result:")
  print(taTestRes)
end

function MNet9s_test.trainOne_test1()
  local taParam = {}
  local mNet = MNet9s.new(taParam)

  local taTestRes = mNet:test(taData)
  print("mNet test result Before:")
  print(taTestRes)

  mNet:trainOne("G5")

  taTestRes = mNet:test(taData)
  print("mNet test result After:")
  print(taTestRes)
end

function MNet9s_test.LoadNew_test1()
  local taParam = {}
  local mNet = MNet9s.new(taParam)

  local taTestRes = mNet:test(taData)
  print("mNet test result:")
  print(taTestRes)

  local mNetNew = MNet9s.LoadNew(taParam, mNet.taWeights)

  taTestRes = mNetNew:test(taData)
  print("mNetNew test result:")
  print(taTestRes)
end

function MNet9s_test.all()
  MNet9s_test.trainTogether_test1()
  MNet9s_test.trainOne_test1()
  MNet9s_test.trainEach_test1()
  MNet9s_test.LoadNew_test1()
end

MNet9s_test.all()

local ClonableUnit = ClonableUnit or torch.class("ClonableUnit")

function ClonableUnit:__init(fuUnitFactory, teWeight)
  self.fuUnitFactory = fuUnitFactory
  self.teWeight = teWeight
  self.mUnit = self.fuUnitFactory(teWeight)
end

function ClonableUnit:clone()
  return ClonableUnit.new(self.fuUnitFactory, self.teWeight:clone())
end

function ClonableUnit:getRaw()
  return self.mUnit
end

local settings = {}

do
  settings.cascade4 = {}
  settings.cascade4.inputFilename = "/home/ameen/gnw/app3_cascade5/processed/input.tsv"
  settings.cascade4.targetFilename= "/home/ameen/gnw/app3_cascade5/processed/target.tsv"
  settings.cascade4.train_inputFilename= "/home/ameen/gnw/app3_cascade5/processed/train_input.tsv"
  settings.cascade4.train_targetFilename= "/home/ameen/gnw/app3_cascade5/processed/train_target.tsv"
  settings.cascade4.test_inputFilename= "/home/ameen/gnw/app3_cascade5/processed/test_input.tsv"
  settings.cascade4.test_targetFilename= "/home/ameen/gnw/app3_cascade5/processed/test_target.tsv"
  settings.cascade4.nInputCols = 5
  settings.cascade4.nTargetCols = 4

  settings.cascade4_big = {}
  settings.cascade4_big.inputFilename = "/home/ameen/gnw/app3_cascade5_big/processed/input.tsv"
  settings.cascade4_big.targetFilename= "/home/ameen/gnw/app3_cascade5_big/processed/target.tsv"
  settings.cascade4_big.nInputCols = 5
  settings.cascade4_big.nTargetCols = 4

  settings.SyngTwo7= {}
  settings.SyngTwo7.inputFilename = "/home/ameen/gnw/app7_SyngTwo/processed/input.tsv"
  settings.SyngTwo7.targetFilename= "/home/ameen/gnw/app7_SyngTwo/processed/target.tsv"
  settings.SyngTwo7.nInputCols = 2
  settings.SyngTwo7.nTargetCols = 1


  return settings
end

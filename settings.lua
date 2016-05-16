local settings = {}

do
  strBaseGnwPath = "/home/ameen/gnw"

  settings.cascade4 = {}
  settings.cascade4.inputFilename = strBaseGnwPath .. "/app3_cascade5/processed/input.tsv"
  settings.cascade4.targetFilename= strBaseGnwPath .. "/app3_cascade5/processed/target.tsv"
  settings.cascade4.train_inputFilename= strBaseGnwPath .. "/app3_cascade5/processed/train_input.tsv"
  settings.cascade4.train_targetFilename= strBaseGnwPath .. "/app3_cascade5/processed/train_target.tsv"
  settings.cascade4.test_inputFilename= strBaseGnwPath .. "/app3_cascade5/processed/test_input.tsv"
  settings.cascade4.test_targetFilename= strBaseGnwPath .. "/app3_cascade5/processed/test_target.tsv"
  settings.cascade4.nInputCols = 5
  settings.cascade4.nTargetCols = 4

  settings.cascade4_big = {}
  settings.cascade4_big.inputFilename = strBaseGnwPath .. "/app3_cascade5_big/processed/input.tsv"
  settings.cascade4_big.targetFilename= strBaseGnwPath .. "/app3_cascade5_big/processed/target.tsv"
  settings.cascade4_big.nInputCols = 5
  settings.cascade4_big.nTargetCols = 4

  settings.SyngTwo7= {}
  settings.SyngTwo7.inputFilename = strBaseGnwPath .. "/app7_SyngTwo/processed/input.tsv"
  settings.SyngTwo7.targetFilename= strBaseGnwPath .. "/app7_SyngTwo/processed/target.tsv"
  settings.SyngTwo7.nInputCols = 2
  settings.SyngTwo7.nTargetCols = 1

  settings.feedforward1= {}
  settings.feedforward1.inputFilename = strBaseGnwPath .. "/app8_FeedForward1/processed/input.tsv"
  settings.feedforward1.targetFilename= strBaseGnwPath .. "/app8_FeedForward1/processed/target.tsv"
  settings.feedforward1.nInputCols = 1
  settings.feedforward1.nTargetCols = 2

  
  settings.cascadeA= {}
  settings.cascadeA.inputFilename = strBaseGnwPath .. "/app9_cascadeA/processed/input.tsv"
  settings.cascadeA.targetFilename= strBaseGnwPath .. "/app9_cascadeA/processed/target.tsv"
  settings.cascadeA.nInputCols = 1
  settings.cascadeA.nTargetCols = 2

  settings.cascadeB= {}
  settings.cascadeB.inputFilename = strBaseGnwPath .. "/app10_cascadeB/processed/input.tsv"
  settings.cascadeB.targetFilename= strBaseGnwPath .. "/app10_cascadeB/processed/target.tsv"
  settings.cascadeB.nInputCols = 1
  settings.cascadeB.nTargetCols = 2

  settings.dimA= {}
  settings.dimA.inputFilename = strBaseGnwPath .. "/app11_dimA/processed/input.tsv"
  settings.dimA.targetFilename= strBaseGnwPath .. "/app11_dimA/processed/target.tsv"
  settings.dimA.nInputCols = 2
  settings.dimA.nTargetCols = 3

  settings.net9s = {}
  settings.net9s.inputFilename = strBaseGnwPath .. "/app12_net9s/processed/input.tsv"
  settings.net9s.targetFilename= strBaseGnwPath .. "/app12_net9s/processed/target.tsv"
  settings.net9s.nInputCols = 2
  settings.net9s.nTargetCols = 7


  settings.net9sb = {}
  settings.net9sb.inputFilename = strBaseGnwPath .. "/app14_net9sb/processed/input.tsv"
  settings.net9sb.targetFilename= strBaseGnwPath .. "/app14_net9sb/processed/target.tsv"
  settings.net9sb.nInputCols = 2
  settings.net9sb.nTargetCols = 7

  settings.feedforward1_many = {}
  settings.feedforward1_many.baseDir = strBaseGnwPath .. "/app15_feedforward1_many"
  settings.feedforward1_many.nRuns = 100
  settings.feedforward1_many.nInputCols = 1
  settings.feedforward1_many.nTargetCols = 2

  settings.dimA_many = {}
  settings.dimA_many.baseDir = strBaseGnwPath .. "/app16_dimA"
  settings.dimA_many.nRuns = 100
  settings.dimA_many.nInputCols = 2
  settings.dimA_many.nTargetCols = 3

  settings.feedforward1SKO_many= {}
  settings.feedforward1SKO_many.baseDir = strBaseGnwPath .. "/app17_feedforward1SKO_many"
  settings.feedforward1SKO_many.nRuns= 2
  settings.feedforward1SKO_many.nInputCols = 1
  settings.feedforward1SKO_many.nTargetCols = 2
  settings.feedforward1SKO_many.nPrt= 3


  return settings
end

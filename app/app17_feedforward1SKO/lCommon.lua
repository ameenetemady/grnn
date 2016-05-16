function genRunSettings(taBaseSettings, id)
  local taSettings = { 
    taInput = {
      strFilename = string.format("%s/d_%d/processed/input.tsv", taBaseSettings.baseDir, id),
      nCols = taBaseSettings.nInputCols
    },
    taTarget = {
      strFilename = string.format("%s/d_%d/processed/target.tsv", taBaseSettings.baseDir, id),
      nCols = taBaseSettings.nTargetCols
    },
    taPrt = {
      strFilename = string.format("%s/d_%d/processed/ptr.tsv", taBaseSettings.baseDir, id),
      nCols = taBaseSettings.nTargetCols
    }

  }

  return taSettings
end


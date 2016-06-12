#!/bin/bash
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19_grnn_noisy runOne.sh test_grnn_kfold_MNetTrainer.lua false
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19_fnn_h0_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua false 0
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19_fnn_h1_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua false 1
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19_fnn_h2_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua false 2
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19_fnn_h3_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua false 3
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19__grnn_noisy runOne.sh test_grnn_kfold_MNetTrainer.lua true
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19__fnn_h0_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua true 0
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19__fnn_h1_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua true 1
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19__fnn_h2_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua true 2
qsub -V -cwd -q flavor.q -S /bin/bash -N japp19__fnn_h3_noisy runOne.sh test_fnn_kfold_FnnTrainer.lua true 3

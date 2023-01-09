#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=jiif-test-nocell
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jean.legoff@isir.upmc.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
cd /data/jean.legoff/jiif
python train_jiif.py --config configs/jiif_test_config_nocell.yaml

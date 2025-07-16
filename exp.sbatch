#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --output=download.out

conda init bash
source ~/.bashrc
conda activate agentsearch

export PYTHONUNBUFFERED=1
python -u -m scripts.download
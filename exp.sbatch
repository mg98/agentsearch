#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0

module load cuda12.6/toolkit/12.6

conda init bash
source ~/.bashrc
conda activate agentsearch

set -a
source .env
set +a

python -u -m scripts.compute_test_result_matrix

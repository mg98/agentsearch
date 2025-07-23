#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0

conda init bash
source ~/.bashrc
conda activate agentsearch

set -a
source .env
set +a

python -u -m agentsearch.graph.graph2

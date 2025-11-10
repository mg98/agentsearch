#!/bin/bash

#SBATCH --job-name=reports
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

module load cuda12.6/toolkit/12.6

conda init bash
source ~/.bashrc
conda activate agentsearch

export $(cat .env | xargs)

srun zsh -c "
    python -m scripts.gen_reports \
        --job-id=${SLURM_ARRAY_TASK_ID} \
        --job-count=${SLURM_ARRAY_TASK_COUNT}
"
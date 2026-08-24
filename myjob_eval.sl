#!/bin/bash -e

#SBATCH --job-name=GATES
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_out_test/%j.out

out_dir="slurm_out_test"
mkdir -p "$out_dir"

# python3 eval_rl_01.py -g $1 -w $2 -log $3 -m $4  # test on the specific model
python3 eval_rl_02.py -g $1 -w $2 -log $3 -m $4  # test on all episode models

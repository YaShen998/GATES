#!/bin/bash

#SBATCH --job-name=GATES
#SBATCH --time=168:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_out_train/%j.out

out_dir="slurm_out_train" 
mkdir -p "$out_dir"

python3 main_01.py -r $1

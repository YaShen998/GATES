#!/bin/bash

#SBATCH --job-name=GATES
#SBATCH --time=168:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1

python3 main_01.py -r $1

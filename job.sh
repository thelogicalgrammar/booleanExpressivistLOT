#!/bin/bash
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 05:00

module load 2019
module load Python/3.6.6-intel-2019b

python utilities.py --primitives $1 --action "single_minimal"

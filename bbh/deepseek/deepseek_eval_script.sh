#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --job-name=deepseek_bbh
#SBATCH --output=outlog
#SBATCH --error=errlog

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python deepseek_eval.py
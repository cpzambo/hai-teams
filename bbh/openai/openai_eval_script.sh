#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=2:30:00
#SBATCH --job-name=openai_bbh
#SBATCH --output=outlog
#SBATCH --error=errlog

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python openai_eval.py
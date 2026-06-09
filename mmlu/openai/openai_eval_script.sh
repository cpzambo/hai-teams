#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --job-name=openai_mmlu
#SBATCH --output=openai_outlog
#SBATCH --error=openai_errlog

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python openai_rescore.py

#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --job-name=gemini_mmlu
#SBATCH --output=gemini_outlog
#SBATCH --error=gemini_errlog

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python gemini_eval.py
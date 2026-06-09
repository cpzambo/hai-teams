#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=118:00:00
#SBATCH --job-name=gemma2_docvqa
#SBATCH --output=gemma2_outlog
#SBATCH --error=gemma2_errlog

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python gemma_eval_half2.py
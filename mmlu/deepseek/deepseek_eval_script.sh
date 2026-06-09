#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=36:00:00
#SBATCH --job-name=deepseek_mmlu
#SBATCH --output=deepseek_outlog
#SBATCH --error=deepseek_errlog

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

<<<<<<< HEAD
python deepseek_eval.py
=======
python deepseek_rescore.py
>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2

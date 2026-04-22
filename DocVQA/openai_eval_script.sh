#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=openai_docvqa
#SBATCH --output=openai_outlog
#SBATCH --error=openai_errlog

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python openai_eval.py

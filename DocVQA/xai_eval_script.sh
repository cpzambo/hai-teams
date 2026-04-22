#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=xai_docvqa
#SBATCH --output=xai_outlog
#SBATCH --error=xai_errlog

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python xai_eval.py

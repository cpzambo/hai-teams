#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --job-name=llama_mmlu
#SBATCH --output=llama_outlog
#SBATCH --error=llama_errlog

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python llama_eval.py

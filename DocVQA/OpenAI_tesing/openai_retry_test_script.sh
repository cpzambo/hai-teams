#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=2:00:00
#SBATCH --job-name=openai_docvqa_retry_test
#SBATCH --output=openai_retry_test_outlog
#SBATCH --error=openai_retry_test_errlog

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python openai_retry_test.py

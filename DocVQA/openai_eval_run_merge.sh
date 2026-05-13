#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --job-name=openai_merge
#SBATCH --output=log_merge.txt
#SBATCH --error=log_merge.err

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python merge_openai_results.py --total-shards 5

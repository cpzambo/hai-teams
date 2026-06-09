#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_emobench
#SBATCH --output=log.txt
#SBATCH --error=log.err

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python qwen_emo_eval.py \
    --model Qwen/Qwen3.5-9B \
    --task all \
    --save-every 20

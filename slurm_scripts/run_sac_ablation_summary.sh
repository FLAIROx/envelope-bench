#!/bin/bash
#SBATCH --job-name=sac_ablation_summary
#SBATCH -c 4
#SBATCH -t 0-00:20
#SBATCH -p seas_gpu
#SBATCH --account=gershman_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o slurmlogs/slurm-%x-%A.out
#SBATCH -e slurmlogs/slurm-%x-%A.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

export UV_LINK_MODE=copy
export UV_NO_SYNC=1

uv run --no-sync python -m ppo_sharding.summarize_ablation \
  --entity "aneeshmuppidi19" \
  --project "envelope-sac-ablation" \
  --metric "episode/return" \
  --final_window 10 \
  --summary_run_name "ablation_summary"

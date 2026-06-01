#!/bin/bash
#SBATCH --job-name=panda_open_cabinet_sac_sharding
#SBATCH -c 4
#SBATCH -t 0-02:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -o slurmlogs/slurm-%x-%A_%a.out
#SBATCH -e slurmlogs/slurm-%x-%A_%a.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

export UV_LINK_MODE=copy

uv run --group ppo --group adapters python -m ppo_sharding.sac \
  --env_name "mujoco_playground::PandaOpenCabinet" \
  --total_timesteps 1500000000 \
  --num_checkpoints 1 \
  --num_envs 8192 \
  --buffer_size 1000000 \
  --batch_size 256 \
  --updates_per_step 1 \
  --random_steps 10000 \
  --log_every 20000 \
  --use_wandb True \
  --wandb_entity "aneeshmuppidi19" \
  --wandb_project "envelope-sac-test"

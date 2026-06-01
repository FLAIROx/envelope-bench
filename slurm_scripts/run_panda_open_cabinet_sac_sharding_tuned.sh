#!/bin/bash
#SBATCH --job-name=panda_open_cabinet_sac_sharding_tuned
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
  --num_envs 4096 \
  --buffer_size 1000000 \
  --batch_size 4096 \
  --updates_per_step 1 \
  --random_steps 10000 \
  --normalize_observations True \
  --critic_layer_norm True \
  --num_layers 3 \
  --log_every 20000 \
  --use_wandb True \
  --wandb_entity "aneeshmuppidi19" \
  --wandb_project "envelope-sac-test"

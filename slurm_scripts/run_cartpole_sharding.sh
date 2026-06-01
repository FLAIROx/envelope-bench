#!/bin/bash
#SBATCH --job-name=cartpole_test
#SBATCH -c 2
#SBATCH -t 0-01:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

uv run python -m ppo_sharding.ppo \
  --env_name "gymnax::CartPole-v1" \
  --total_timesteps 5000000 \
  --num_envs 16000 \
  --num_steps 64 \
  --log_every 20000 \
  --use_wandb True \
  --wandb_entity "aneeshmuppidi19" \
  --wandb_project "envelope-ppo-test"

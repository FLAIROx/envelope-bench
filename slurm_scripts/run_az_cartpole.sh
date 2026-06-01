#!/bin/bash
#SBATCH --job-name=az_cartpole
#SBATCH -c 2
#SBATCH -t 0-02:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

uv run python -m alpha_zero.az \
  --env_name "gymnax::CartPole-v1" \
  --total_timesteps 5000000 \
  --num_envs 2048 \
  --num_simulations 32 \
  --num_steps 20 \
  --num_epochs 4 \
  --num_minibatches 8 \
  --log_every 20000 \
  --use_wandb True \
  --wandb_entity "aneeshmuppidi19" \
  --wandb_project "envelope-az"

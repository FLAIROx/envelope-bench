#!/bin/bash
#SBATCH --job-name=az_sokoban
#SBATCH -c 2
#SBATCH -t 1-00:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

uv run --group ppo --group az --group adapters python -m alpha_zero.az \
  --env_name "jumanji::Sokoban-v0" \
  --total_timesteps 5000000 \
  --num_envs 64 \
  --num_simulations 32 \
  --num_steps 20 \
  --num_epochs 1 \
  --num_minibatches 1 \
  --gumbel_scale 1.0 \
  --gamma 0.99 \
  --learning_rate 3e-4 \
  --layer_size 256 \
  --num_layers 3 \
  --log_every 20000 \
  --use_wandb True \
  --wandb_entity "aneeshmuppidi19" \
  --wandb_project "envelope-az" \
  --stagger_steps 0

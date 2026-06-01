#!/bin/bash
#SBATCH --job-name=leap_reinforce
#SBATCH -c 4
#SBATCH -t 0-02:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH -o slurmlogs/slurm-%x-%A_%a.out
#SBATCH -e slurmlogs/slurm-%x-%A_%a.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

uv run python -m ppo_sharding.reinforce \
  --env_name "mujoco_playground::LeapCubeReorient" \
  --total_timesteps 1800000000 \
  --num_envs 64000 \
  --num_steps 64 \
  --log_every 20000 \
  --value_epochs 10 \
  --use_wandb True \
  --wandb_entity "aneeshmuppidi19" \
  --wandb_project "envelope-ppo-test"

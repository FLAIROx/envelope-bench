#!/bin/bash
#SBATCH --job-name=panda_open_cabinet_sac_stagea
#SBATCH -c 4
#SBATCH -t 0-08:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=0-26%12
#SBATCH -o slurmlogs/slurm-%x-%A_%a.out
#SBATCH -e slurmlogs/slurm-%x-%A_%a.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

export UV_LINK_MODE=copy
# Read-only consumer of the pre-synced shared .venv (see prior NFS corruption notes).
export UV_NO_SYNC=1

# Stage A HP sweep on the ablation winner (no_obsnorm base): obs-norm OFF, critic
# LayerNorm ON, 3 layers, 4096 envs / batch 4096. Vary the two highest-leverage
# SAC knobs: updates_per_step (UTD) x target_entropy. 9 configs x 3 seeds = 27 tasks.
#   config_idx = id % 9 ; seed = id // 9
#   updates_per_step = {1,1,1, 2,2,2, 4,4,4}[config_idx]
#   target_entropy   = {-8,-4,-2}[config_idx % 3]   (action_dim=8, so -8 = default)
# updates_per_step=1, target_entropy=-8 reproduces no_obsnorm (anchor).
CONFIG=$((SLURM_ARRAY_TASK_ID % 9))
SEED=$((SLURM_ARRAY_TASK_ID / 9))
UPS_LIST=(1 1 1 2 2 2 4 4 4)
TE_LIST=(-8 -4 -2 -8 -4 -2 -8 -4 -2)
UPS=${UPS_LIST[$CONFIG]}
TE=${TE_LIST[$CONFIG]}

echo "Stage A: updates_per_step=$UPS target_entropy=$TE seed=$SEED (config $CONFIG)"

# Stagger initial wave to avoid concurrent NFS asset-read bursts at env construction.
sleep $(( (SLURM_ARRAY_TASK_ID % 12) * 20 ))

run_sac() {
  uv run --no-sync python -m ppo_sharding.sac \
    --env_name "mujoco_playground::PandaOpenCabinet" \
    --total_timesteps 1500000000 \
    --num_checkpoints 0 \
    --num_envs 4096 \
    --buffer_size 1000000 \
    --batch_size 4096 \
    --updates_per_step "$UPS" \
    --random_steps 10000 \
    --normalize_observations False \
    --critic_layer_norm True \
    --num_layers 3 \
    --target_entropy="$TE" \
    --seed "$SEED" \
    --run_name "stagea_u${UPS}_te${TE}_seed${SEED}" \
    --log_every 20000 \
    --use_wandb True \
    --wandb_entity "aneeshmuppidi19" \
    --wandb_project "envelope-sac-stagea"
}

# One retry for startup-only flakiness (NFS asset reads). num_checkpoints=0 removes
# the end-of-run orbax hang, so mid-run failures are not expected.
max_attempts=2
for attempt in $(seq 1 $max_attempts); do
  echo "=== Attempt $attempt/$max_attempts (u=$UPS te=$TE seed=$SEED) ==="
  run_sac && { echo "=== Succeeded on attempt $attempt ==="; exit 0; }
  rc=$?
  echo "=== Attempt $attempt failed (exit $rc) ==="
  [ "$attempt" -lt "$max_attempts" ] && sleep $(( RANDOM % 30 + 15 ))
done
echo "=== All $max_attempts attempts failed (u=$UPS te=$TE seed=$SEED) ==="
exit 1

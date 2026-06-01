#!/bin/bash
#SBATCH --job-name=panda_open_cabinet_sac_ablation
#SBATCH -c 4
#SBATCH -t 0-03:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=0-17%12
#SBATCH -o slurmlogs/slurm-%x-%A_%a.out
#SBATCH -e slurmlogs/slurm-%x-%A_%a.err

cd /n/netscratch/gershman_lab/Lab/amuppidi/work/envelope-bench

export UV_LINK_MODE=copy

# Run against the already-synced shared .venv WITHOUT mutating it. Concurrent
# `uv run` syncs over NFS corrupt the venv (stale file handles / SIGBUS); --no-sync
# makes every task a read-only consumer of the venv. Sync once before submitting:
#   uv sync --group ppo --group adapters
export UV_NO_SYNC=1

# Leave-one-out ablation x 3 seeds = 18 tasks (array 0-17), throttled to 12
# concurrent (%12) for the kempner_h100 GPU cap, 1 GPU per task.
# Variant 0 is the full tuned config; variants 1-4 each revert exactly ONE change
# back to its baseline value; variant 5 is the original baseline (all reverted).
# The full->no_X drop measures the marginal contribution of change X.
#
#   factor          baseline           tuned
#   UTD             num_envs=8192       num_envs=4096
#                   batch_size=256      batch_size=4096
#   obs norm        False              True
#   critic norm     False              True
#   num_layers      2                  3
VARIANT=$((SLURM_ARRAY_TASK_ID % 6))
SEED=$((SLURM_ARRAY_TASK_ID / 6))
case $VARIANT in
  0) TAG=full;          NUM_ENVS=4096; BATCH=4096; OBSNORM=True;  CRITICNORM=True;  LAYERS=3 ;;
  1) TAG=no_utd;        NUM_ENVS=8192; BATCH=256;  OBSNORM=True;  CRITICNORM=True;  LAYERS=3 ;;
  2) TAG=no_obsnorm;    NUM_ENVS=4096; BATCH=4096; OBSNORM=False; CRITICNORM=True;  LAYERS=3 ;;
  3) TAG=no_criticnorm; NUM_ENVS=4096; BATCH=4096; OBSNORM=True;  CRITICNORM=False; LAYERS=3 ;;
  4) TAG=no_3layers;    NUM_ENVS=4096; BATCH=4096; OBSNORM=True;  CRITICNORM=True;  LAYERS=2 ;;
  5) TAG=baseline;      NUM_ENVS=8192; BATCH=256;  OBSNORM=False; CRITICNORM=False; LAYERS=2 ;;
  *) echo "Unknown VARIANT=$VARIANT"; exit 1 ;;
esac

echo "Running ablation variant: $TAG seed=$SEED (num_envs=$NUM_ENVS batch=$BATCH obsnorm=$OBSNORM criticnorm=$CRITICNORM layers=$LAYERS)"

# Stagger the initial wave so concurrent tasks don't hammer NFS while reading
# mujoco_playground's bundled XML/mesh assets at env-construction time (transient
# "Error opening file 'mjx_panda.xml'" failures). Only the first %12 wave starts
# simultaneously; queued tasks are released one-at-a-time and don't need spreading.
sleep $(( (SLURM_ARRAY_TASK_ID % 12) * 20 ))

# Retry startup-only flakiness: a failed env construction logs no data, so a clean
# restart is safe. Mid-training failures are not expected now that the venv race is
# fixed; if one happens, the retry simply starts a fresh (deduped) run.
run_sac() {
  uv run --no-sync python -m ppo_sharding.sac \
    --env_name "mujoco_playground::PandaOpenCabinet" \
    --total_timesteps 1500000000 \
    --num_checkpoints 1 \
    --num_envs "$NUM_ENVS" \
    --buffer_size 1000000 \
    --batch_size "$BATCH" \
    --updates_per_step 1 \
    --random_steps 10000 \
    --normalize_observations "$OBSNORM" \
    --critic_layer_norm "$CRITICNORM" \
    --num_layers "$LAYERS" \
    --seed "$SEED" \
    --run_name "ablation_${TAG}_seed${SEED}" \
    --log_every 20000 \
    --use_wandb True \
    --wandb_entity "aneeshmuppidi19" \
    --wandb_project "envelope-sac-ablation"
}

max_attempts=3
for attempt in $(seq 1 $max_attempts); do
  echo "=== Attempt $attempt/$max_attempts ($TAG seed=$SEED) ==="
  run_sac && { echo "=== Succeeded on attempt $attempt ==="; exit 0; }
  rc=$?
  echo "=== Attempt $attempt failed (exit $rc) ==="
  [ "$attempt" -lt "$max_attempts" ] && sleep $(( RANDOM % 30 + 15 ))
done
echo "=== All $max_attempts attempts failed for $TAG seed=$SEED ==="
exit 1

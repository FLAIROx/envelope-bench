#!/bin/bash
set -e

SWEEP_DIR="sweep_configs_random/envs"
ARRAY_SIZE=1       # 0-1 = 2 agents per sweep
TIME="24:00:00"
WANDB_ENTITY="flair"
WANDB_PROJECT="envelope-bench"

if [ ! -d "$SWEEP_DIR" ]; then
    echo "ERROR: $SWEEP_DIR not found. Run: uv run generate_random_sweep_configs.py"
    exit 1
fi

source "$(dirname "$0")/.venv/bin/activate"

for config in "$SWEEP_DIR"/*.yaml; do
    sweep_name=$(basename "$config" .yaml)
    echo "=== Creating sweep from: $config ==="

    sweep_output=$(wandb sweep --entity flair --project envelope-bench-random "$config" 2>&1)
    echo "$sweep_output"

    sweep_id=$(echo "$sweep_output" | grep -oP 'wandb agent \K\S+' | tail -1)

    if [ -z "$sweep_id" ]; then
        echo "ERROR: Failed to extract sweep ID from: $config"
        continue
    fi

    echo "Sweep ID: $sweep_id"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=rs-${sweep_name}
#SBATCH --output=outputs/random/${sweep_name}-%A_%a.out
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --array=0-${ARRAY_SIZE}

cd ~/envelope-bench
source .venv/bin/activate
hostname
nvidia-smi --list-gpus

srun python -m wandb agent ${sweep_id}
EOF

    echo "Submitted SLURM job for $sweep_name (sweep: $sweep_id)"
    echo ""
done

#!/bin/bash
set -e

SWEEP_DIR="sweep_configs"
ARRAY_SIZE=1       # 0-30 = 31 agents per sweep
TIME="24:00:00"
WANDB_ENTITY="flair"
WANDB_PROJECT="envelope-bench"

for config in "$SWEEP_DIR"/*.yaml; do
    sweep_name=$(basename "$config" .yaml)
    echo "=== Creating sweep from: $config ==="

    # Create the sweep and capture the sweep ID from the last line of output
    # wandb sweep prints: "Created sweep with ID: <id>"
    # and then: "wandb agent ENTITY/PROJECT/SWEEP_ID"
    sweep_output=$(wandb sweep --entity flair "$config" 2>&1)
    echo "$sweep_output"

    # Extract sweep ID from the "wandb agent ..." line
    sweep_id=$(echo "$sweep_output" | grep -oP 'wandb agent \K\S+' | tail -1)

    if [ -z "$sweep_id" ]; then
        echo "ERROR: Failed to extract sweep ID from: $config"
        echo "$sweep_output"
        continue
    fi

    echo "Sweep ID: $sweep_id"

    # Submit SLURM array job for this sweep
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${sweep_name}
#SBATCH --output=outputs/${sweep_name}-%A_%a.out
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

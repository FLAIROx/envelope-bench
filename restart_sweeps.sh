#!/bin/bash
set -e

# Restart wandb sweep agents for existing sweeps.
# Usage: ./restart_sweeps.sh
#
# Set SWEEP_IDS below to the sweep IDs from your original launch.
# You can find them with: wandb sweep --list or in the wandb UI.

ARRAY_SIZE=2       # match the original launch
TIME="24:00:00"
WANDB_ENTITY="flair"
WANDB_PROJECT="envelope-bench"

# Map sweep name -> sweep ID (fill these in from wandb UI or logs)
declare -A SWEEP_IDS=(
    # ["sweep_config_act"]="flair/envelope-bench/XXXXXXXX"
    # ["sweep_config_adam_eps"]="flair/envelope-bench/XXXXXXXX"
    # ...
)

# If SWEEP_IDS is empty, try to auto-detect from wandb
if [ ${#SWEEP_IDS[@]} -eq 0 ]; then
    echo "No SWEEP_IDS set manually, fetching from wandb API..."

    # List all sweeps and match by name to config files
    sweeps_json=$(wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" list 2>&1 || true)

    # Fallback: ask user to provide sweep IDs via a file
    if [ -z "$sweeps_json" ] || echo "$sweeps_json" | grep -qi "error"; then
        echo ""
        echo "Could not auto-detect sweeps. Please create a file 'sweep_ids.txt'"
        echo "with one entry per line in the format:"
        echo "  sweep_name sweep_id"
        echo ""
        echo "Example:"
        echo "  sweep_config_act TWWB/envelope-bench/abc12345"
        echo "  sweep_config_lr  TWWB/envelope-bench/def67890"
        echo ""

        if [ -f sweep_ids.txt ]; then
            echo "Found sweep_ids.txt, reading..."
            while read -r name sid; do
                [ -z "$name" ] && continue
                [[ "$name" == \#* ]] && continue
                SWEEP_IDS["$name"]="$sid"
            done < sweep_ids.txt
        else
            echo "No sweep_ids.txt found. Exiting."
            exit 1
        fi
    fi
fi

if [ ${#SWEEP_IDS[@]} -eq 0 ]; then
    echo "ERROR: No sweep IDs found. Exiting."
    exit 1
fi

echo "Restarting agents for ${#SWEEP_IDS[@]} sweeps..."
echo ""

for sweep_name in "${!SWEEP_IDS[@]}"; do
    sweep_id="${SWEEP_IDS[$sweep_name]}"
    echo "=== Restarting: $sweep_name ($sweep_id) ==="

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

echo "Done. All sweep agents resubmitted."

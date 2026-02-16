# Determine GPU argument
GPU_ID="$1"
if [ -n "$GPU_ID" ]; then
    echo "Using GPU device(s): $GPU_ID"
    GPU_ARG="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPU_ID"
    shift
else
    echo "No GPU specified, using no GPU"
    GPU_ID="cpu"
    GPU_ARG=""
fi

# Set the username inside the container (must match the user set up in the Dockerfile)
USERNAME=$(whoami)

# If additional arguments remain, use them as the command
if [ $# -gt 0 ]; then
    CMD=("$@")
    RUN_FLAGS="-dit --rm"   # detached + TTY so attach/detach with Ctrl+P, Ctrl+Q
else
    CMD=("/bin/bash")
    RUN_FLAGS="-it --rm"  # interactive shell when no command
fi

# Load secrets file if it exists
SECRETS_FILE="$(dirname "$0")/secrets.env"
if [ -f "$SECRETS_FILE" ]; then
    ENV_FILE_ARG="--env-file $SECRETS_FILE"
    echo "Loading secrets from: $SECRETS_FILE"
else
    ENV_FILE_ARG=""
    echo "Warning: secrets file not found at $SECRETS_FILE"
fi

# Run container
docker run \
    ${RUN_FLAGS} \
    --shm-size=16g \
    -v "$(pwd)":/app \
    -v "$(pwd)/cache:/cache" \
    ${GPU_ARG} \
    ${ENV_FILE_ARG} \
    --name "${USERNAME}_envelope_${GPU_ID//,/-}" \
    "${USERNAME}_envelope_gpu" \
    "${CMD[@]}"

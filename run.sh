#!/usr/bin/env bash
set -e

IMAGE_NAME="tlens-demo"
CONTAINER_NAME="tlens-demo"
USE_GPU=true

# Parse flags
for arg in "$@"; do
  case $arg in
    --cpu) USE_GPU=false ;;
  esac
done

# Build
docker build -t "$IMAGE_NAME" .

# Remove any existing container with the same name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Run
GPU_FLAG=""
if [ "$USE_GPU" = true ]; then
  GPU_FLAG="--gpus all"
fi

docker run $GPU_FLAG \
  --name "$CONTAINER_NAME" \
  -p 7860:7860 \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  "$IMAGE_NAME"

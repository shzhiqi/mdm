#!/usr/bin/env bash
set -e

IMAGE_NAME="tlens-demo"
CONTAINER_NAME="tlens-demo"
USE_GPU=""

# Parse flags
for arg in "$@"; do
  case $arg in
    --cpu) USE_GPU=false ;;
    --gpu) USE_GPU=true ;;
  esac
done

if [ -z "$USE_GPU" ]; then
  echo "Usage: $0 --cpu | --gpu"
  exit 1
fi

# Build
docker build --platform linux/amd64 -t "$IMAGE_NAME" .

# Remove any existing container with the same name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Run
GPU_FLAG=""
if [ "$USE_GPU" = true ]; then
  GPU_FLAG="--gpus all"
fi

docker run $GPU_FLAG \
  --platform linux/amd64 \
  --name "$CONTAINER_NAME" \
  -p 7860:7860 \
  -v "$(pwd)/demo:/app" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  "$IMAGE_NAME"

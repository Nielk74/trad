#!/usr/bin/env bash
set -e

CONFIG=${1:-small}
HOST=${2:-0.0.0.0}
PORT=${3:-8080}

VLLM_PID=""

cleanup() {
  if [ -n "$VLLM_PID" ]; then
    echo "Stopping vLLM (pid $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$CONFIG" = "high" ]; then
  echo "Starting vLLM for Voxtral-4B-TTS-2603..."
  vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --host 127.0.0.1 --port 8000 &
  VLLM_PID=$!

  echo "Waiting for vLLM to be ready..."
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
      echo "vLLM is ready."
      break
    fi
    sleep 2
  done
fi

echo "Starting FastAPI (config=$CONFIG, port=$PORT)..."
python app.py --config "$CONFIG" --host "$HOST" --port "$PORT"

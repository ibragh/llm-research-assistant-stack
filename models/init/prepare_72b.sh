#!/usr/bin/env bash
set -euo pipefail

OLLAMA_URL="http://localhost:11434"
WEIGHTS_DIR="/models/weights"
GGUF_PATH="${WEIGHTS_DIR}/qwen2-72b-q4_k_m.gguf"
# Override QWEN72B_Q4KM_URL to use a different mirror if desired
GGUF_URL="${QWEN72B_Q4KM_URL:-https://huggingface.co/Qwen/Qwen2-72B-Instruct-GGUF/resolve/main/qwen2-72b-instruct-q4_k_m.gguf}"

echo "[prepare_72b] Waiting for Ollama API..."
until curl -sf "${OLLAMA_URL}/api/tags" >/dev/null; do sleep 1; done

mkdir -p "${WEIGHTS_DIR}"
if [ ! -f "${GGUF_PATH}" ]; then
  echo "[prepare_72b] Downloading GGUF (this may take a while): ${GGUF_URL}"
  curl -L "${GGUF_URL}" -o "${GGUF_PATH}.partial"
  mv "${GGUF_PATH}.partial" "${GGUF_PATH}"
fi

if ! ollama list | grep -q "^qwen2-72b-q4km"; then
  echo "[prepare_72b] Building custom model qwen2-72b-q4km..."
  ollama create qwen2-72b-q4km -f /models/qwen2-72b-q4km/Modelfile
fi

echo "[prepare_72b] Pre-pulling embedding model..."
ollama pull multilingual-e5-large || true
echo "[prepare_72b] Done."

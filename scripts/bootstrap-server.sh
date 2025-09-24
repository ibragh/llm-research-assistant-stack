#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
# Install Docker & Compose plugin
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Create app dir
mkdir -p "${1:-$HOME/llm-stack}"
echo "Done. Re-login to get docker group."

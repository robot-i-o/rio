#!/usr/bin/env bash

# Run from the repo root:
#   bash scripts/setup/humanoid/gearsonic_setup.sh [--cuda13]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CUDA13=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda13)
            CUDA13=true
            shift
            ;;
        *)
            echo "Usage: bash scripts/setup/humanoid/gearsonic_setup.sh [--cuda13]" 1>&2
            exit 1
            ;;
    esac
done

GROOT_DIR="third_party/GR00T-WholeBodyControl"

echo "Starting GEAR-SONIC policy setup..."
echo "==================================="

echo "[1/5] Installing git-lfs..."
if ! command -v git-lfs >/dev/null 2>&1; then
    sudo apt install -y git-lfs
fi
git lfs install

echo "[2/5] Cloning GR00T-WholeBodyControl..."
if [ ! -d "$GROOT_DIR" ]; then
    mkdir -p third_party
    git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$GROOT_DIR"
else
    echo "$GROOT_DIR already exists, skipping clone."
fi
(cd "$GROOT_DIR" && git lfs pull)

echo "[3/5] Downloading models from Hugging Face..."
uv pip install huggingface_hub
(cd "$GROOT_DIR" && uv run python download_from_hf.py)

echo "[4/5] Installing gear_sonic module (editable)..."
uv pip install -e "$GROOT_DIR/gear_sonic"

if [ "$CUDA13" = true ]; then
    echo "[5/5] Installing onnxruntime-gpu (CUDA 13.x nightly)..."
    uv pip install coloredlogs flatbuffers numpy packaging protobuf sympy
    uv pip install --pre \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
        onnxruntime-gpu
else
    echo "[5/5] Installing onnxruntime-gpu (CUDA 12.x)..."
    uv pip install onnxruntime-gpu
fi
uv run python -c "import onnxruntime as ort; print('ONNX providers:', ort.get_available_providers())"

echo "==================================="
echo "GEAR-SONIC policy setup complete."
echo "Next, complete the teleoperation and Unitree G1 interface setup manually"
echo "(see the Humanoid Control tutorial)."

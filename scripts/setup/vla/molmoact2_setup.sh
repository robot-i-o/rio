#!/usr/bin/env bash

# MolmoAct2 Setup Script
# Installs MolmoAct2 into the current rio venv (third_party/molmoact2 is cloned for local
# patching). The checkpoints ship their own modeling code and were saved by transformers
# 5.3, which is newer than openpi's pin, so this will break a Pi0 install in the same venv.
# Run from the repo root: bash scripts/setup/vla/molmoact2_setup.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "Starting MolmoAct2 installation..."
echo "================================"

# No --recurse-submodules: the root lerobot/ submodule is the separate LeRobot-native
# deploy path, and it aborts on YAM, which declares a nested submodule with no url.
if [ ! -d "third_party/molmoact2" ]; then
    echo "Cloning molmoact2 into third_party/molmoact2..."
    mkdir -p third_party
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        https://github.com/allenai/molmoact2.git \
        third_party/molmoact2
else
    echo "third_party/molmoact2 already exists, skipping clone."
fi

# transformers 5.3 is a hard floor: the checkpoints' remote modeling code imports symbols
# earlier releases do not have. einops and torchvision are imported by that same remote
# code (image_processing_molmoact2.py) and are not pulled in by anything else.
uv pip install \
    "transformers>=5.3.0" \
    "torch" \
    "torchvision" \
    "einops" \
    "pillow" \
    "huggingface_hub" \
    "accelerate" \
	"omegaconf" \
	"cached_path" \
	"torchmetrcis" \
	"datasets" \
	"decord" \

"${VIRTUAL_ENV:-.venv}/bin/python" - <<'PYEOF'
import sys

import torch

# A torch whose wheel predates the GPU fails quietly: import and is_available() both
# succeed, and only the first real kernel launch dies.
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device sm_{cap[0]}{cap[1]}")
if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
    sys.exit(f"torch built for {torch.cuda.get_arch_list()}, GPU is sm_{cap[0]}{cap[1]}; install a newer torch")
torch.zeros(8, device="cuda").sum().item()  # the check that actually matters
print("CUDA kernel OK")
PYEOF

echo "================================"
echo "MolmoAct2 installation complete."
echo
echo "Fetch the bimanual YAM checkpoint (~22GB) with:"
echo "  hf download allenai/MolmoAct2-BimanualYAM"

#!/usr/bin/env bash

# MolmoAct2 Training Setup Script
# Builds .venv-molmoact2-train (Python 3.12) for data conversion and fine-tuning, and leaves
# the project venv alone. Training cannot share it: the vendored LeRobot fork uses PEP 695
# syntax, so it is a SyntaxError on the 3.11 that rio's camera stack pins.
# Run molmoact2_setup.sh first for inference and deployment.
# Run from the repo root: bash scripts/setup/vla/molmoact2_train_setup.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

VENV=".venv-molmoact2-train"

echo "Starting MolmoAct2 training installation..."
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

uv venv "$VENV" --python 3.12

# --no-config throughout: this repo's pyproject overrides av to <=13.1.0 for robodm, and uv
# applies that to anything installed from the repo root. LeRobot v3.0 wants av>=15, and
# robodm decodes .vla identically under either, so let LeRobot's constraint win here.
TPY="--python $VENV/bin/python --no-config"

# shellcheck disable=SC2086
uv pip install $TPY -e "third_party/molmoact2/experiments[all]"
# shellcheck disable=SC2086
uv pip install $TPY -e "third_party/molmoact2/experiments/lerobot[async]"

# rio without its dependencies: conversion reads the repo's own modules, and the station
# stack underneath them pins a pyrealsense2 with no cp312 wheel. debugpy is imported at
# module scope by launch_scripts/train_lerobot.py but is not in the experiments extras.
# shellcheck disable=SC2086
uv pip install $TPY --no-deps -e .
# shellcheck disable=SC2086
uv pip install $TPY robodm tyro loguru debugpy

# protobuf last, and pinned to the only window all three consumers accept. LeRobot's
# grpcio-dep extra asks for <6.32, but olmo.io imports cached_path, whose google-cloud
# stack carries gencode 6.33.5 and refuses an older runtime; wandb's generated stubs
# break on 7. The trainer cannot be imported outside this range.
# shellcheck disable=SC2086
uv pip install $TPY "protobuf>=6.33.5,<7"

"$VENV/bin/python" - <<'PYEOF'
import sys

import torch

# A torch whose wheel predates the GPU fails quietly: import and is_available() both
# succeed, and only the first real kernel launch dies.
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device sm_{cap[0]}{cap[1]}")
if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
    sys.exit(f"torch built for {torch.cuda.get_arch_list()}, GPU is sm_{cap[0]}{cap[1]}; install a newer torch")
torch.zeros(8, device="cuda").sum().item()  # the check that actually matters

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

if CODEBASE_VERSION != "v3.0":
    sys.exit(f"expected LeRobot dataset format v3.0, got {CODEBASE_VERSION}")

sys.path.insert(0, "third_party/molmoact2/experiments")
from launch_scripts import train_lerobot  # noqa: F401

print(f"trainer imports, LeRobot dataset format {CODEBASE_VERSION}")
PYEOF

echo "================================"
echo "MolmoAct2 training installation complete."
echo
echo "Convert and train with: $VENV/bin/python"
echo "Walkthrough: docs/workflow/molmoact2_finetuning.md"

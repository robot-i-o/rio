<p align="center">
  <a href="https://github.com/astral-sh/uv">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" /></a>
  <a href="https://github.com/astral-sh/ruff">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
  <a href="https://github.com/astral-sh/ty">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" /></a>
</p>

# RIO

RIO: flexible real-time Robot I/O for cross-embodiment robot learning.

This project provides a Python-based interface to use different robot arms (Franka, Kinova, Universal Robots, UFACTORY, SO100, ...), grippers, cameras, and teleop interfaces, with built-in support for data collection, teleoperation, and Vision-Language-Action (VLA) policy deployment.

## Setup

Tested on Ubuntu 22.04 LTS with an optional real-time kernel patch. See [`docs/ubuntu.md`](docs/ubuntu.md) for setup instructions.

```bash
git clone git@github.com:robot-i-o/rio.git

# install open-pi
cd rio
mkdir third_party && cd third_party
git clone git@github.com:Physical-Intelligence/openpi.git

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install dependencies
uv venv --python 3.10
source .venv/bin/activate
uv sync --all-extras
```

## Documentation

Build and browse the docs locally at http://localhost:8000:

```bash
uv sync --group docs
uv run mkdocs serve
```

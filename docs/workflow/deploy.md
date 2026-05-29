# Policy Inference

Run a fine-tuned VLA policy on the robot in closed-loop at a fixed frequency.

## 1. Install policy dependencies

Policy inference needs `openpi` and `lerobot`. They're opt-in (the install is large and conflicts with the `docs` extra):

```bash
uv sync --group pi
```

## 2. Pick a station and policy

| Env var | Purpose | Where defined |
|---------|---------|---------------|
| `STATION` | Robot hardware config (arm, gripper, cameras) | `examples/cfg/` |
| `POLICY`  | Policy config (checkpoint, transforms, action space) | `examples/policy_cfgs/` |

**Available policies:**

| `POLICY` | Model | Notes |
|----------|-------|-------|
| `Pi05Cfg` | [openpi](https://github.com/Physical-Intelligence/openpi) pi0.5 (DROID-style) | 3 cameras, 32-dim actions, `joint_vel` |
| `SmolVLACfg` | [lerobot SmolVLA](https://github.com/huggingface/lerobot) | 2 cameras, `joint_pos` |

List available stations and policies:

```bash
uv run -m examples.cfg
uv run -m examples.policy_cfgs
```

## 3. Point the policy at your checkpoint

The three fields you'll typically set live at the top of each policy config:

| Field | Pi05Cfg | SmolVLACfg |
|-------|---------|------------|
| `policy_path` | Dir with `params/` + `assets/` | lerobot checkpoint dir |
| `asset_id`    | Subfolder under `<policy_path>/assets/` containing `norm_stats.json` | — |
| `instruction` | Language prompt for the task | Language prompt (optional) |

You can either edit the defaults in [`examples/policy_cfgs/pi05.py`](../../examples/policy_cfgs/pi05.py) / [`smolvla.py`](../../examples/policy_cfgs/smolvla.py), or override them on the CLI (see below).

The `action_space` in the policy config automatically sets the arm controller — no need to edit station files.

## 4. Run

```bash
# pi0.5
STATION=Xarm7EEFStation POLICY=Pi05Cfg uv run -m examples.policy_inference \
  --policy-path /data/ckpt/pi05_multi_fold \
  --asset-id robotio/multirobot_fold \
  --instruction "Fold the shirt in half."

# SmolVLA
STATION=Xarm7EEFStation POLICY=SmolVLACfg uv run -m examples.policy_inference \
  --policy-path ckpts/smolvla \
  --instruction "Place the cup on the shelf."
```

Press **Enter** when prompted to start the loop.

## Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-path` | from config | Checkpoint directory |
| `--asset-id` (pi0.5) | from config | Norm-stats asset id |
| `--instruction` | from config | Language instruction |
| `--freq` | `50` | Control frequency (Hz) |
| `--visualizer` | `None` | Set to `Rerun` for live visualization |
| `--policy-node-cfg.chunk-size` | `16` | Actions per chunk |
| `--policy-node-cfg.chunk-request-threshold` | `0.1` | Fraction consumed before next chunk request |

## Inference loop

1. Collect an observation (camera images + proprioception)
2. Send it to the policy server
3. Receive an action chunk and step through it
4. Request a new chunk before the current one is exhausted

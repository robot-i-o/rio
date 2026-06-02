# Data Conversion

Convert collected `.vla` trajectories to [LeRobot](https://github.com/huggingface/lerobot) / DROID format for use with the openpi training pipeline.

## DROID Schema

The exporter targets the DROID LeRobot schema:

- Three named cameras: `wrist_image_left`, `exterior_image_1_left`, `exterior_image_2_left`
- `joint_position` — first `num_joints` dims of `observation/proprio_joints`
- `actions` — joint velocities (`num_joints`) concatenated with a zeroed gripper channel

## Configuration

All fields can be overridden at the command line with tyro.

**`DatasetCfg` fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `image_height` | `180` | Output image height in pixels |
| `image_width` | `320` | Output image width in pixels |
| `fps` | `50` | Dataset frames per second |
| `robot_type` | `"panda"` | Robot type label in LeRobot metadata |
| `repo_id` | `"rio"` | HuggingFace repo ID for the dataset |
| `num_joints` | `None` | Number of joints; inferred from data if `None` |
| `action_dim` | `None` | Action dimensionality; inferred if `None` |
| `camera_mapping` | `None` | Dict mapping camera keys → DROID names; inferred if `None` |

**Script-specific `Args` fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `input` | `"/tmp/dummy_data/"` | Path to a `.vla` file or directory |
| `output` | `None` | Output directory (defaults to `~/.cache/huggingface/lerobot/{repo_id}`) |
| `robot_type` | `"xarm"` | Overrides `DatasetCfg` default for xarm datasets |
| `verbose` | `False` | Enable verbose logging |
| `clean` | `False` | Delete output directory before converting |

## Auto-Inference

When `num_joints`, `action_dim`, or `camera_mapping` are left as `None`, the script infers them from the first trajectory:

- `num_joints` — last dimension of `observation/proprio_joints`
- `action_dim` — `num_joints` + gripper dimensions
- `camera_mapping` — assigns DROID names by ascending resolution (smallest → `wrist_image_left`)

## Convert to DROID

**Minimal — auto-infer everything:**

```bash
uv run examples/data/convert_to_lerobot_droid.py --input /data/rollouts/my_task/
```

Output lands in `~/.cache/huggingface/lerobot/rio` by default.

**Specify output path and repo ID:**

```bash
uv run examples/data/convert_to_lerobot_droid.py \
    --input /data/rollouts/my_task/ \
    --output /data/lerobot/my_task/ \
    --repo-id myuser/my_task
```

**Re-run from scratch (wipe existing output):**

```bash
uv run examples/data/convert_to_lerobot_droid.py \
    --input /data/rollouts/my_task/ \
    --output /data/lerobot/my_task/ \
    --clean
```

**Override inferred fields:**

```bash
uv run examples/data/convert_to_lerobot_droid.py \
    --input /data/my_task \
    --num-joints 7 \
    --action-dim 8
```

## Compute Normalization Statistics

After conversion, compute normalization stats required by the openpi training pipeline:

```bash
cd third_party/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune_test
```


# Rerun Visualization

Visualize a saved trajectory in Rerun without running any hardware.

```bash
uv run -m examples.replay_data --loader_cfg.path /data/rollouts/my_task/traj_0001.vla
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loader_cfg.path` | (required) | Path to a `.vla` trajectory file |
| `--freq` | `50` | Replay frequency (Hz) |


## Robot model logging
This visualizer use the mujoco menangerie for robot model logging. If you wish to use a custom cache directory. Else specify a mujoco menagerie string

```bash
export URDF_PATH=/path/to/robot_descriptions/cache
```

*Note:* For using over SSH simply make sure to forward the ReRun address, `ssh -R 9876:localhost:9876 <your_ssh_addrs>`
# Collect Demonstrations

Use a teleoperation example to collect demonstrations. The recorder saves trajectories automatically when configured.

## Verify Your Camera Setup

Before running teleoperation, stream your cameras to confirm they're working:

```bash
uv run -m examples.stream_cameras
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--freq` | `50` | Streaming frequency (Hz) |
| `--mw` | `Thread` | Middleware backend |

## Configure the Recorder

The recorder is already set up in each config file. Open `examples/cfg/xarm_eef.py` (or whichever config you're using) and update the path:

```python
# examples/cfg/xarm_eef.py

TASK = "pick_and_place"  # ← trajectories save to data/{TASK}/

recorder: str | None = "Recorder"
recorder_cfg: RecorderCfg = field(
    default_factory=lambda: RecorderCfg(path=f"data/{TASK}/")
)
```

Set `recorder = None` to disable recording during testing:

```python
recorder: str | None = None
```

## Run Teleoperation for Data Collection

Select your station by setting `STATION=<ClassName>` before running. Run `uv run -m examples.cfg` to list all registered stations. Override any config field at the command line via tyro.

=== "XArm7 + Spacemouse / Gamepad / Keyboard"

    Control the robot in Cartesian space (EEF delta):

    ```bash
    STATION=Xarm7EEFStation uv run -m examples.teleop_eef

    # override device:
    STATION=Xarm7EEFStation uv run -m examples.teleop_eef --teleop "Gamepad"
    STATION=Xarm7EEFStation uv run -m examples.teleop_eef --teleop "Keyboard"

    # language instructions
    STATION=Xarm7EEFStation uv run -m examples.teleop_eef --instruction "pick up can"

    # data visualization while collecting
    STATION=Xarm7EEFStation uv run -m examples.teleop_eef --visualizer "Rerun"
    ```

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `--teleop` | `Spacemouse` | Teleop device (`Spacemouse`, `Gamepad`, `Keyboard`) |
    | `--freq` | `50` | Control frequency (Hz) |
    | `--instruction` | `""` | Language instruction recorded with the trajectory |
    | `--visualizer` | `None` | Visualizer to use (e.g. `Rerun`) |

    See [Gamepad](../hardware/teleop/gamepad.md) for setup instructions.

=== "XArm7 + Gello"

    Mirror joint positions using a Gello leader arm. On startup the script checks that the Gello joints are aligned with the robot — fix any misaligned joints before confirming, then wait ~3.5 s for the arm to settle before control begins.

    ```bash
    STATION=Xarm7GelloStation uv run -m examples.teleop_leader_follower

    # load Gello config from YAML:
    STATION=Xarm7GelloStation uv run -m examples.teleop_leader_follower --teleop_cfg_yaml path/to/gello.yaml

    # bimanual setup:
    STATION=Xarm7GelloStation uv run -m examples.teleop_leader_follower --teleop2 "Gello" --teleop2_cfg_yaml path/to/gello2.yaml

    # language instructions
    STATION=Xarm7GelloStation uv run -m examples.teleop_leader_follower --instruction "pick up can"
    ```

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `--teleop_cfg_yaml` | `None` | Path to Gello YAML config for the primary arm |
    | `--teleop2` | `None` | Second Gello device class (bimanual) |
    | `--teleop2_cfg_yaml` | `None` | Path to Gello YAML config for the second arm |
    | `--freq` | `15` | Control frequency (Hz) |
    | `--instruction` | `""` | Language instruction recorded with the trajectory |

    See [Gello](../hardware/teleop/gello.md) for setup instructions.

=== "SO100 Leader-Follower"

    Leader-follower joint mirroring over serial. Before running, update the calibration paths and serial ports in `examples/cfg/so100.py`:

    ```python
    # follower arm
    calibration_file="/path/to/follower.json"
    port="/dev/ttyACM0"

    # leader arm (teleop)
    calibration_file="/path/to/leader.json"
    port="/dev/ttyACM1"
    ```

    ```bash
    STATION=SO100Station uv run -m examples.teleop_leader_follower
    ```

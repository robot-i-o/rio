# Quick Start

Rio is built around one idea: **separate the hardware description from the control logic**.
A *station* declares every node in your setup; a *script* drives any station without modification.

To see all available components:

```bash
uv run rio-list-stations   
uv run rio-list-robots     
uv run rio-list-cameras     
uv run rio-list-interfaces 
```

---

## How it works

A **station** is a Python dataclass in `examples/cfg/` that declares every hardware node:

- **arm / gripper** — robot driver and gripper
- **cameras** — named dict of camera streams
- **teleop** — input device (motion controller, leader arm, or keyboard)
- **recorder** — saves demonstrations as `.vla` files

The **teleoperation scripts** (`teleop_eef`, `teleop_leader_follower`) read the station config, wire up all components, and run the control loop. The scripts are platform-agnostic — swap the station to use different hardware without touching the script.

---

## 1. Configure a station

Pick the config in `examples/cfg/` closest to your hardware and update the hardware-specific fields: robot IP or serial port, camera serial numbers, and recording path.

!!! note
    The example scripts select the station via the `STATION` environment variable (e.g. `STATION=Xarm7EEFStation`). This resolves the class name from `examples/cfg/__init__.py` at startup.

See [Station Configuration](workflow/station_cfg.md) for the full field reference and how to compose your own station from scratch.

---

## 2. Pick a control mode and run

Rio has two teleoperation paradigms. Choose the one that matches your teleop device.

### EEF / Cartesian control

Use this when your input device outputs **motion deltas** (Spacemouse, Gamepad, Keyboard).
The script drives the robot in end-effector space.

```bash
STATION=Xarm7EEFStation uv run -m examples.teleop_eef
```

Switch the input device without editing the config:

```bash
STATION=Xarm7EEFStation uv run -m examples.teleop_eef --teleop Gamepad
STATION=Xarm7EEFStation uv run -m examples.teleop_eef --teleop Keyboard
```

**Teleop modes** (switchable at runtime via a device button):

| Mode | Axes active |
|------|-------------|
| `TRANSLATION` | XYZ only (default) |
| `TRANSLATION_ROTATION` | XYZ + RPY |
| `TRANSLATION_2D` | XY only |
| `ROTATION` | RPY only |

### Leader-follower / joint mirroring

Use this when your input device is itself a **robot arm** that the follower mirrors joint-by-joint (Gello, SO100 leader).

```bash
STATION=Xarm7GelloStation uv run -m examples.teleop_leader_follower
STATION=SO100Station      uv run -m examples.teleop_leader_follower
```

On startup the script checks that the leader joints are aligned with the follower — fix any misaligned joints before confirming.

---

## Example stations at a glance

| Station class | Config file | Control mode | Script |
|---------------|-------------|--------------|--------|
| `Xarm7EEFStation` | `examples/cfg/xarm_eef.py` | EEF (Spacemouse) | `teleop_eef` |
| `Xarm7GelloStation` | `examples/cfg/xarm_gello.py` | Leader-follower (Gello) | `teleop_leader_follower` |
| `SO100Station` | `examples/cfg/so100.py` | Leader-follower (SO100) | `teleop_leader_follower` |
| `BimanualSO100Station` | `examples/cfg/bimanual_so100.py` | Leader-follower bimanual | `teleop_leader_follower` |
| `G1Station` | `examples/cfg/humanoid.py` | Humanoid whole-body (XRobotoolkit) | `teleop_humanoid` |

---

## Adding your own station

1. Copy the closest config in `examples/cfg/` and rename the class.
2. Update hardware fields (IP, ports, serials, calibration paths).
3. Register it in `examples/cfg/__init__.py` — add the import and class name to `__all__`.
4. Run with the appropriate script:

```bash
STATION=MyRobotStation uv run -m examples.teleop_eef             # EEF control
STATION=MyRobotStation uv run -m examples.teleop_leader_follower  # joint mirroring
```

---

## Common CLI overrides

Any config field can be overridden on the command line without editing the file:

```bash
STATION=Xarm7EEFStation uv run -m examples.teleop_eef \
    --instruction "pick up the can" \
    --visualizer Rerun \
    --freq 30
```

Recordings are saved as `.vla` files to `recorder_cfg.path`. See [Collect Demonstrations](workflow/data-collection.md) for replay, conversion, and next steps.

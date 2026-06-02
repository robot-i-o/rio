# Station Configuration

A station config is a dataclass that defines all hardware and software components for your robot setup. We provide factory functions that dynamically instantiate the servers/clients from this dataclass.

## How the examples are organized

| Folder | Purpose | Edit? |
|--------|---------|-------|
| `examples/cfg/` | One config per robot × teleop combo — arm IP, gripper port, camera serials, teleop settings, recorder path | **Yes** — edit before running |
| `examples/teleop_*.py` | Control loops (spacemouse EEF, gello joint, leader-follower) | No |

### What to edit before running

Open the config for your setup (e.g. `examples/cfg/xarm_eef.py`, `examples/cfg/so100.py`) and update:

**1. Your hardware:**

```python
arm_cfg: NodeCfg | None = field(
    default_factory=lambda: NodeCfg(
        robot_ip="192.168.1.205",  # ← change to your robot's IP
        ...
    )
)
# Update serial numbers to match your cameras:
"camera_1": Camera(serial="821212062747", ...)
"camera_2": Camera(serial="218622273888", ...)
```

**2. Your task:**

```python
TASK = "pick_and_place"  # ← sets the recorder output path: data/{TASK}/

recorder_cfg: RecorderCfg = field(
    default_factory=lambda: RecorderCfg(path=f"data/{TASK}/")
)
```

You don't need to touch anything else to get started.

### Available configs

| Config file | Class | Teleop script |
|-------------|-------|---------------|
| `examples/cfg/xarm_eef.py` | `Xarm7EEFStation` | `examples.teleop_eef` |
| `examples/cfg/xarm_gello.py` | `Xarm7GelloStation` | `examples.teleop_leader_follower` |
| `examples/cfg/so100.py` | `SO100Station` | `examples.teleop_leader_follower` |
| `examples/cfg/bimanual_so100.py` | `BimanualSO100Station` | `examples.teleop_leader_follower` |
| `examples/cfg/humanoid.py` | `G1Station` | `examples.teleop_humanoid` |

!!! note "Humanoid extra setup required"
    `G1Station` requires additional hardware, policy models, and dependencies before running.
    See [Humanoid Control](../setup-tutorials/humanoid.md) for the full setup guide.

Run `uv run -m examples.cfg` to see all registered station classes.

### Adding a new robot

1. Copy the closest config in `examples/cfg/` and update the hardware and teleop fields.
2. Register it in `examples/cfg/__init__.py`: add the import and add the class name to `__all__`.
3. Run with:

```bash
STATION=MyRobotStation uv run -m examples.teleop_leader_follower
```

---

## Component Pattern

The factory uses a **field + field_cfg** naming convention; the **module** is 

| Field | Config Field | Module |
|-------|--------------|--------|
| `arm`, `arm1`, `arm2` | `arm_cfg`, `arm1_cfg`, `arm2_cfg` | `robots` |
| `gripper`, `gripper1` | `gripper_cfg`, `gripper1_cfg` | `robots` |
| `visualizer` | `visualizer_cfg` | `visualization` |
| `recorder` | `recorder_cfg` | `data` |
| `teleop_*` | `teleop_*_cfg` | `interfaces` |
| `policy` | `policy_cfg`, `policy_node_cfg` | `policies` |

**Rules:**

- Field value = class name as string (e.g., `"XarmArm"`, `"Rerun"`) or `None` to disable
- Config = dataclass with parameters passed to the class constructor

## Cameras
Cameras use a dict with a special `Camera` helper class:

```python
cameras: dict[str, Camera] = field(
    default_factory=lambda: {
        "camera_1": Camera(
            cam_type="Realsense",      # Class name in rio.cameras
            serial="821212062747",
            model="D400",
            enable_depth=False,
            resolution=(480, 640),
        ),
    }
)
```

## Minimal Example

```python
from dataclasses import dataclass, field

@dataclass
class MyStationCfg:
    @dataclass
    class ArmCfg:
        robot_ip: str = "192.168.1.100"
        robot_model: str = "xarm7"
        freq: int = 250

    @dataclass
    class VisualizerCfg:
        app_id: str = "my_demo"
        spawn: bool = True
        freq: int = 30

    class Camera:
        def __init__(self, cam_type: str, module: str = "cameras", **kwargs):
            self.cam_type = cam_type
            self.module = module
            self.cfg = kwargs

    # Nodes
    arm: str | None = "XarmArm"
    arm_cfg: ArmCfg = field(default_factory=ArmCfg)

    gripper: str | None = None  # Disabled
    
    cameras: dict[str, Camera] = field(default_factory=dict)

    visualizer: str | None = "Rerun"
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    # Middleware
    mw: str = "Thread"  # or "Process"

    instruction: str = "Pick up the object."
```

## Bimanual Setup

For bimanual robots, use numbered fields:

```python
@dataclass
class BimanualStationCfg:
    arm1: str | None = "SoArm"
    arm1_cfg: ArmCfg = field(default_factory=lambda: ArmCfg(port="/dev/ttyACM0"))
    
    arm2: str | None = "SoArm"
    arm2_cfg: ArmCfg = field(default_factory=lambda: ArmCfg(port="/dev/ttyACM1"))

    embodiment_type: str = "BIMANUAL"
```

## Adding a Policy

```python
@dataclass
class PolicyStationCfg(BaseStationCfg):
    @dataclass
    class PolicyInterfaceConfig:
        instruction: str | None = None
        proprio_dim: int = 12
        action_dim: int = 12
        chunk_size: int = 50
        freq: int = 100
        camera_keys: list[str] = field(default_factory=list)

    @dataclass
    class PolicyConfig:
        policy_path: str = "/path/to/checkpoint"
        device: str = "cuda:0"

    policy: str = "Pi0"
    policy_node_cfg: PolicyInterfaceConfig = field(default_factory=PolicyInterfaceConfig)
    policy_cfg: PolicyConfig = field(default_factory=PolicyConfig)
```

## How Factory Resolution Works

1. `instantiate_station_cfg(args)` iterates over all fields
2. For each string field `xyz`, it looks for `xyz_cfg`
3. Module is inferred from field name (see table above)
4. Server/client pair is created: `make_node(mw, module, node_class, cfg_dict)`

To override the module for a field, pass `xyz_module="custom_module"` to the factory.

from dataclasses import dataclass, field

from rio.cfg import NodeCfg
from rio.policies.gear_sonic import DEFAULT_ANGLES, KDS, KPS


@dataclass
class G1Station:
    """Station configuration for Unitree G1 humanoid robot."""

    humanoid: str = "UnitreeG1"
    humanoid_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            network_interface="enx9c69d33c49cc",
            motor_kp=KPS,
            motor_kd=KDS,
            freq=200,
        )
    )

    teleop: str | None = None
    teleop_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            enable_body_tracking=True,
            freq=100,
        )
    )

    controller: str = "GearSonic"
    controller_module: str = "policies"
    controller_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            robot_type="g1",
            freq=50,
        )
    )

    planner: str = "GearSonicPlanner"
    planner_module: str = "policies"
    planner_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            freq=50,
        )
    )

    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 200

    default_dof_pos: tuple = tuple(DEFAULT_ANGLES.tolist())
    duration: float = 3.0

    embodiment_type: str = "HUMANOID"
    action_space: str = "JOINT_POS"
    sim: bool = False

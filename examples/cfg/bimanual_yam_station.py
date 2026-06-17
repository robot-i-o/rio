from dataclasses import dataclass, field

from rio.cfg import NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "pick_and_place"


@dataclass
class BimanualYamStation:
    # Left follower YAM arm (receives joint commands)
    arm1: str = "YamArm"
    arm1_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            channel="can_follow_l",
            gripper_type="linear_4310",
            zero_gravity_mode=False,
            freq=50,
        )
    )

    # Right follower YAM arm (receives joint commands)
    arm2: str | None = "YamArm"
    arm2_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            channel="can_follow_r",
            gripper_type="linear_4310",
            zero_gravity_mode=False,
            freq=50,
        )
    )

    # YamArm has an integrated gripper (commanded via moveG), so no separate
    # gripper nodes are needed; the embodiment auto-resolves it.
    gripper1: str | None = None
    gripper2: str | None = None

    # Left leader YAM arm in zero-gravity mode (publishes state only, moved by hand)
    teleop: str = "YamArm"
    teleop_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            channel="can_lead_l",
            gripper_type="yam_teaching_handle",
            zero_gravity_mode=True,
            freq=50,
        )
    )

    # Right leader YAM arm in zero-gravity mode (publishes state only, moved by hand)
    teleop2: str | None = "YamArm"
    teleop2_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            channel="can_lead_r",
            gripper_type="yam_teaching_handle",
            zero_gravity_mode=True,
            freq=50,
        )
    )

    teleop_module: str = "robots"
    teleop2_module: str = "robots"

    arm_latency: float = 0.0
    gripper_latency: float = 0.1
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 50

    action_space: str = "JOINT_POS"
    embodiment_type: str = "BIMANUAL"

    instruction: str = "Pick up the plushie"
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = None
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))

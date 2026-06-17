from dataclasses import dataclass, field

from rio.cfg import Camera, NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "pick_and_place"


@dataclass
class YamStation:
    cameras: dict[str, Camera] = field(
        default_factory=lambda: {
            # Wrist camera: Realsense D405
            "wrist": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="352122273371",
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            )
        }
    )

    # Follower YAM arm (receives joint commands)
    arm: str = "YamArm"
    arm_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            channel="can_follow_l",
            gripper_type="linear_4310",
            zero_gravity_mode=False,
            freq=50,
        )
    )

    gripper: str | None = None  # YamArm has integrated_gripper

    # Leader YAM arm in zero-gravity mode (publishes state only, moved by hand)
    teleop: str = "YamArm"
    teleop_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            channel="can_lead_l",
            gripper_type="yam_teaching_handle",
            zero_gravity_mode=True,
            freq=50,
        )
    )

    teleop_module: str = "robots"

    arm_latency: float = 0.0
    gripper_latency: float = 0.1
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 50

    action_space: str = "JOINT_POS"
    embodiment_type: str = "SINGLE_ARM"

    instruction: str = "Pick up the plushie"
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))

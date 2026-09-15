from dataclasses import dataclass, field

from rio.cfg import Camera, NodeCfg, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "pick_and_place"


@dataclass
class BimanualYamStation:
    cameras: dict[str, Camera] = field(
        default_factory=lambda: {
            # Fixed overhead camera looking at the workspace.
            "overhead": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="346522060488", # you will need to replace this with your camera's serial number
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
            # Left wrist camera.
            "left": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="352122272365", # you will need to replace this with your camera's serial number
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
            # Right wrist camera.
            "right": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial="352122273371", # you will need to replace this with your camera's serial number
                model="D400",
                enable_depth=False,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
        }
    )

    # Left follower YAM arm (receives joint commands)
    arm1: str = "YamArm"
    arm1_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            channel="can_follow_l",
            gripper_type="linear_4310",
            zero_gravity_mode=False,
            freq=50,
            enable_auto_recovery=True,
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
            enable_auto_recovery=True,
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
            enable_auto_recovery=True,
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
            enable_auto_recovery=True,
        )
    )

    teleop_module: str = "robots"
    teleop2_module: str = "robots"

    # USB button pad plugged into the robot host. Read via the Linux input layer, so it
    # works over SSH with no display and no terminal focus. Matched by USB id so the
    # device node number does not matter; `device_path` can pin a specific unit when
    # more than one identical pad is attached.
    buttons: str | None = "EvdevKeyboard"
    buttons_module: str = "interfaces"
    buttons_cfg: NodeCfg | None = field(
        default_factory=lambda: NodeCfg(
            vendor_id=0x3553,  # PCsensor MK321U 3-button pad
            product_id=0xC011,
            grab=True,  # keep presses out of the operator's shell
            freq=100,
            # If the pad is not plugged in, give up quickly and run without it rather
            # than stalling startup on the default readiness timeout.
            timeout=2.0,
        )
    )

    # Physical button to operator command, during teleoperated data collection.
    button_bindings: dict[str, str] = field(
        default_factory=lambda: {
            "a": "TOGGLE_EPISODE",
            "c": "DISCARD_LAST",
        }
    )

    # Bindings during policy inference. Kept separate because a policy config subclasses
    # this station config, so a single shared field would apply the collection bindings
    # to inference, where those commands mean nothing.
    inference_button_bindings: dict[str, str] = field(
        default_factory=lambda: {
            "a": "TOGGLE_INFERENCE",
            "c": "GO_HOME",
        }
    )

    # Home pose for GO_HOME
    home_pose: list[float] | None = field(default_factory=lambda: [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0])
    home_duration: float = 5.0  # seconds; raise for a gentler return

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
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/", start_recording=False))

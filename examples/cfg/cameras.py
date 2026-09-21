from dataclasses import dataclass, field

from rio.cfg import Camera, VisualizerCfg

OVERHEAD_CAMERA_SERIAL = "347322062075"
WRIST_CAMERA_SERIAL = "352122273371"


@dataclass
class CameraStation:
    """Camera-only station (no arm, gripper or teleop). Use with `examples.stream_cameras`."""

    cameras: dict[str, Camera] = field(
        default_factory=lambda: {
            "overhead": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial=OVERHEAD_CAMERA_SERIAL,
                model="D400",
                enable_depth=True,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
            "wrist": Camera(
                addr="127.0.0.1:5130",
                cam_type="Realsense",
                serial=WRIST_CAMERA_SERIAL,
                model="D400",
                enable_depth=True,
                resolution=(480, 640),
                resolution_depth=(480, 640),
            ),
        }
    )

    depth: bool = True

    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 50

    visualizer: str | None = "Rerun"
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

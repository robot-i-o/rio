try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy as SmolVLAModel  # noqa: F401
except ImportError as e:
    raise ImportError("Install lerobot with: uv sync --group pi") from e

from dataclasses import dataclass, field

import numpy as np

from examples import get_station_cfg
from rio.cfg.common import VisualizerCfg
from rio.schema import ActionSpace

_Base = get_station_cfg()


@dataclass
class SmolVLACfg(_Base):
    policy_path: str = "ckpts/smolvla"
    instruction: str | None = None

    def obs_transforms(obs, prompt=None):
        return {
            "observation/images": {k: obs[k] for k in obs if k.startswith("camera_")},
            "observation/state": obs["proprio_joints"],
            "task": prompt,
        }

    @dataclass
    class PolicyInterfaceConfig:
        instruction: str | None = None  # Instruction is overridden by global instruction if provided
        resolutions: list[tuple[int, int]] = None
        proprio_dim: int = 8
        action_dim: int = 7
        chunk_size: int = 16
        use_rtc: bool = False
        freq: int = 50
        max_buffer_size: int = 30
        chunk_request_threshold: float = 0.1  # request new chunk when this fraction of current chunk is consumed
        action_space: str | None = None
        required_action_space: str = "joint_pos"
        camera_keys: list[str] = field(default_factory=lambda: ["camera_1", "camera_2"])

        def __post_init__(self):
            if self.resolutions is None:
                self.resolutions = [(480, 640), (480, 640)]

    @dataclass
    class PolicyConfig:
        policy_path: str | None = None
        compile: bool = False
        device: str = "cuda:0"
        obs_transforms: callable = field(default_factory=lambda: SmolVLACfg.obs_transforms)
        dummy_obs: dict[str, np.ndarray] = field(
            default_factory=lambda: {
                "camera_1": np.zeros((480, 640, 3), dtype=np.uint8),
                "camera_2": np.zeros((480, 640, 3), dtype=np.uint8),
                "proprio_joints": np.zeros((7,), dtype=np.float32),
                "gripper_position": 0.0,
            }
        )

    policy: str = "SmolVLA"
    policy_node_cfg: PolicyInterfaceConfig = field(default_factory=lambda: SmolVLACfg.PolicyInterfaceConfig())
    policy_cfg: PolicyConfig = field(default_factory=lambda: SmolVLACfg.PolicyConfig())

    arm_latency: float = 0.0
    gripper_latency: float = 0.1

    mw: str = "Thread"
    mp_method: str | None = "spawn"
    freq: int = 50

    visualizer: str | None = "Rerun"
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    def __post_init__(self):
        if not self.policy_path:
            raise ValueError("policy_path is required — set SmolVLACfg.policy_path or pass --policy-path")

        self.policy_cfg.policy_path = self.policy_path
        self.policy_node_cfg.instruction = self.instruction
        self.policy_node_cfg.action_space = self.action_space

        if self.action_space.upper() not in ActionSpace.__members__:
            raise ValueError(f"Invalid action_space: {self.action_space}")

        if self.arm_cfg is not None:
            self.arm_cfg.cfg["robot_controller"] = self.action_space

        self.policy_node_cfg.camera_keys = list(self.cameras.keys())
        self.policy_node_cfg.resolutions = [cam.cfg["resolution"] for cam in self.cameras.values()]

        if len(self.policy_node_cfg.camera_keys) != len(self.policy_node_cfg.resolutions):
            raise ValueError("Length of camera_keys must match length of resolutions")

from dataclasses import dataclass, field

import numpy as np

from examples import get_station_cfg
from rio.cfg.common import VisualizerCfg
from rio.schema import ActionSpace

_Base = get_station_cfg()


@dataclass
class MolmoAct2Cfg(_Base):
    policy_path: str = "/data/ckpt/"
    norm_tag: str = ""
    instruction: str | None = "Pick the coke can and place it in the blue bowl."

    def obs_transforms(obs, prompt=None):
        joint_q = obs["proprio_joints"].astype(np.float32)
        gripper = np.atleast_1d(np.asarray(obs["gripper_position"], dtype=np.float32))

        transformed = dict(obs)
        transformed["proprio"] = np.concatenate([joint_q, gripper])
        return transformed

    @dataclass
    class PolicyInterfaceConfig:
        instruction: str | None = None
        resolutions: list[tuple[int, int]] = None
        proprio_dim: int = 8
        action_dim: int = 8
        chunk_size: int = 30
        use_rtc: bool = False
        freq: int = 50
        max_buffer_size: int = 30
        chunk_request_threshold: float = 0.1
        camera_keys: list[str] = field(default_factory=lambda: ["camera_1", "camera_2", "camera_3"])
        action_space: str | None = None
        required_action_space: str = "joint_pos"

        def __post_init__(self):
            if self.resolutions is None:
                self.resolutions = [(480, 640), (480, 640), (480, 640)]

    @dataclass
    class PolicyConfig:
        policy_path: str | None = None
        norm_tag: str = ""
        camera_keys: list[str] = field(default_factory=lambda: ["camera_1", "camera_2", "camera_3"])
        state_key: str = "proprio"
        device: str = "cuda:0"
        dtype: str = "float32"
        inference_action_mode: str = "continuous"
        action_tokenizer_path: str | None = None
        num_steps: int | None = None
        n_action_steps: int | None = None
        seq_len: int | None = None
        normalize_language: bool = True
        enable_cuda_graph: bool = True
        enable_depth_reasoning: bool = False
        warm_start_iters: int = 3
        obs_transforms: callable = field(default_factory=lambda: MolmoAct2Cfg.obs_transforms)
        dummy_obs: dict[str, np.ndarray] = field(
            default_factory=lambda: {
                "camera_1": np.zeros((480, 640, 3), dtype=np.uint8),
                "camera_2": np.zeros((480, 640, 3), dtype=np.uint8),
                "camera_3": np.zeros((480, 640, 3), dtype=np.uint8),
                "proprio_joints": np.zeros((7,), dtype=np.float32),
                "gripper_position": 0.0,
            }
        )

    policy: str = "MolmoAct2"
    policy_node_cfg: PolicyInterfaceConfig = field(default_factory=lambda: MolmoAct2Cfg.PolicyInterfaceConfig())
    policy_cfg: PolicyConfig = field(default_factory=lambda: MolmoAct2Cfg.PolicyConfig())

    arm_latency: float = 0.0
    gripper_latency: float = 0.1

    mw: str = "Thread"
    mp_method: str | None = "spawn"
    freq: int = 30

    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    def __post_init__(self):
        if not self.policy_path:
            raise ValueError("policy_path is required — set MolmoAct2Cfg.policy_path or pass --policy-path")
        if not self.norm_tag:
            raise ValueError("norm_tag is required — set MolmoAct2Cfg.norm_tag or pass --norm-tag")

        self.policy_cfg.policy_path = self.policy_path
        self.policy_cfg.norm_tag = self.norm_tag
        self.policy_node_cfg.instruction = self.instruction
        self.policy_node_cfg.action_space = self.action_space

        if self.action_space.upper() not in ActionSpace.__members__:
            raise ValueError(f"Invalid action_space: {self.action_space}")

        if self.arm_cfg is not None:
            self.arm_cfg.cfg["robot_controller"] = self.action_space

        self.policy_node_cfg.camera_keys = list(self.cameras.keys())
        self.policy_node_cfg.resolutions = [cam.cfg["resolution"] for cam in self.cameras.values()]
        self.policy_cfg.camera_keys = list(self.policy_node_cfg.camera_keys)

        if len(self.policy_node_cfg.camera_keys) != len(self.policy_node_cfg.resolutions):
            raise ValueError("Length of camera_keys must match length of resolutions")

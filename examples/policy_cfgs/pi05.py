try:
    import openpi.models.pi0_config as pi0_config
    import openpi.training.weight_loaders as weight_loaders
    from openpi.training.config import AssetsConfig, DataConfig, LeRobotDROIDDataConfig, TrainConfig
except ImportError as e:
    raise ImportError("Install openpi with: uv sync --group pi") from e


import dataclasses
from dataclasses import dataclass, field

import numpy as np

from examples import get_station_cfg
from rio.cfg.common import VisualizerCfg
from rio.schema import ActionSpace

_Base = get_station_cfg()


@dataclass
class Pi05Cfg(_Base):
    policy_path: str = "/data/ckpt/"
    asset_id: str = "robotio/pnp"
    instruction: str | None = "Pick the coke can and place it in the blue bowl."

    def obs_transforms(obs, prompt=None):
        wrist = obs["camera_2"]
        left = obs["camera_1"]
        right = obs["camera_3"]

        joint_q = obs["proprio_joints"].astype(np.float64)
        gripper = obs["gripper_position"]

        return {
            "observation/exterior_image_1_left": left,
            "observation/exterior_image_2_left": right,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joint_q,
            "observation/gripper_position": gripper,
            "prompt": prompt,
        }

    @dataclass
    class PolicyInterfaceConfig:
        instruction: str | None = None
        resolutions: list[tuple[int, int]] = None
        proprio_dim: int = 8
        action_dim: int = 8
        chunk_size: int = 16
        use_rtc: bool = False
        freq: int = 200
        max_buffer_size: int = 30
        chunk_request_threshold: float = 0.1
        action_space: str | None = None
        required_action_space: str = "joint_vel"
        camera_keys: list[str] = field(default_factory=lambda: ["camera_1", "camera_2", "camera_3"])

        def __post_init__(self):
            if self.resolutions is None:
                self.resolutions = [(480, 640), (480, 640), (480, 640)]

    @dataclass
    class PolicyConfig:
        policy_path: str | None = None
        train_config: str | TrainConfig = field(
            default_factory=lambda: TrainConfig(
                name="pi05_droid_finetune_xarm",
                exp_name="pi05_droid_finetune_xarm_exp",
                model=pi0_config.Pi0Config(
                    pi05=True,
                    action_dim=32,
                    action_horizon=16,
                ),
                data=LeRobotDROIDDataConfig(
                    repo_id="your_hf_username/droid_xarm",
                    base_config=DataConfig(prompt_from_task=True),
                    assets=AssetsConfig(),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
                num_train_steps=20_000,
                batch_size=32,
            )
        )
        compile: bool = False
        device: str = "cuda:0"
        obs_transforms: callable = field(default_factory=lambda: Pi05Cfg.obs_transforms)
        dummy_obs: dict[str, np.ndarray] = field(
            default_factory=lambda: {
                "camera_1": np.zeros((224, 224, 3), dtype=np.uint8),
                "camera_2": np.zeros((224, 224, 3), dtype=np.uint8),
                "camera_3": np.zeros((224, 224, 3), dtype=np.uint8),
                "proprio_joints": np.zeros((7,), dtype=np.float32),
                "gripper_position": 0.0,
            }
        )

    policy: str = "Pi0"
    policy_node_cfg: PolicyInterfaceConfig = field(default_factory=lambda: Pi05Cfg.PolicyInterfaceConfig())
    policy_cfg: PolicyConfig = field(default_factory=lambda: Pi05Cfg.PolicyConfig())

    arm_latency: float = 0.0
    gripper_latency: float = 0.1

    mw: str = "Thread"
    mp_method: str | None = "spawn"
    freq: int = 50

    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    def __post_init__(self):
        if not self.policy_path:
            raise ValueError("policy_path is required — set Pi05Cfg.policy_path or pass --policy-path")
        if not self.asset_id:
            raise ValueError("asset_id is required — set Pi05Cfg.asset_id or pass --asset-id")

        self.policy_cfg.policy_path = self.policy_path
        data = self.policy_cfg.train_config.data
        self.policy_cfg.train_config = dataclasses.replace(
            self.policy_cfg.train_config,
            data=dataclasses.replace(data, assets=dataclasses.replace(data.assets, asset_id=self.asset_id)),
        )
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

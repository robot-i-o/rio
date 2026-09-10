"""MolmoAct2 on the `allenai/MolmoAct2-BimanualYAM` checkpoint.

The checkpoint takes three views -- overhead, left, right -- and fourteen numbers of
absolute joint state, and returns thirty steps of the same fourteen: six joints and one
gripper travel per arm, left before right. Its views are positional rather than named, so
`camera_order` decides which station camera lands in which slot; getting it wrong gives the
model the wrong angle rather than an error.

    STATION=BimanualYamStation POLICY=MolmoAct2BimanualYamCfg python examples/policy_inference.py
"""

from dataclasses import dataclass, field

import numpy as np

from examples.policy_cfgs.molmoact2 import MolmoAct2Cfg
from rio.schema import ActionSpace

YAM_DUAL_DIM = 14
YAM_DUAL_HORIZON = 30
NORM_TAG = "yam_dual_molmoact2"


@dataclass
class MolmoAct2BimanualYamCfg(MolmoAct2Cfg):
    policy_path: str = "allenai/MolmoAct2-BimanualYAM"
    norm_tag: str = NORM_TAG
    instruction: str | None = "pick up the plushie and put it in the bin"

    # Station camera names in the checkpoint's own order.
    camera_order: list[str] = field(default_factory=lambda: ["overhead", "left", "right"])

    # Fraction of the way from the current pose to the chunk's target to command each tick.
    # The wrist cameras are inside the feedback path, so a full-rate command oscillates.
    action_alpha: float = 0.3

    def obs_transforms(obs, prompt=None):
        transformed = dict(obs)
        transformed["proprio"] = np.asarray(obs["proprio"], dtype=np.float32)
        return transformed

    @dataclass
    class PolicyInterfaceConfig(MolmoAct2Cfg.PolicyInterfaceConfig):
        proprio_dim: int = YAM_DUAL_DIM
        action_dim: int = YAM_DUAL_DIM
        chunk_size: int = YAM_DUAL_HORIZON

    @dataclass
    class PolicyConfig(MolmoAct2Cfg.PolicyConfig):
        norm_tag: str = NORM_TAG
        obs_transforms: callable = field(default_factory=lambda: MolmoAct2BimanualYamCfg.obs_transforms)
        dummy_obs: dict[str, np.ndarray] = field(
            default_factory=lambda: {
                "overhead": np.zeros((480, 640, 3), dtype=np.uint8),
                "left": np.zeros((480, 640, 3), dtype=np.uint8),
                "right": np.zeros((480, 640, 3), dtype=np.uint8),
                "proprio": np.zeros((YAM_DUAL_DIM,), dtype=np.float32),
            }
        )

    policy_node_cfg: PolicyInterfaceConfig = field(default_factory=lambda: MolmoAct2BimanualYamCfg.PolicyInterfaceConfig())
    policy_cfg: PolicyConfig = field(default_factory=lambda: MolmoAct2BimanualYamCfg.PolicyConfig())

    def __post_init__(self):
        if not self.policy_path:
            raise ValueError("policy_path is required — set MolmoAct2BimanualYamCfg.policy_path or pass --policy-path")
        if not self.norm_tag:
            raise ValueError("norm_tag is required — set MolmoAct2BimanualYamCfg.norm_tag or pass --norm-tag")

        self.policy_cfg.policy_path = self.policy_path
        self.policy_cfg.norm_tag = self.norm_tag
        self.policy_node_cfg.instruction = self.instruction
        self.policy_node_cfg.action_space = self.action_space

        if self.action_space.upper() not in ActionSpace.__members__:
            raise ValueError(f"Invalid action_space: {self.action_space}")

        for name in ["arm1_cfg", "arm2_cfg"]:
            arm_cfg = getattr(self, name, None)
            if arm_cfg is not None:
                arm_cfg.cfg["robot_controller"] = self.action_space

        missing = [name for name in self.camera_order if name not in self.cameras]
        if missing:
            raise ValueError(f"camera_order names {missing}, which the station does not have: {sorted(self.cameras)}")

        self.policy_node_cfg.camera_keys = list(self.camera_order)
        self.policy_node_cfg.resolutions = [self.cameras[name].cfg["resolution"] for name in self.camera_order]
        self.policy_cfg.camera_keys = list(self.camera_order)

        if len(self.policy_node_cfg.camera_keys) != len(self.policy_node_cfg.resolutions):
            raise ValueError("Length of camera_keys must match length of resolutions")

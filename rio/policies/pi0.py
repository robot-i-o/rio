# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from loguru import logger

try:
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config
    from openpi.training.config import LeRobotDROIDDataConfig, TrainConfig

    IMPORT_ERROR = None
except ImportError as e:
    policy_config = download = _config = LeRobotDROIDDataConfig = TrainConfig = None
    IMPORT_ERROR = e

from ._policy import Policy


class Pi0(Policy):
    def __init__(
        self,
        repo_id: str | None = None,
        train_config: str | TrainConfig = "pi0_fast_droid",
        policy_path: str = "gs://openpi-assets/checkpoints/pi0_fast_droid",
        libero: bool = False,
        obs_transforms: callable = lambda x: x,
        dummy_obs: dict[str, np.ndarray] | None = None,
        **kwargs,
    ):
        if IMPORT_ERROR is not None:
            raise ImportError("Pi0 dependencies not installed. Run: bash scripts/setup/vla/pi0_setup.sh") from IMPORT_ERROR

        self.ready = False
        self.config = _config.get_config(train_config) if isinstance(train_config, str) else train_config
        self.checkpoint_dir = download.maybe_download(policy_path)
        self.policy = None
        self.repo_id = repo_id
        self.instruction = None
        self.resolution = (224, 224)  # pi0/pi0.5 requires 224x224 images
        self.dummy_obs = dummy_obs

        self.obs_transforms = obs_transforms

        self.libero = libero

    def _warm_start(self, action_dim: int = 8):
        if self.dummy_obs is None:
            dummy_obs = {
                "camera_1": np.zeros((224, 224, 3), dtype=np.uint8),
                "camera_2": np.zeros((224, 224, 3), dtype=np.uint8),
                "camera_3": np.zeros((224, 224, 3), dtype=np.uint8),
                "proprio_joints": np.zeros((7,), dtype=np.float32),
                "gripper_position": 0.0,
            }
            logger.warning("Dummy obs not set.")
        else:
            dummy_obs = self.dummy_obs
        self.inference(dummy_obs)

    def construct_policy(self):
        self.policy = policy_config.create_trained_policy(self.config, self.checkpoint_dir)
        self._warm_start()
        self.ready = True

    def create_obs(env):
        formatted_obs = {}
        state = env.get_state()
        for key in state.observation.cameras:
            formatted_obs[key] = state.observation.cameras[key].rgb
        formatted_obs["proprio_joints"] = state.observation.proprio_joints
        formatted_obs["gripper_position"] = state.observation.gripper_position
        return formatted_obs

    def set_instruction(self, instruction):
        self.instruction = instruction

    def _process_observation(self, obs: dict):
        assert self.instruction is not None, "Instruction not set"

        if self.libero:
            raise NotImplementedError("Libero observation processing not implemented yet")

        if callable(self.obs_transforms):
            processed_obs = self.obs_transforms(obs, prompt=self.instruction)
        else:
            logger.warning("No processing function")
            processed_obs = obs

        # processed_obs["prompt"] = self.instruction
        return processed_obs

    def inference(self, observation, current_plan=None):
        assert self.policy is not None, "Policy.construct_policy not called!"

        obs = self._process_observation(observation)
        action_chunk = self.policy.infer(obs)["actions"]

        return action_chunk

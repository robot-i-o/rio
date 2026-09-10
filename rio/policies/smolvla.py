# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from loguru import logger

try:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_ERROR = e


from ._policy import Policy


class SmolVLA(Policy):
    def __init__(
        self,
        policy_path: str | Path = "lerobot/smolvla_base",
        libero=False,
        compile=False,
        device="cuda:0",
    ):
        if IMPORT_ERROR is not None:
            raise ImportError(
                "SmolVLA dependencies not installed. Run: bash scripts/setup/vla/smolvla_setup.sh"
            ) from IMPORT_ERROR
        self.compile = compile
        self.device = device
        self.policy_path = policy_path
        self.policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        self.policy_cfg.device = device

        self.policy = None
        self.instruction = None

        self.libero = libero

    def construct_policy(self):
        self.policy = SmolVLAPolicy.from_pretrained(self.policy_path, config=self.policy_cfg)
        if self.compile:
            self.policy.model.sample_actions = torch.compile(self.policy.model.sample_actions)

        # data is zeroed when processors are serialized
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=self.policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

        print(self.preprocessor.steps[-1])

        if self.libero:
            self.libero_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    # TODO: Need to verify this works for this policy since it was copied from policy_inference.py during abstraction
    def create_obs(env):
        logger.warning("The create_obs function has not yet been verified for smolvla, it may require changes")
        formatted_obs = {}
        state = env.get_state()
        for key in state.observation.cameras:
            formatted_obs[key] = state.observation.cameras[key].rgb
        formatted_obs["proprio_joints"] = state.observation.proprio_joints
        formatted_obs["gripper_position"] = state.observation.gripper_position
        return formatted_obs

    def set_instruction(self, instruction):
        self.instruction = instruction

    def _process_observation(self, obs):
        assert self.instruction is not None, "Instruction not set"
        torch_obs = {key: torch.tensor(obs[key]).to(self.device) for key in obs}

        batch = {
            "observation.state": torch_obs["proprio"],
            "observation.images.camera1": torch_obs["camera1"].permute(2, 0, 1),  # (H,W,C) -> (C,H,W)
            "task": [self.instruction],
        }

        # optional cameras
        if "camera2" in torch_obs:
            batch["observation.images.camera2"] = torch_obs["camera2"].permute(2, 0, 1)

        if "camera3" in torch_obs:
            batch["observation.images.camera3"] = torch_obs["camera3"].permute(2, 0, 1)

        # smolvla_libero has image, image2 keys instead of camera1, camera2
        if self.libero:
            batch["observation.state"] = batch["observation.state"].unsqueeze(0)
            batch["observation.images.image"] = batch["observation.images.camera1"].unsqueeze(0).type(torch.float32) / 255
            batch["observation.images.image2"] = batch["observation.images.camera2"].unsqueeze(0).type(torch.float32) / 255
            del batch["observation.images.camera1"]
            del batch["observation.images.camera2"]
            batch = self.libero_preprocessor(batch)

        processed_batch = self.preprocessor(batch)

        return processed_batch

    def inference(self, observation, current_plan=None):
        assert self.policy is not None, "Policy.construct_policy not called!"
        batch = self._process_observation(observation)

        actions = self.policy.predict_action_chunk(batch)
        postprocessed_actions = self.postprocessor(actions)
        action_chunk = postprocessed_actions.squeeze(0)

        return action_chunk

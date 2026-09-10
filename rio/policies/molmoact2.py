# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from loguru import logger

from ._policy import Policy

# The MolmoAct2 stack is vendored rather than installed, so the vendored `olmo` and
# LeRobot are put on the path ahead of any installed copy, as serve_policy.py does.
_EXPERIMENTS = Path(__file__).resolve().parents[2] / "third_party" / "molmoact2" / "experiments"
for _candidate in (_EXPERIMENTS, _EXPERIMENTS / "lerobot" / "src"):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_ERROR = e


class MolmoAct2(Policy):
    def __init__(
        self,
        policy_path: str | None = None,
        norm_tag: str = "",
        camera_keys: list[str] | None = None,
        state_key: str = "proprio",
        device: str = "cuda:0",
        dtype: str = "float32",
        inference_action_mode: str = "continuous",
        action_tokenizer_path: str | None = None,
        num_steps: int | None = None,
        n_action_steps: int | None = None,
        seq_len: int | None = None,
        normalize_language: bool = True,
        enable_cuda_graph: bool = True,
        enable_depth_reasoning: bool = False,
        warm_start_iters: int = 0,
        obs_transforms: callable = lambda obs, prompt=None: obs,
        dummy_obs: dict[str, np.ndarray] | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        if IMPORT_ERROR is not None:
            raise ImportError(
                "MolmoAct2 dependencies not installed. Run: bash scripts/setup/vla/molmoact2_setup.sh"
            ) from IMPORT_ERROR

        self.inference_action_mode = str(inference_action_mode or "").strip().lower()
        if self.inference_action_mode not in ["continuous", "discrete"]:
            raise ValueError(f"Invalid inference_action_mode: {inference_action_mode}, expected continuous or discrete")
        if self.inference_action_mode == "discrete" and not action_tokenizer_path:
            raise ValueError("inference_action_mode='discrete' requires action_tokenizer_path")

        self.ready = False
        self.policy_path = str(policy_path or "")
        self.norm_tag = str(norm_tag or "")
        self.camera_keys = list(camera_keys) if camera_keys else ["camera_1", "camera_2", "camera_3"]
        self.state_key = state_key
        self.device = device
        self.dtype = dtype
        self.action_tokenizer_path = action_tokenizer_path
        self.num_steps = num_steps
        self.n_action_steps = n_action_steps
        self.seq_len = seq_len
        self.normalize_language = normalize_language
        self.enable_cuda_graph = enable_cuda_graph
        self.enable_depth_reasoning = enable_depth_reasoning
        self.warm_start_iters = warm_start_iters
        self.dummy_obs = dummy_obs
        self.verbose = verbose

        self.obs_transforms = obs_transforms

        self.model = None
        self.processor = None
        self.action_tokenizer = None
        self.policy = None  # LeRobot MolmoAct2Policy, native checkpoints only
        self.instruction = None

        # Read off the checkpoint in construct_policy rather than trusted from the config
        self.state_dim = None
        self.action_dim = None
        self.chunk_size = None

    def _warm_start(self):
        if self.warm_start_iters < 1:
            return
        if not self.dummy_obs:
            logger.warning("Dummy obs not set.")
            return

        instruction = self.instruction
        self.instruction = instruction or "warm start"
        try:
            for _ in range(self.warm_start_iters):
                self.inference(self.dummy_obs)
        finally:
            self.instruction = instruction

    def _is_native_checkpoint(self):
        # Native training checkpoints are a directory with the training config.yaml in it
        checkpoint_dir = Path(self.policy_path).expanduser()
        return checkpoint_dir.is_dir() and (checkpoint_dir / "config.yaml").exists()

    def _construct_native(self):
        try:
            from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config  # noqa: PLC0415
            from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                f"{self.policy_path} is a native training checkpoint, which is loaded through LeRobot "
                "and needs the MolmoAct2 training environment. Convert it for deployment with: "
                "python -m olmo.hf_model.convert_molmoact2_to_hf <checkpoint> <output_dir>"
            ) from e

        cfg = MolmoAct2Config(
            checkpoint_path=str(Path(self.policy_path).expanduser()),
            device=self.device,
            seq_len=self.seq_len,
            num_steps=self.num_steps,
            inference_action_mode=self.inference_action_mode,
            discrete_action_tokenizer=self.action_tokenizer_path,
            enable_depth_reasoning=self.enable_depth_reasoning,
            norm_tag=self.norm_tag,
            enable_inference_cuda_graph=self.enable_cuda_graph,
            verbose=self.verbose,
        )
        self.policy = MolmoAct2Policy(cfg)
        self.policy.eval()
        if not self.normalize_language:
            logger.warning("normalize_language is not plumbed through MolmoAct2Policy; ignoring it.")

        self.model = self.policy._handles.model
        if self.dtype != "float32":
            self.model.to(getattr(torch, self.dtype))

    def _construct_hf(self):
        checkpoint_dir = Path(self.policy_path).expanduser()
        source = str(checkpoint_dir) if checkpoint_dir.exists() else self.policy_path

        self.processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            trust_remote_code=True,
            dtype=getattr(torch, self.dtype),
            low_cpu_mem_usage=True,
        )
        self.model = model.to(torch.device(self.device))
        self.model.eval()

        if self.inference_action_mode == "discrete":
            self.action_tokenizer = AutoProcessor.from_pretrained(self.action_tokenizer_path, trust_remote_code=True)

    def _read_checkpoint_shapes(self):
        # max_action_dim in the checkpoint config is the padded width the action expert emits;
        # the norm tag's stats are the only record of how many of those columns mean anything
        stats = self.policy._handles.robot_processor if self.policy is not None else self.model._get_robot_stats()
        if stats is None:
            raise ValueError(f"{self.policy_path} carries no normalization stats")

        tags = sorted(getattr(stats, "metadata_by_tag", {}) or {})
        if self.norm_tag not in tags:
            raise ValueError(f"Invalid norm_tag: {self.norm_tag}, this checkpoint has {tags}")

        self.state_dim = stats.get_state_dim(self.norm_tag)
        self.action_dim = stats.get_action_dim(self.norm_tag)
        self.chunk_size = self.n_action_steps or stats.get_n_action_steps(self.norm_tag)
        logger.info(f"MolmoAct2 {self.norm_tag}: state {self.state_dim}, action {self.action_dim}, chunk {self.chunk_size}")

    def construct_policy(self):
        if not self.policy_path:
            raise ValueError("policy_path is required to construct MolmoAct2")

        if self._is_native_checkpoint():
            self._construct_native()
        else:
            self._construct_hf()

        self._read_checkpoint_shapes()
        self._warm_start()
        self.ready = True

    def create_obs(env):
        formatted_obs = {}
        state = env.get_state()
        for key in state.observation.cameras:
            formatted_obs[key] = state.observation.cameras[key].rgb
        formatted_obs["proprio"] = np.asarray(state.observation.proprio, dtype=np.float32)
        return formatted_obs

    def set_instruction(self, instruction):
        self.instruction = instruction

    def reset(self):
        if self.policy is not None:
            self.policy.reset()

    def _convert_image(self, frame):
        # The policy node's shared-memory schema widens uint8 frames to float32. Those values
        # are still 0-255, so only frames actually normalized to 0-1 are rescaled.
        if isinstance(frame, Image.Image):
            return np.asarray(frame.convert("RGB"))
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)

        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected an HxWx3 RGB frame, got shape {frame.shape}")
        if frame.dtype == np.uint8:
            return frame
        if np.issubdtype(frame.dtype, np.floating) and frame.size and float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _process_observation(self, obs: dict):
        assert self.instruction is not None, "Instruction not set"

        if callable(self.obs_transforms):
            processed_obs = self.obs_transforms(obs, prompt=self.instruction)
        else:
            logger.warning("No processing function")
            processed_obs = obs

        missing = [key for key in [*self.camera_keys, self.state_key] if key not in processed_obs]
        if missing:
            raise KeyError(f"Observation is missing {missing}, it has {sorted(processed_obs)}")

        images = [self._convert_image(processed_obs[key]) for key in self.camera_keys]

        state = np.asarray(processed_obs[self.state_key], dtype=np.float32).reshape(-1)
        if self.state_dim is not None and state.size != self.state_dim:
            raise ValueError(f"{self.norm_tag} expects a state of {self.state_dim} numbers, got {state.size}")

        return images, state

    def _autocast(self):
        if self.dtype == "float32":
            return nullcontext()
        return torch.autocast(device_type=torch.device(self.device).type, dtype=getattr(torch, self.dtype))

    def _infer_native(self, images, state, task):
        # _obs_to_example reads the views in dict order, so this insertion order is what
        # assigns them to the checkpoint's positional slots
        obs = {"task": task, "observation.state": state}
        for key, image in zip(self.camera_keys, images, strict=True):
            obs[f"observation.images.{key}"] = image

        # The policy node buffers chunks itself and re-observes between requests, so each
        # call is stateless
        self.policy.reset()
        actions, _tokens, _style = self.policy.generate_action_chunk_from_observations(
            [obs],
            norm_tag=self.norm_tag,
            num_steps=self.num_steps,
            n_action_steps=self.n_action_steps,
        )
        return actions

    def _infer_hf(self, images, state, task):
        return self.model.predict_action(
            processor=self.processor,
            images=images,
            task=task,
            state=state,
            norm_tag=self.norm_tag,
            inference_action_mode=self.inference_action_mode,
            enable_depth_reasoning=self.enable_depth_reasoning,
            action_tokenizer=self.action_tokenizer,
            num_steps=self.num_steps,
            n_action_steps=self.n_action_steps,
            normalize_language=self.normalize_language,
            enable_cuda_graph=self.enable_cuda_graph,
        ).actions

    def inference(self, observation, current_plan=None):
        assert self.model is not None, "Policy.construct_policy not called!"

        images, state = self._process_observation(observation)
        task = str(current_plan if current_plan is not None else self.instruction)

        with self._autocast():
            if self.policy is not None:
                action_chunk = self._infer_native(images, state, task)
            else:
                action_chunk = self._infer_hf(images, state, task)

        return action_chunk[0].detach().float().cpu().numpy()

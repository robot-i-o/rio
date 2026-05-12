# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory
from rio_hw.node import Node
from threadpoolctl import threadpool_limits

from ._policy import Policy


class PolicyInterface(Node):
    __api__ = [
        "send_observation",
        "get_action_chunk",
        "reset",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        policy: Policy,
        instruction: str,
        resolutions: list[tuple[int, int]],
        action_dim: int,
        proprio_dim: int,
        chunk_size: int,
        use_rtc: bool = False,
        freq: int = 100,
        max_buffer_size: int = 30,
        chunk_request_threshold: float = 0.75,
        camera_keys: list[str] | None = None,
    ):
        self.policy = policy
        self.instruction = instruction
        self.resolutions = resolutions
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.use_rtc = use_rtc
        self.freq = freq
        self.chunk_size = chunk_size
        self.max_buffer_size = max_buffer_size
        self.last_timestamp = time.now()
        self.chunk_request_threshold = chunk_request_threshold
        self.camera_keys = camera_keys

        super().__init__()

    # NOTE: Defines request/pub schemas in post_init following other constructors
    def __post_init__(self):
        if len(self.camera_keys) != len(self.resolutions):
            logger.error("Length of camera_keys must match length of resolutions")
            raise ValueError("Length of camera_keys must match length of resolutions")
        if self.camera_keys is None:
            logger.warning("camera_keys is None, defaulting to camera_1, camera_2, ...")
            self.camera_keys = [f"camera_{i + 1}" for i in range(len(self.resolutions))]

        obs_schema = {"proprio": np.zeros(shape=(self.proprio_dim,), dtype=np.float32)}
        for i, resolution in enumerate(self.resolutions):
            obs_schema[self.camera_keys[i]] = np.zeros(shape=(*resolution, 3), dtype=np.float32)

        action_schema = {"actions": np.zeros(shape=(self.chunk_size, self.action_dim), dtype=np.float32)}

        self.example_request = {**obs_schema}
        self.example_data = {**action_schema, "timestamp": time.now(), "ready": False, "policy_loaded": False}

        self.worker = None
        self.run = self.pubreq
        super().__post_init__()

    def pubreq(self):
        """
        Handles policy inference requests and pushes action chunks onto the ring buffer
        """
        threadpool_limits(1)

        # initialize model on device
        self.policy.set_instruction(self.instruction)
        self.policy.construct_policy()
        logger.info("PolicyInterface: Policy constructed.")
        self.ring_buffer.put(
            {
                "actions": np.zeros((self.chunk_size, self.action_dim), dtype=np.float32),
                "timestamp": time.now(),
                "ready": False,
                "policy_loaded": True,
            },
            wait=False,
        )
        logger.debug("PolicyInterface: Policy constructed and instruction set.")
        try:
            # Main loop
            rate = time.Rate(self.freq)
            self.pub_ready_event.set()
            while not self.exit_event.is_set():
                # Pop from request queue
                if not self.request_queue.empty():
                    obs = self.request_queue.get()
                    receive_time = time.now()
                    # Inference
                    action_chunk = self.policy.inference(obs)
                    # Store action chunk in ring buffer
                    data = {"actions": action_chunk, "timestamp": receive_time, "ready": True, "policy_loaded": True}
                    self.ring_buffer.put(data, wait=False)

                rate.precise_sleep()
        except KeyboardInterrupt:
            pass

    def send_observation(self, raw_obs):
        self.request_queue.put(raw_obs)

    def get_action_chunk(self):
        """
        Retrieves an action chunk from the ring buffer if available.
        Otherwise, sends an empty array.
        """
        data = self.ring_buffer.get()
        # handle stale action chunks
        # TODO: is there a better way to mark stale action chunks?

        if not isinstance(data, dict) or data.get("timestamp") is None or data["timestamp"] <= self.last_timestamp:
            empty_data = {
                "actions": np.zeros((self.chunk_size, self.action_dim)),
                "timestamp": time.now(),
                "ready": False,
                "policy_loaded": False,
            }
            return empty_data
        else:
            self.last_timestamp = data["timestamp"]
            return data

    def get_action_chunk_blocking(self, obs):
        action_chunk = self.policy.inference(obs)
        data = {"actions": action_chunk}
        return data

    def reset(self):
        # TODO: addpt to use the middleware reset method
        """
        Resets the policy interface state, clearing buffers and resetting timestamps.
        """
        # Clear the request queue
        while not self.request_queue.empty():
            try:
                self.request_queue.clear()
            except Exception:
                break

        # Reset the policy if it has a reset method
        if hasattr(self.policy, "reset") and callable(self.policy.reset):
            self.policy.reset()
            logger.info("PolicyInterface: Policy reset.")

        # Clear ring buffer and put initial state
        self.ring_buffer.clear()

        logger.debug("PolicyInterface: Reset complete.")

    @property
    def policy_loaded(self):
        data = self.ring_buffer.get()
        if not isinstance(data, dict):
            return False
        policy_loaded = data.get("policy_loaded", False)
        return policy_loaded


def PolicyInterfaceServer(mw, *args, **kwargs):
    return ServerFactory(mw, PolicyInterface, *args, **kwargs)


def PolicyInterfaceClient(mw, *args, **kwargs):
    return ClientFactory(mw, PolicyInterface, *args, **kwargs)

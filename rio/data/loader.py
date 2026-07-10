# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Data loading module for robodm format replay."""

from collections import defaultdict
from importlib import import_module
from typing import Any

import robodm
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory
from rio_hw.node import Node

from ..embodiments.base import EmbodimentType
from ..schema import Camera, Step


def dict_to_step(data: dict[str, Any], embodiment_type: EmbodimentType = EmbodimentType.SINGLE_ARM) -> Step:
    """
    Convert a flat dictionary into a Step object.
    """
    module = import_module(f"rio.embodiments.{embodiment_type.name.lower()}")
    ObsClass = getattr(module, f"{embodiment_type.name.title().replace('_', '')}Obs")

    observation = {}
    cameras = defaultdict(dict)
    sensors = defaultdict(dict)
    for k, v in data.items():
        if k.startswith("observation/cameras/"):
            cam_name = k.split("/")[2]
            cam_input = k.split("/")[3]
            cameras[cam_name][cam_input] = v
        elif k.startswith("observation/sensors/"):
            parts = k.split("/")
            if len(parts) >= 4:
                sensor_name = parts[2]
                sensor_input = "/".join(parts[3:])
                sensors[sensor_name][sensor_input] = v
        elif k.startswith("observation/"):
            observation[k.split("/")[1]] = v
    observation["cameras"] = {name: Camera(**cam_data) for name, cam_data in cameras.items()}
    observation["sensors"] = {name: dict(sensor_data) for name, sensor_data in sensors.items()}

    observation = ObsClass(**observation)
    step = Step(
        timestep=data.get("timestep", 0),
        observation=observation,
        action=data.get("action"),
        instruction=data.get("instruction"),
        meta=data.get("meta", {}),
    )

    return step


class Loader(Node):
    __api__ = [
        "get_state",
        "get_all_state",
        "get_timestep",
        "get_num_timesteps",
        "get_features",
        "reset",
        "step",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        path: str,
        auto_play: bool = False,
        loop: bool = False,
        embodiment_type: str | EmbodimentType = EmbodimentType.SINGLE_ARM,
        *,
        freq: int = 10,
        max_buffer_size: int = 30,
        **kwargs,
    ):
        """
        Initialize the Loader node.

        Args:
            path: Path to the trajectory file to load (e.g., "/tmp/demo.vla")
            auto_play: If True, automatically iterate through timesteps at the specified freq
            loop: If True, loop back to the beginning when reaching the end
            freq: Playback frequency in Hz (if auto_play is True) or request processing frequency
            max_buffer_size: Maximum size for internal buffers
        """
        self.path = path
        self.auto_play = auto_play
        self.loop = loop

        if isinstance(embodiment_type, str):
            embodiment_type = EmbodimentType[embodiment_type.upper()]

        self.embodiment_type = embodiment_type

        super().__init__(freq=freq, max_buffer_size=max_buffer_size, **kwargs)

        # Configure logger level based on verbose flag
        if not self.verbose:
            logger.disable("rio.data.loader")

    def __post_init__(self):
        """Initialize the robodm Trajectory and set up the node."""
        # Load the trajectory
        self.datamanager = robodm.Trajectory(
            path=self.path,
            mode="r",
        )

        # Load all data into memory
        self.data = self.datamanager.load()
        self.features = list(self.data.keys())

        # Determine the number of timesteps (use the first feature)
        if self.features:
            first_feature = self.features[0]
            self.num_timesteps = len(self.data[first_feature])
        else:
            self.num_timesteps = 0

        # Current timestep index
        self.current_timestep = 0

        # Set example request and data
        self.example_request = {
            "type": "get_timestep",
            "timestep": None,
        }
        self.example_data = self._get_current_state()

        # This node processes requests and publishes current state
        self.worker = None
        self.run = self.pubreq

        super().__post_init__()

    def _get_current_state(self) -> dict[str, Any]:
        """Get the current timestep's data."""
        if self.num_timesteps == 0:
            return {"timestep": 0, "num_timesteps": 0, "features": []}

        state = {
            "timestep": self.current_timestep,
            "num_timesteps": self.num_timesteps,
            "features": self.features,
        }

        # Add data for each feature at the current timestep
        for feature in self.features:
            try:
                state[feature] = self.data[feature][self.current_timestep]
            except (IndexError, KeyError):
                pass

        return state

    def pubreq(self):
        """Combined publish and request processing loop."""
        try:
            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            self.pub_ready_event.set()

            logger.debug(f"Loader started: {self.num_timesteps} timesteps, {len(self.features)} features")
            logger.debug(f"Features: {self.features}")

            while not self.exit_event.is_set():
                # Process all pending requests
                while not self.request_queue.empty():
                    req = self.request_queue.get()
                    if req is None:
                        break

                    req_type = req.get("type")
                    if req_type == "step":
                        self._step()
                    elif req_type == "reset":
                        self._reset()
                    elif req_type == "get_timestep":
                        timestep = req.get("timestep")
                        if timestep is not None:
                            self._set_timestep(timestep)

                # Auto-play: advance to next timestep
                if self.auto_play:
                    self._step()

                # Publish current state
                current_state = self._get_current_state()
                self.ring_buffer.put(current_state, wait=False)

                rate.precise_sleep()

        except KeyboardInterrupt:
            pass

    def _step(self):
        """Advance to the next timestep."""
        if self.num_timesteps == 0:
            return

        self.current_timestep += 1

        if self.current_timestep >= self.num_timesteps:
            if self.loop:
                self.current_timestep = 0
                logger.debug("Looping back to beginning")
            else:
                self.current_timestep = self.num_timesteps - 1
                logger.debug("Reached end of trajectory")

    def _reset(self):
        """Reset to the first timestep."""
        self.current_timestep = 0
        logger.debug("Reset to timestep 0")

    def _set_timestep(self, timestep: int):
        """Set the current timestep to a specific value."""
        if 0 <= timestep < self.num_timesteps:
            self.current_timestep = timestep
        else:
            logger.warning(f"timestep {timestep} out of range [0, {self.num_timesteps})")

    def get_step(self):
        state = self.get_state()
        # convert to STEP object
        step = dict_to_step(state, self.embodiment_type)
        return step

    def get_state(self):
        """Get the current timestep's state."""
        return self.ring_buffer.get()

    def get_all_state(self):
        """Get all states in the ring buffer."""
        return self.ring_buffer.get_all()

    def step(self):
        """Request: Advance to the next timestep."""
        self.request_queue.put({"type": "step"})

    def reset(self):
        """Request: Reset to the first timestep."""
        self.request_queue.put({"type": "reset"})

    def get_timestep(self, timestep: int):
        """Request: Get data at a specific timestep."""
        self.request_queue.put({"type": "get_timestep", "timestep": timestep})

    def get_num_timesteps(self) -> int:
        """Get the total number of timesteps."""
        return self.num_timesteps

    def get_features(self) -> list[str]:
        """Get the list of available features."""
        return self.features


def LoaderServer(mw, *args, **kwargs):
    """Create a Loader server instance."""
    return ServerFactory(mw, Loader, *args, **kwargs)


def LoaderClient(mw, *args, **kwargs):
    """Create a Loader client instance."""
    return ClientFactory(mw, Loader, *args, **kwargs)

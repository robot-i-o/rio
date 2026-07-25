# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import numpy as np
from loguru import logger
from rio_hw import time

from ..schema import Camera, Step
from .factory import _is_sensor_field, get_mbody_components, instantiate_station_cfg


class Env:
    def __init__(self, mw, clients, embodiment_type, camera_clients=None, action_space="joint_vel", **kwargs):
        self.mw = mw
        self._clients = clients
        self._camera_clients: dict = camera_clients or {}
        self.action_space = action_space.upper()
        self.embodiment_type = embodiment_type
        self._exit_stack = None
        self.robot = None
        self.cameras: dict = {}
        self.sensors: dict = {}
        self.kwargs = kwargs
        self.instruction = kwargs.get("instruction", "")
        self.start_time = None
        if len(self.instruction) > 0:
            logger.info(f"Environment instruction: {self.instruction}")

        self.rgb_key = kwargs.get("rgb_key", "color")
        self.depth_key = kwargs.get("depth_key", "depth")

    def build_action(self, arm_cmd, gripper_cmd):
        if self.robot is None:
            raise RuntimeError("Robot embodiment is not initialized.")

        action = self.robot.build_action(arm_cmd, gripper_cmd)
        return action

    def get_client(self, key):
        if self._clients.get(key) is None:
            return None
        return self._exit_stack.enter_context(self._clients[key]())

    def set_start_time(self, start_time: float):
        logger.info(f"Setting environment start time to {start_time:.4f}")
        self.start_time = start_time

    def set_instruction(self, instruction: str):
        logger.info(f"Setting environment instruction to: {instruction}")
        self.instruction = instruction

    def start(self):
        self._exit_stack = ExitStack()

        # Embodiment Components
        components, EmbodimentClass = get_mbody_components(embodiment_type=self.embodiment_type, clients=self._clients)
        for comp_name, client_factory in components.items():
            components[comp_name] = self._exit_stack.enter_context(client_factory())

        self.robot = EmbodimentClass(
            **components,
            action_space=self.action_space,
            **self.kwargs,
        )

        # Data recorder
        self.recorder = self.get_client("recorder")

        # Cameras
        for key, client_factory in self._camera_clients.items():
            if client_factory is not None:
                self.cameras[key] = self._exit_stack.enter_context(client_factory())

        # Sensors
        for key, client_factory in self._clients.items():
            if key not in components and _is_sensor_field(key) and client_factory is not None:
                self.sensors[key] = self._exit_stack.enter_context(client_factory())

        # Visualizer
        self.visualizer = self.get_client("visualizer")

    def stop(self):
        if self._exit_stack:
            self._exit_stack.close()
            self._exit_stack = None
            self.robot = None
            self.cameras = {}
            self.sensors = {}
            self.visualizer = None

    def reset(self):
        return self.get_state()

    def get_state(self, action: np.ndarray | None = None, use_relative_time: bool = True) -> Step:
        cams = self.get_cameras()
        obs = self.robot.get_obs(cams)
        obs.sensors = self.get_sensors()

        if use_relative_time:
            _timestamp = time.now() - self.start_time
        else:
            _timestamp = time.now()

        return Step(
            timestep=_timestamp,
            instruction=self.instruction,
            observation=obs,
            action=action,
            meta={},
        )

    def step(
        self,
        action: np.ndarray,
        t_cmd_target: float | None,
    ):
        self.robot.move(
            action,
            t_cmd_target,
        )
        # action = self.build_action(arm_cmd, gripper_cmd)
        return self.get_state(action=action)

    def move(self, action: np.ndarray, t_cmd_target: float | None):
        self.robot.move(action, t_cmd_target)

    def get_cameras(self) -> dict[str, Camera]:
        obs = {}
        for key, camera in self.cameras.items():
            state = camera.get_state()
            meta = {
                "depth_units": state.get("depth_units", None),
            }
            obs[key] = Camera(
                rgb=state.get("color"),
                depth=state.get("depth"),
                meta=meta,
            )
        return obs

    def get_sensors(self) -> dict[str, dict]:
        obs = {}
        for key, sensor in self.sensors.items():
            state = sensor.get_state()
            if state is not None:
                obs[key] = state
        return obs

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def make_env(station_cfg, **kwargs):
    servers, clients, camera_clients = instantiate_station_cfg(station_cfg, **kwargs)
    action_space = getattr(station_cfg, "action_space", "EEF_POSE")
    urdf_path = getattr(station_cfg, "urdf_path", None)

    # Determine embodiment type from components
    embodiment_type = getattr(station_cfg, "embodiment_type", None)
    if embodiment_type is None:
        # Infer from available components - if we have arm2, it's dual-arm
        if "arm2" in clients and clients["arm2"] is not None:
            embodiment_type = "DUAL_ARM"
        else:
            embodiment_type = "SINGLE_ARM"

    logger.debug(f"URDF Path: {urdf_path}")
    logger.debug(f"Embodiment Type: {embodiment_type}")
    return (
        servers,
        clients,
        Env(
            mw=station_cfg.mw,
            clients=clients,
            camera_clients=camera_clients,
            embodiment_type=embodiment_type,
            action_space=action_space,
            urdf_path=urdf_path,
            **kwargs,
        ),
    )

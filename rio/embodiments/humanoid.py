# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
from loguru import logger
from rio_hw.middlewares._middleware import Client

from ..schema import ActionSpace, Observation
from .base import BaseEmbodiment


@dataclass
class HumanoidObs(Observation):
    q: np.ndarray | None = None
    dq: np.ndarray | None = None
    quaternion: np.ndarray | None = None
    gyroscope: np.ndarray | None = None


class Humanoid(BaseEmbodiment):
    def __init__(
        self,
        humanoid: Client,
        action_space: str = "JOINT_POS",
        **kwargs,
    ):
        self.humanoid = humanoid
        self.action_space = ActionSpace[action_space]
        self.urdf_path = kwargs.get("urdf_path", "")
        self.__obs_schema__ = HumanoidObs

        # Get number of joints dynamically
        self.num_joints = self._get_num_joints(self.humanoid)

    def _get_num_joints(self, humanoid: Client) -> int:
        if hasattr(humanoid, "num_joints"):
            return humanoid.num_joints
        else:
            logger.warning(
                "Humanoid client does not have 'num_joints' attribute; using state to determine joint count.  \
                    Consider adding 'num_joints' attribute to the client for efficiency."
            )
            # Try to determine from state
            state = humanoid.get_state()
            if state and "q" in state:
                return len(state["q"])
            else:
                raise ValueError("Cannot determine number of joints for the given humanoid client.")

    def move(
        self,
        action: np.ndarray,
        t_cmd_target: float,
        **kwargs,
    ):
        if self.action_space == ActionSpace.JOINT_POS:
            self.humanoid.moveJ(action.tolist(), t_cmd_target)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not implemented for Humanoid")

    def parse_action(self, action: np.ndarray) -> dict:
        return {"q": action}

    def build_action(self, arm_cmd: np.ndarray) -> np.ndarray:
        return arm_cmd

    def get_state(self):
        return self.humanoid.get_state()

    def get_obs(self, cams: dict) -> HumanoidObs:
        """Build humanoid observation."""
        state = self.humanoid.get_state()

        q = state.get("q", None)
        dq = state.get("dq", None)
        quaternion = state.get("quaternion", None)
        gyroscope = state.get("gyroscope", None)

        obs = HumanoidObs(
            proprio=q,
            q=q,
            dq=dq,
            quaternion=quaternion,
            gyroscope=gyroscope,
        )
        return obs

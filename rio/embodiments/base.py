# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np

from ..schema import ActionSpace, Observation

# Type variables for generic typing
ObsType = TypeVar("ObsType", bound="Observation")


class EmbodimentType(Enum):
    SINGLE_ARM = auto()
    BIMANUAL = auto()
    HUMANOID = auto()


class BaseEmbodiment(ABC, Generic[ObsType]):
    # Set by each concrete embodiment; read by the shared dispatch helpers.
    action_space: ActionSpace

    @abstractmethod
    def get_state(self, robot_state: dict, cameras: dict, **kwargs) -> ObsType:
        raise NotImplementedError("Should be implemented for each embodiment.")

    @abstractmethod
    def move(self, cmd: np.ndarray, t_cmd_target: float, **kwargs) -> None:
        raise NotImplementedError("Should be implemented by each embodiment.")

    @abstractmethod
    def build_action(self, arm_cmd: np.ndarray, gripper_cmd: np.ndarray | None = None, **kwargs) -> np.ndarray:
        raise NotImplementedError("Should be implemented by each embodiment.")

    @abstractmethod
    def parse_action(self, action: np.ndarray) -> dict:
        raise NotImplementedError("Should be implemented by each embodiment.")

    @staticmethod
    def _resolve_gripper(arm, gripper=None):
        """Resolve the node that owns the gripper channel.

        An explicit gripper client always wins; otherwise the arm itself drives
        its gripper when it exposes ``moveG`` (integrated gripper).

        Args:
            arm: Arm client.
            gripper: Optional dedicated gripper client.

        Returns:
            The gripper controller, or ``None`` if no gripper is available.
        """
        if gripper is not None:
            return gripper
        if hasattr(arm, "moveG"):
            return arm
        return None

    def _dispatch_arm(self, arm, arm_cmd, t_cmd_target):
        """Send an arm command via the method matching ``self.action_space``."""
        arm_cmd = np.asarray(arm_cmd).tolist()
        if self.action_space == ActionSpace.TASK_POS:
            arm.moveL(arm_cmd, t_cmd_target)
        elif self.action_space == ActionSpace.JOINT_POS:
            arm.moveJ(arm_cmd, t_cmd_target)
        elif self.action_space == ActionSpace.JOINT_VEL:
            arm.speedJ(arm_cmd, t_cmd_target)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not implemented")

    @staticmethod
    def _dispatch_gripper(gripper, gripper_cmd, t_cmd_target, binarize=False):
        """Send a gripper command via ``moveG``; no-op if either is missing."""
        if gripper is None or gripper_cmd is None:
            return
        if binarize:
            gripper_cmd = gripper_cmd if gripper_cmd > 0.5 else gripper_cmd - 0.1
        gripper.moveG([float(gripper_cmd)], t_cmd_target)

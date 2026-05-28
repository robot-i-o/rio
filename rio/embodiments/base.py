# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np

from ..schema import Observation

# Type variables for generic typing
ObsType = TypeVar("ObsType", bound="Observation")


class EmbodimentType(Enum):
    SINGLE_ARM = auto()
    BIMANUAL = auto()
    HUMANOID = auto()


class BaseEmbodiment(ABC, Generic[ObsType]):
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

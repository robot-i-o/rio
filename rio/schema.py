# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class ActionSpace(Enum):
    JOINT_POS = auto()
    TASK_POS = auto()
    JOINT_VEL = auto()
    TASK_VEL = auto()
    JOINT_TORQUE = auto()


@dataclass
class Camera:
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Observation:
    proprio: np.ndarray  # Defaults to policy action space
    cameras: dict[str, Camera] = field(default_factory=dict)
    sensors: dict[str, dict] = field(default_factory=dict)


@dataclass
class Step:
    timestep: int | None
    observation: Observation
    instruction: str | None
    action: np.ndarray | None
    meta: dict | None = field(default_factory=dict)

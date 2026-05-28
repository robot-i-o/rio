# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from .gear_sonic import GearSonicClient, GearSonicServer
from .gear_sonic_planner import GearSonicPlannerClient, GearSonicPlannerServer
from .policy_interface import PolicyInterfaceClient, PolicyInterfaceServer

try:
    from .smolvla import SmolVLA
except ImportError:
    logger.debug("SmolVLA not available. Install deps via scripts/setup/vla/smolvla_setup.sh")
    SmolVLA = None

try:
    from .pi0 import Pi0
except ImportError:
    logger.debug("Pi0 not available. Install deps via scripts/setup/vla/pi0_setup.sh")
    Pi0 = None

__all__ = [
    "GearSonic",
    "GearSonicPlanner",
    "PolicyInterface",
]

__policies__ = ["SmolVLA", "Pi0", "Dummy", "GearSonic", "GearSonicPlanner"]

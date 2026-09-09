# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

try:
    from .gear_sonic import GearSonicClient, GearSonicServer
except ImportError:
    logger.debug("GearSonic not available (missing torch or gear_sonic package)")
    GearSonicClient = None
    GearSonicServer = None

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

try:
    from .molmoact2 import MolmoAct2
except ImportError:
    logger.debug("MolmoAct2 not available. Install deps via scripts/setup/vla/molmoact2_setup.sh")
    MolmoAct2 = None


__all__ = [
    "GearSonic",
    "GearSonicPlanner",
    "PolicyInterface",
]

__policies__ = ["SmolVLA", "Pi0", "MolmoAct2", "Dummy", "GearSonic", "GearSonicPlanner"]

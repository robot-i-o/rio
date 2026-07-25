# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import time
from dataclasses import dataclass

import numpy as np
import pytest
from loguru import logger
from rio_hw.middleware import ServerManager

from rio.envs.factory import make_policy
from rio.policies import __policies__
from rio.policies.policy_interface import PolicyInterfaceClient, PolicyInterfaceServer

# Determine which policies to test
POLICY_ENV = os.environ.get("RECONTROL_TEST_POLICY", None)
if POLICY_ENV is None:
    logger.info("RECONTROL_TEST_POLICY not set, testing all policies")
    POLICIES_TO_TEST = __policies__
else:
    logger.info(f"RECONTROL_TEST_POLICY set to {POLICY_ENV}")
    POLICIES_TO_TEST = [POLICY_ENV]


@dataclass
class PolicyConfig:
    policy_path: str = "lerobot/smolvla_base"
    compile: bool = False
    device: str = "cuda:0"


@dataclass
class PolicyInterfaceConfig:
    instruction: str = "Move the robot arm to the left."
    resolutions: list[tuple[int, int]] = None
    action_dim: int = 6
    proprio_dim: int = 6
    chunk_size: int = 50
    use_rtc: bool = False
    freq: int = 100
    max_buffer_size: int = 30

    def __post_init__(self):
        if self.resolutions is None:
            self.resolutions = [(256, 256), (256, 256), (256, 256)]


@dataclass
class TestConfig:
    mw: str = "Shm"
    mp_method: str | None = "spawn"  # "fork" or "spawn"
    freq: int = 30

    num_cams: int = 3
    cam_resolution: tuple = (256, 256, 3)
    action_dim: int = 6


@pytest.fixture
def policy_cfg():
    return PolicyConfig()


@pytest.fixture
def policy_interface_cfg():
    return PolicyInterfaceConfig()


@pytest.fixture
def test_cfg():
    return TestConfig()


@pytest.fixture
def make_dummy_observation(test_cfg):
    """Create dummy observation matching policy schema"""
    rng = np.random.default_rng()
    obs = {
        "proprio": rng.standard_normal(test_cfg.action_dim).astype(np.float32),
    }
    for i in range(test_cfg.num_cams):
        obs[f"camera{i + 1}"] = rng.standard_normal(test_cfg.cam_resolution).astype(np.float32)
    return obs


def shm_addr():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


def make_policy_or_skip(policy_name, policy_kwargs):
    try:
        policy = make_policy(policy_name, policy_kwargs)
    except ImportError as exc:
        pytest.skip(f"{policy_name} dependencies are not installed: {exc}")

    required_methods = ("construct_policy", "set_instruction", "inference")
    missing = [name for name in required_methods if not callable(getattr(policy, name, None))]
    if missing:
        pytest.skip(f"{policy_name} does not implement the policy interface API: missing {missing}")

    return policy


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("policy_name", POLICIES_TO_TEST)
def test_policy_interface(policy_name, policy_cfg, policy_interface_cfg, test_cfg, make_dummy_observation):
    # 1. Instantiate policy wrapper
    policy = make_policy_or_skip(policy_name, vars(policy_cfg))

    # 2. Instantiate policy node (server and client factories)
    policy_interface_kwargs = {
        "policy": policy,
        "instruction": policy_interface_cfg.instruction,
        "resolutions": policy_interface_cfg.resolutions,
        "action_dim": policy_interface_cfg.action_dim,
        "proprio_dim": policy_interface_cfg.proprio_dim,
        "chunk_size": policy_interface_cfg.chunk_size,
        "use_rtc": policy_interface_cfg.use_rtc,
        "freq": policy_interface_cfg.freq,
        "max_buffer_size": policy_interface_cfg.max_buffer_size,
        "camera_keys": [f"camera{i + 1}" for i in range(test_cfg.num_cams)],
        "shm_addr": shm_addr(),
    }

    server = lambda: PolicyInterfaceServer(test_cfg.mw, **policy_interface_kwargs)
    client = lambda: PolicyInterfaceClient(test_cfg.mw, **policy_interface_kwargs)

    # Start server and client
    with ServerManager(test_cfg.mw, [server]):
        with client() as policy_client:
            # Wait for policy to initialize
            time.sleep(2)

            # 3. Put dummy obs in the request queue
            obs = make_dummy_observation
            policy_client.send_observation(obs)

            # Give the node time to process
            time.sleep(0.5)

            # 4. Read action from action buffer
            t0 = time.time()
            action_data = policy_client.get_action_chunk()
            t1 = time.time()

            # Verify action data
            assert action_data is not None, "No action data received"
            assert "actions" in action_data, "Action data missing 'actions' key"
            assert "timestamp" in action_data, "Action data missing 'timestamp' key"
            assert "ready" in action_data, "Action data missing 'ready' key"

            if action_data["ready"]:
                assert action_data["actions"].shape == (
                    policy_interface_cfg.chunk_size,
                    policy_interface_cfg.action_dim,
                ), f"Unexpected action shape: {action_data['actions'].shape}"
                logger.info(
                    f"Policy: {policy_name} - Interface test successful - "
                    f"Action chunk shape: {action_data['actions'].shape}, "
                    f"Retrieval time: {t1 - t0:.4f} seconds"
                )
            else:
                logger.warning(f"Policy: {policy_name} - Action not ready, may need more processing time")

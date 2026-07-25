# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import os
import time
from dataclasses import dataclass

import numpy as np
import pytest
from loguru import logger

from rio.envs.factory import make_policy
from rio.policies import __policies__

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
class TestConfig:
    num_cams: int = 3
    cam_resolution: tuple = (256, 256, 3)
    action_dim: int = 7


@pytest.fixture
def policy_cfg():
    return PolicyConfig()


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


def make_policy_or_skip(policy_name, policy_kwargs):
    try:
        policy = make_policy(policy_name, policy_kwargs)
    except ImportError as exc:
        pytest.skip(f"{policy_name} dependencies are not installed: {exc}")

    required_methods = ("construct_policy", "set_instruction", "get_action")
    missing = [name for name in required_methods if not callable(getattr(policy, name, None))]
    if missing:
        pytest.skip(f"{policy_name} does not implement the policy inference API: missing {missing}")

    return policy


@pytest.mark.gpu
@pytest.mark.parametrize("policy_name", POLICIES_TO_TEST)
def test_simple_inference(policy_name, policy_cfg, make_dummy_observation):
    # Build policy
    policy = make_policy_or_skip(policy_name, vars(policy_cfg))
    policy.construct_policy()
    policy.set_instruction("Move the robot arm to the left.")
    # Create dummy observation
    obs = make_dummy_observation
    # Inference pass
    t0 = time.time()
    action = policy.get_action(obs)
    t1 = time.time()
    logger.info(f"Policy: {policy_name} - Inference time: {t1 - t0:.4f} seconds")
    assert action is not None

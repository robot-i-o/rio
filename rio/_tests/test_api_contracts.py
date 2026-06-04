# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tier 1: upstream API surface + timing bounds
These guard against rio_hw changing method names
"""

import time as _time

import numpy as np
import pytest
from rio_hw import time as rio_time
from rio_hw.middleware import ClientFactory, ServerFactory, ServerManager
from rio_hw.middlewares import SERVERLESS_MW
from rio_hw.middlewares import __all__ as MW_ALL
from rio_hw.node import Node
from rio_hw.robots._arm import Arm
from rio_hw.robots._gripper import Gripper

from rio.policies._policy import Policy
from rio.policies.dummy import Dummy

pytestmark = pytest.mark.unit


def test_arm_api_unchanged():
    for m in ("moveL", "moveJ", "speedL", "speedJ"):
        assert callable(getattr(Arm, m))


def test_gripper_api_unchanged():
    assert callable(Gripper.moveG)


def test_node_api_unchanged():
    for m in ("start", "stop", "pub", "req", "pubreq"):
        assert callable(getattr(Node, m))
    for attr in ("__api__", "__pub__", "__req__"):
        assert attr in Node.__annotations__


def test_middleware_factories_present():
    assert callable(ServerManager) and callable(ServerFactory) and callable(ClientFactory)
    assert {"Shm", "Thread"}.issubset(set(SERVERLESS_MW))
    assert {"Zenoh", "Shm", "Thread"}.issubset(set(MW_ALL))


def test_policy_protocol_surface():
    for m in ("construct_policy", "set_instruction", "inference", "get_action"):
        assert callable(getattr(Policy, m))


def test_dummy_policy_output_shape():
    policy = Dummy(action_dim=7, chunk_size=8)
    actions = policy.inference({})
    assert actions.shape == (8, 7)
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()


def test_rate_period():
    assert rio_time.Rate(100).period == pytest.approx(0.01)
    assert rio_time.Rate(0).period == 0


def test_now_is_monotonic():
    assert rio_time.now_ns() <= rio_time.now_ns()
    assert rio_time.now() > 0


def test_precise_wait_in_past_returns_fast():
    t0 = _time.perf_counter()
    rio_time.precise_wait(rio_time.now() - 1.0)
    assert _time.perf_counter() - t0 < 0.05


def test_precise_sleep_is_bounded():
    t0 = _time.perf_counter()
    rio_time.precise_sleep(0.02)
    elapsed = _time.perf_counter() - t0
    assert 0.015 < elapsed < 0.5  # actually sleeps, but never hangs

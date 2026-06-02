# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tier 2: policy-node safety end-to-end
Uses the Dummy policy over the in-process Shm middleware so we can exercise
the stale/gap-detection path the control loop relies on when the policy is
slow or hasn't produced anything yet.
"""

import multiprocessing as mp
import time

import numpy as np
import pytest
from rio_hw.middleware import ServerManager

from rio.policies.dummy import Dummy
from rio.policies.policy_interface import PolicyInterfaceClient, PolicyInterfaceServer

pytestmark = pytest.mark.integration

mp.set_start_method("spawn", force=True)

MW = "Shm"
ACTION_DIM = 7
CHUNK = 8


def _kwargs():
    return {
        "policy": Dummy(action_dim=ACTION_DIM, chunk_size=CHUNK),
        "instruction": "test",
        "resolutions": [(16, 16)],
        "action_dim": ACTION_DIM,
        "proprio_dim": ACTION_DIM,
        "chunk_size": CHUNK,
        "freq": 50,
        "max_buffer_size": 30,
        "camera_keys": ["camera_1"],
    }


def _obs():
    return {
        "proprio": np.zeros(ACTION_DIM, dtype=np.float32),
        "camera_1": np.zeros((16, 16, 3), dtype=np.float32),
    }


def test_gap_returns_safe_default_then_recovers():
    kw = _kwargs()
    server = lambda: PolicyInterfaceServer(MW, **kw)
    client = lambda: PolicyInterfaceClient(MW, **kw)

    with ServerManager(MW, [server]), client() as pc:
        # No observation sent yet -> must hand back a well-formed not-ready
        # chunk quickly, never None and never a hang.
        t0 = time.perf_counter()
        data = pc.get_action_chunk()
        assert time.perf_counter() - t0 < 0.5
        assert isinstance(data, dict)
        assert not data["ready"]
        assert set(data) >= {"actions", "timestamp", "ready"}

        pc.send_observation(_obs())
        deadline = time.time() + 10
        ready = None
        while time.time() < deadline:
            ready = pc.get_action_chunk()
            if ready.get("ready"):
                break
            time.sleep(0.05)

        assert ready is not None and bool(ready["ready"]), "policy never produced a fresh chunk"
        assert ready["actions"].shape == (CHUNK, ACTION_DIM)


def test_get_action_chunk_never_blocks():
    kw = _kwargs()
    server = lambda: PolicyInterfaceServer(MW, **kw)
    client = lambda: PolicyInterfaceClient(MW, **kw)

    with ServerManager(MW, [server]), client() as pc:
        for _ in range(20):
            t0 = time.perf_counter()
            data = pc.get_action_chunk()
            assert time.perf_counter() - t0 < 0.5
            assert isinstance(data, dict) and "ready" in data

# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tier 1: action-space parsing and proprioception safety logic.
"""

import numpy as np
import pytest

from rio.embodiments.base import EmbodimentType
from rio.embodiments.bimanual import Bimanual
from rio.embodiments.single_arm import SingleArm
from rio.schema import ActionSpace

pytestmark = pytest.mark.unit


class StubArm:
    # Minimal arm client stand-in (no IPC, no hardware)
    integrated_gripper = False

    def __init__(self, num_joints=7, state=None):
        self.num_joints = num_joints
        self._state = state or {}

    def get_state(self):
        return self._state


def test_action_space_enum():
    names = {e.name for e in ActionSpace}
    assert names == {"JOINT_POS", "TASK_POS", "JOINT_VEL", "TASK_VEL", "JOINT_TORQUE"}
    assert ActionSpace["TASK_POS"] is ActionSpace.TASK_POS
    assert len({e.value for e in ActionSpace}) == len(names)


def test_embodiment_type_enum():
    # Only these two are defined. envs/env.py infers "DUAL_ARM" elsewhere
    assert {e.name for e in EmbodimentType} == {"SINGLE_ARM", "BIMANUAL"}
    with pytest.raises(KeyError):
        EmbodimentType["DUAL_ARM"]


def test_single_arm_task_pos_dim():
    arm = SingleArm(arm=StubArm(), action_space="TASK_POS")
    assert arm.arm_dim == 6


def test_single_arm_parse_build_roundtrip():
    arm = SingleArm(arm=StubArm(num_joints=7), action_space="JOINT_POS")
    action = arm.build_action(np.arange(7.0), gripper_cmd=0.5)
    assert action.shape == (8,)

    parsed = arm.parse_action(action)
    np.testing.assert_array_equal(parsed["arm_cmd"], np.arange(7.0))
    assert parsed["gripper_cmd"] == 0.5


def test_single_arm_action_too_short_is_rejected():
    arm = SingleArm(arm=StubArm(num_joints=7), action_space="JOINT_POS")
    with pytest.raises(IndexError):
        arm.parse_action(np.zeros(4)) 


def test_bimanual_parse_indices():
    arm = Bimanual(arm1=StubArm(), arm2=StubArm(), action_space="TASK_POS")
    action = np.arange(14.0)
    parsed = arm.parse_action(action)

    np.testing.assert_array_equal(parsed["arm1_cmd"], np.arange(6.0))
    assert parsed["gripper1_cmd"] == 6.0
    np.testing.assert_array_equal(parsed["arm2_cmd"], np.arange(7.0, 13.0))
    assert parsed["gripper2_cmd"] == 13.0


def test_bimanual_nan_proprio_is_dropped():
    """Non-finite arm state must not leak into proprio (safety guard)."""
    bad = StubArm(state={"eef_pose": np.full(6, np.nan), "joint_q": np.zeros(6)})
    good = StubArm(state={"eef_pose": np.ones(6), "joint_q": np.zeros(6)})
    arm = Bimanual(arm1=bad, arm2=good, action_space="TASK_POS")

    obs = arm.get_obs(cams={})
    assert obs.proprio is None
    assert np.isnan(obs.arm1_proprio_eef).any()


def test_bimanual_valid_proprio_is_concatenated():
    a1 = StubArm(state={"eef_pose": np.ones(6), "joint_q": np.zeros(6)})
    a2 = StubArm(state={"eef_pose": np.full(6, 2.0), "joint_q": np.zeros(6)})
    arm = Bimanual(arm1=a1, arm2=a2, action_space="TASK_POS")

    obs = arm.get_obs(cams={})
    assert obs.proprio.shape == (14,)

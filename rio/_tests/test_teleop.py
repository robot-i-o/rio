# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Tier 1: teleop input -> motion mapping."""

import numpy as np
import pytest

from rio.embodiments.single_arm import SingleArm
from rio.envs.poll import Interface, TeleopMode

pytestmark = pytest.mark.unit


class FakeKeyboard:
    def __init__(self, pressed=""):
        self.pressed = pressed

    def get_state(self):
        return {"alphanumeric_state": [ord(c) for c in self.pressed]}


class FakeStick:
    """Spacemouse or gamepad stand in"""
    def __init__(self, motion=None, buttons=()):
        self._motion = np.zeros(6) if motion is None else np.asarray(motion, float)
        self._buttons = set(buttons)

    def get_motion_state_transformed(self):
        return self._motion

    def is_button_pressed(self, idx):
        return idx in self._buttons


def test_keyboard_translation_keys():
    motion, gripper, _, mode = Interface.poll_keyboard(FakeKeyboard("w"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert motion[0] == 1.0 and not motion[1:].any()
    assert gripper is None

    motion, *_ = Interface.poll_keyboard(FakeKeyboard("s"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert motion[0] == -1.0


def test_keyboard_gripper_and_mode():
    _, gripper, _, _ = Interface.poll_keyboard(FakeKeyboard("["), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 0.0
    _, gripper, _, _ = Interface.poll_keyboard(FakeKeyboard("]"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 1.0
    _, _, _, mode = Interface.poll_keyboard(FakeKeyboard("2"), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert mode is TeleopMode.ROTATION


def test_interface_poll_dispatch_matches_direct():
    kb = FakeKeyboard("d")
    via_poll = Interface.poll("Keyboard", kb, 0.0, 0.0, TeleopMode.TRANSLATION)
    direct = Interface.poll_keyboard(kb, 0.0, 0.0, TeleopMode.TRANSLATION)
    np.testing.assert_array_equal(via_poll[0], direct[0])


def test_spacemouse_buttons():
    _, gripper, _, _ = Interface.poll_spacemouse(FakeStick(buttons=[0]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 0.0 
    _, gripper, _, _ = Interface.poll_spacemouse(FakeStick(buttons=[1]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 1.0

    _, _, t_change, mode = Interface.poll_spacemouse(FakeStick(buttons=[0, 1]), 5.0, 0.0, TeleopMode.TRANSLATION)
    assert mode is not TeleopMode.TRANSLATION
    assert t_change == 5.0


def test_gamepad_gripper_buttons():
    _, gripper, _, _ = Interface.poll_gamepad(FakeStick(buttons=[0]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 1.0
    _, gripper, _, _ = Interface.poll_gamepad(FakeStick(buttons=[3]), 0.0, 0.0, TeleopMode.TRANSLATION)
    assert gripper == 0.0


def test_eef_cmd_translation_zeroes_rotation():
    target = np.zeros(6)
    delta = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    out = SingleArm.make_teleop_eef_cmd(10, TeleopMode.TRANSLATION, delta, target, 1.0, 1.0)
    np.testing.assert_allclose(out[:3], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(out[3:], 0.0, atol=1e-9)


def test_eef_cmd_2d_drops_z():
    out = SingleArm.make_teleop_eef_cmd(10, TeleopMode.TRANSLATION_2D, np.ones(6), np.zeros(6), 1.0, 1.0)
    np.testing.assert_allclose(out[:3], [0.1, 0.1, 0.0])


def test_eef_cmd_rotation_zeroes_translation():
    out = SingleArm.make_teleop_eef_cmd(10, TeleopMode.ROTATION, np.ones(6), np.zeros(6), 1.0, 1.0)
    np.testing.assert_allclose(out[:3], 0.0, atol=1e-9)
    assert np.linalg.norm(out[3:]) > 0


def test_gello_cmd_clips_and_deadbands():
    target = np.zeros(3)
    out = SingleArm.make_gello_joint_cmd(100, np.array([1.0, 1.0, 1.0]), target.copy(), max_joint_delta=0.02)
    np.testing.assert_allclose(out, 0.02)

    out = SingleArm.make_gello_joint_cmd(100, np.array([0.001, 0.001, 0.001]), np.zeros(3), max_joint_delta=0.02, deadband=0.01)
    np.testing.assert_allclose(out, 0.0)

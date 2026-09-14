# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from rio.control import (
    ButtonPoller,
    Command,
    RecordingSession,
    home_pose,
    optional_buttons,
    ramp_to_pose,
    resolve_bindings,
)
from rio.control.buttons import resolve_keycode
from rio.control.homing import gripper_indices

KEY_A, KEY_B, KEY_C = 30, 48, 46


class FakeKeyboard:
    """Stands in for a keyboard node client publishing monotonic press counters."""

    def __init__(self):
        self.counts = np.zeros((768,), dtype=np.uint16)

    def press(self, keycode: int, times: int = 1):
        for _ in range(times):
            self.counts[keycode] = np.uint16((int(self.counts[keycode]) + 1) % 65536)

    def get_state(self):
        return {"press_count": self.counts.copy()}


class FakeRecorder:
    """Minimal recorder stand-in tracking the calls a session makes."""

    def __init__(self, is_closed=True):
        self.calls = []
        self._is_closed = is_closed
        self._is_saving = False

    def get_state(self):
        return {"is_closed": self._is_closed, "is_saving": self._is_saving}

    def new_trajectory(self, wait=True):
        self.calls.append("new_trajectory")
        self._is_closed = False

    def save(self, wait=True):
        self.calls.append("save")
        self._is_saving = not wait
        self._is_closed = True

    def discard_last(self):
        self.calls.append("discard_last")

    def record_step(self, step):
        self.calls.append("record_step")

    def finish_saving(self):
        self._is_saving = False


@pytest.fixture
def poller():
    keyboard = FakeKeyboard()
    return keyboard, ButtonPoller(keyboard, {"a": Command.TOGGLE_EPISODE, "c": Command.DISCARD_LAST})


def test_resolve_keycode():
    assert resolve_keycode("a") == KEY_A
    assert resolve_keycode("A") == KEY_A
    assert resolve_keycode(KEY_C) == KEY_C
    with pytest.raises(ValueError):
        resolve_keycode("@")


def test_first_poll_establishes_baseline(poller):
    keyboard, button_poller = poller
    keyboard.press(KEY_A, times=4)
    # Presses made before the loop starts must not replay into the first iteration.
    assert button_poller.poll() == []


def test_press_reported_once(poller):
    keyboard, button_poller = poller
    button_poller.poll()

    keyboard.press(KEY_A)
    assert button_poller.poll() == [Command.TOGGLE_EPISODE]
    # Held or not, a press is never reported twice.
    assert button_poller.poll() == []


def test_repeated_presses_between_polls(poller):
    keyboard, button_poller = poller
    button_poller.poll()

    keyboard.press(KEY_A, times=3)
    assert button_poller.poll() == [Command.TOGGLE_EPISODE] * 3


def test_unbound_key_ignored(poller):
    keyboard, button_poller = poller
    button_poller.poll()

    keyboard.press(KEY_B, times=5)
    assert button_poller.poll() == []


def test_counter_wraparound(poller):
    keyboard, button_poller = poller
    keyboard.counts[KEY_A] = 65535
    button_poller.poll()

    keyboard.press(KEY_A, times=2)
    assert button_poller.poll() == [Command.TOGGLE_EPISODE] * 2


def test_poller_without_keyboard():
    assert ButtonPoller(None, {"a": Command.TOGGLE_EPISODE}).poll() == []


def test_session_toggles_episode():
    recorder = FakeRecorder()
    session = RecordingSession(recorder, announce=False)
    assert not session.is_recording

    session.handle(Command.TOGGLE_EPISODE)
    assert session.is_recording
    assert recorder.calls == ["new_trajectory"]

    session.handle(Command.TOGGLE_EPISODE)
    assert not session.is_recording
    assert recorder.calls == ["new_trajectory", "save"]


def test_session_records_only_while_recording():
    recorder = FakeRecorder()
    session = RecordingSession(recorder, announce=False)

    session.record_step(object())
    assert "record_step" not in recorder.calls

    session.handle(Command.TOGGLE_EPISODE)
    session.record_step(object())
    assert recorder.calls.count("record_step") == 1


def test_session_discard_only_when_idle():
    recorder = FakeRecorder()
    session = RecordingSession(recorder, announce=False)

    session.handle(Command.TOGGLE_EPISODE)
    session.handle(Command.DISCARD_LAST)
    # Refused mid-episode so a mispress cannot delete a demonstration being recorded.
    assert "discard_last" not in recorder.calls

    session.handle(Command.TOGGLE_EPISODE)
    session.refresh()
    recorder.finish_saving()
    session.refresh()
    session.handle(Command.DISCARD_LAST)
    assert "discard_last" in recorder.calls


def test_session_saves_on_close():
    recorder = FakeRecorder()
    session = RecordingSession(recorder, announce=False)
    session.handle(Command.TOGGLE_EPISODE)

    session.close()
    assert recorder.calls == ["new_trajectory", "save"]
    assert not session.is_recording


def test_session_without_recorder_is_inert():
    session = RecordingSession(None, announce=False)
    session.handle(Command.TOGGLE_EPISODE)
    session.record_step(object())
    session.close()
    assert not session.is_recording


class FakeArm:
    """Embodiment stand-in with the bimanual action layout: 6 joints + gripper, twice."""

    arm1_dim = 6
    arm2_dim = 6

    def parse_action(self, action):
        return {
            "arm1_cmd": action[:6],
            "gripper1_cmd": action[6],
            "arm2_cmd": action[7:13],
            "gripper2_cmd": action[13],
            "hand1_cmd": None,
            "hand2_cmd": None,
        }


class FakeEnv:
    """Records every commanded pose so a ramp can be inspected."""

    def __init__(self, proprio):
        self.robot = FakeArm()
        self._proprio = np.asarray(proprio, dtype=np.float32)
        self.commands = []

    def get_state(self, action=None):
        observation = type("Obs", (), {"proprio": self._proprio})()
        return type("Step", (), {"observation": observation})()

    def move(self, action, t_cmd_target):
        self.commands.append(np.asarray(action, dtype=np.float32).copy())


def test_gripper_indices_bimanual():
    assert gripper_indices(FakeArm(), 14) == [6, 13]


def test_home_pose_opens_grippers():
    pose = home_pose(FakeArm(), 14)
    assert pose.shape == (14,)
    assert pose[6] == 1.0 and pose[13] == 1.0
    assert np.all(pose[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] == 0.0)


def test_ramp_reaches_target_gradually():
    start = np.full((14,), 0.5, dtype=np.float32)
    env = FakeEnv(start)
    target = home_pose(FakeArm(), 14)

    final = ramp_to_pose(env, target, freq=50, duration=0.2)

    assert len(env.commands) == 10  # 0.2s at 50 Hz
    np.testing.assert_allclose(env.commands[-1], target, atol=1e-5)
    np.testing.assert_allclose(final, target, atol=1e-5)

    # The first command must be a small step, not a jump to the target.
    first_step = np.abs(env.commands[0] - start).max()
    total = np.abs(target - start).max()
    assert first_step < 0.1 * total, f"first step {first_step} is too large for a ramp"


def test_ramp_is_monotonic_per_joint():
    start = np.zeros((14,), dtype=np.float32)
    env = FakeEnv(start)
    target = np.full((14,), 1.0, dtype=np.float32)

    ramp_to_pose(env, target, freq=50, duration=0.4)

    path = np.stack(env.commands)
    deltas = np.diff(path, axis=0)
    # A cosine ease never reverses direction, so the arm never backtracks.
    assert np.all(deltas >= -1e-6)


def test_ramp_rejects_mismatched_pose():
    env = FakeEnv(np.zeros((14,), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        ramp_to_pose(env, np.zeros((7,), dtype=np.float32), freq=50, duration=0.1)


def test_resolve_bindings_uses_defaults_when_unset():
    class Cfg:
        pass

    defaults = {"a": Command.TOGGLE_EPISODE}
    assert resolve_bindings(Cfg(), defaults, "button_bindings") == defaults


def test_resolve_bindings_reads_named_field():
    class Cfg:
        button_bindings = {"a": "TOGGLE_EPISODE"}
        inference_button_bindings = {"b": "GO_HOME"}

    # Each entrypoint reads its own field, so station bindings do not leak into inference.
    assert resolve_bindings(Cfg(), {}, "button_bindings") == {"a": Command.TOGGLE_EPISODE}
    assert resolve_bindings(Cfg(), {}, "inference_button_bindings") == {"b": Command.GO_HOME}


def test_resolve_bindings_rejects_unknown_command():
    class Cfg:
        button_bindings = {"a": "LAUNCH_ROCKET"}

    with pytest.raises(ValueError, match="LAUNCH_ROCKET"):
        resolve_bindings(Cfg(), {}, "button_bindings")


class FailingClient:
    """Node client whose connection fails, as when the device is not plugged in."""

    def __enter__(self):
        raise AssertionError  # what the middleware raises when a node never becomes ready

    def __exit__(self, *exc):
        return False


class WorkingClient:
    def __init__(self):
        self.exited = False

    def __enter__(self):
        return "connected"

    def __exit__(self, *exc):
        self.exited = True
        return False


def test_optional_buttons_without_config():
    # No buttons declared in the station config at all.
    with optional_buttons(None) as buttons:
        assert buttons is None


def test_optional_buttons_when_device_missing():
    # Declared in the config, but not plugged in: degrade instead of failing the script.
    with optional_buttons(FailingClient) as buttons:
        assert buttons is None


def test_optional_buttons_connects_and_closes():
    client = WorkingClient()
    with optional_buttons(lambda: client) as buttons:
        assert buttons == "connected"
    assert client.exited


def test_optional_buttons_does_not_swallow_body_errors():
    # Only connection failures degrade; a bug in the loop must still surface.
    client = WorkingClient()
    with pytest.raises(ZeroDivisionError):
        with optional_buttons(lambda: client) as buttons:
            assert buttons == "connected"
            raise ZeroDivisionError
    assert client.exited


def test_session_records_continuously_without_buttons():
    """Without buttons the caller starts one episode covering the whole session."""
    recorder = FakeRecorder()
    session = RecordingSession(recorder, announce=False)

    session.handle(Command.TOGGLE_EPISODE)  # what the script does after the Enter prompt
    for _ in range(3):
        session.record_step(object())
    assert recorder.calls.count("record_step") == 3

    session.close()
    assert recorder.calls[-1] == "save"

# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical button input, translated into transport-independent commands.

A control loop should consume `Command` values rather than raw key codes, so that the
same intent can later be raised by a different input source (a web UI, a foot pedal)
without touching the loop.
"""

from contextlib import contextmanager
from enum import Enum, auto

import numpy as np
from loguru import logger

# Linux input-event codes for letters and digits (linux/input-event-codes.h). Defined
# here so configs can bind buttons by character without depending on evdev being
# installed on the machine reading the config.
_KEYCODES = {
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34, "h": 35, "i": 23,
    "j": 36, "k": 37, "l": 38, "m": 50, "n": 49, "o": 24, "p": 25, "q": 16, "r": 19,
    "s": 31, "t": 20, "u": 22, "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
}  # fmt: skip


class Command(Enum):
    """An operator intent, independent of which button or device produced it."""

    TOGGLE_EPISODE = auto()
    """Start recording a demonstration, or stop and save the one in progress."""

    DISCARD_LAST = auto()
    """Delete the most recently saved demonstration."""

    TOGGLE_INFERENCE = auto()
    """Start or stop the policy inference loop."""

    GO_HOME = auto()
    """Return the robot to its home pose, ready for another trial."""


def resolve_keycode(key: str | int) -> int:
    """Resolve a binding key to a Linux input-event code.

    Args:
        key: A single character (e.g. "a") or a raw input-event code (e.g. 30).

    Returns:
        The input-event code.

    Raises:
        ValueError: If a character has no known code.
    """
    if isinstance(key, int):
        return key
    code = _KEYCODES.get(key.lower())
    if code is None:
        raise ValueError(f"No known input-event code for key {key!r}. Bind by integer code instead.")
    return code


def resolve_bindings(args, defaults: dict[str | int, Command], field_name: str = "button_bindings") -> dict:
    """Resolve button bindings from a config, falling back to the caller's defaults.

    Each entrypoint reads its own config field, because a policy config subclasses its
    station config: a single shared field would silently apply the station's
    teleoperation bindings to inference, where those commands mean nothing.

    Args:
        args: Config object, which may define `field_name` as a mapping of key to a
            `Command` or its name (e.g. {"b": "GO_HOME"}).
        defaults: Bindings to use when the config does not specify any.
        field_name: Config field to read.

    Returns:
        Mapping of key to `Command`.

    Raises:
        ValueError: If a configured command name is not a known `Command`.
    """
    configured = getattr(args, field_name, None)
    if not configured:
        return dict(defaults)

    bindings = {}
    for key, name in configured.items():
        if isinstance(name, Command):
            bindings[key] = name
            continue
        try:
            bindings[key] = Command[str(name).upper()]
        except KeyError as e:
            valid = ", ".join(command.name for command in Command)
            raise ValueError(f"Unknown command {name!r} bound to key {key!r}. Valid commands: {valid}") from e
    return bindings


@contextmanager
def optional_buttons(client_factory):
    """Connect a buttons node, yielding None if it is absent or unavailable.

    A station may declare a button pad that is not currently plugged in, and an operator
    may want to run without one at all. Neither should stop the script: the node's own
    failure surfaces as a bare timeout several seconds later, which says nothing about
    the cause. Only a failure to connect is caught; once connected, errors propagate.

    Args:
        client_factory: Callable returning the node client context manager, or None when
            no buttons are configured.

    Yields:
        The connected client, or None to run without buttons.
    """
    if client_factory is None:
        yield None
        return

    client = client_factory()
    try:
        buttons = client.__enter__()
    except Exception as e:
        logger.warning(
            f"Button pad unavailable, continuing without it ({type(e).__name__}: {e}). "
            "Check that the device is plugged in and that /dev/input is readable."
        )
        yield None
        return

    try:
        yield buttons
    finally:
        client.__exit__(None, None, None)


class ButtonPoller:
    """Turns published keyboard state into discrete commands, one per physical press.

    Detection uses the monotonic per-key press counters published by `EvdevKeyboard`
    rather than the held-key state, so a press is reported exactly once no matter how
    briefly the button was held or how the loop and node frequencies relate. Reading
    held state instead would both miss short taps and re-fire while a button is held.
    """

    def __init__(self, keyboard, bindings: dict[str | int, Command]):
        """Initialize the poller.

        Args:
            keyboard: A connected keyboard node client publishing `press_count`, such as
                `EvdevKeyboardClient`. May be None, in which case polling yields nothing.
            bindings: Maps a key (character or input-event code) to the command it raises.
        """
        self._keyboard = keyboard
        self._bindings = {resolve_keycode(key): command for key, command in bindings.items()}
        self._last_counts: np.ndarray | None = None
        self._warned_unsupported = False

        if keyboard is not None:
            pretty = ", ".join(f"{key}->{command.name}" for key, command in bindings.items())
            logger.info(f"Button bindings: {pretty}")

    def poll(self) -> list[Command]:
        """Collect commands from button presses since the previous call.

        Returns:
            Commands in key order, one entry per press. A button pressed twice between
            calls yields its command twice.
        """
        if self._keyboard is None:
            return []

        state = self._keyboard.get_state()
        counts = state.get("press_count")
        if counts is None:
            if not self._warned_unsupported:
                logger.warning(
                    "Keyboard node does not publish 'press_count'; buttons are disabled. "
                    "Use EvdevKeyboard for reliable button input."
                )
                self._warned_unsupported = True
            return []

        counts = np.asarray(counts)
        if self._last_counts is None:
            # Adopt the first sample as the baseline so presses predating startup,
            # including any used to launch the script, are not replayed.
            self._last_counts = counts.copy()
            return []

        # Counters are unsigned and wrap; the modular difference stays correct.
        deltas = (counts - self._last_counts).astype(np.uint16)
        self._last_counts = counts.copy()

        commands = []
        for keycode, command in self._bindings.items():
            if keycode >= len(deltas):
                continue
            commands.extend([command] * int(deltas[keycode]))
        return commands

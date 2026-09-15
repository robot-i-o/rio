# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Recording session state, driven by operator commands."""

from enum import Enum, auto

from loguru import logger

from .buttons import Command

# ANSI colours, for feedback on terminal since buttons have no feedback
_RESET = "\033[0m"
_BANNERS = {
    "IDLE": "\033[1;30;47m",  # black on white
    "RECORDING": "\033[1;37;41m",  # white on red
    "SAVING": "\033[1;30;43m",  # black on yellow
}


class RecordingSession:
    """Tracks whether a demonstration is being recorded and applies operator commands.

    Owns the meaning of a button press: the same `TOGGLE_EPISODE` command starts an
    episode when idle and stops and saves one while recording, which is what lets a
    small button pad cover a full collection workflow.
    """

    class State(Enum):
        IDLE = auto()
        RECORDING = auto()
        SAVING = auto()

    def __init__(self, recorder, announce: bool = True):
        """Initialize the session.

        Args:
            recorder: A connected recorder node client, or None to disable recording.
            announce: Whether to print the state banner on each transition.
        """
        self._recorder = recorder
        self._announce = announce
        self._state = self.State.IDLE
        self._episode_count = 0

        if recorder is not None:
            # The recorder may have opened a trajectory at startup; close it so the
            # first episode begins on an operator press rather than mid-setup.
            if not recorder.get_state().get("is_closed", True):
                recorder.save(wait=True)
                self._recorder.discard_last()
                logger.info("Closed the auto-started trajectory; press the start button to record.")

        self._print_banner()

    @property
    def state(self) -> "RecordingSession.State":
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._state is self.State.RECORDING

    def handle(self, command: Command) -> None:
        """Apply a single operator command to the session.

        Commands that do not apply in the current state are logged and ignored, so a
        mistaken press never leaves the session inconsistent.

        Args:
            command: The command to apply.
        """
        if self._recorder is None:
            logger.warning(f"Ignoring {command.name}: no recorder is configured.")
            return

        if command is Command.TOGGLE_EPISODE:
            self._toggle_episode()
        elif command is Command.DISCARD_LAST:
            self._discard_last()
        else:
            logger.warning(f"Ignoring {command.name}: not handled by a recording session.")

    def record_step(self, step) -> None:
        """Record a step if an episode is in progress.

        Args:
            step: The environment step to record.
        """
        if self._state is self.State.RECORDING:
            self._recorder.record_step(step)

    def refresh(self) -> None:
        """Reconcile session state with the recorder, which saves asynchronously."""
        if self._state is not self.State.SAVING:
            return
        if self._recorder.get_state().get("is_saving", False):
            return
        self._state = self.State.IDLE
        self._print_banner()

    def close(self) -> None:
        """Save any episode still in progress. Safe to call more than once."""
        if self._recorder is None:
            return
        if self._state is self.State.RECORDING:
            logger.info("Exiting mid-episode; saving the trajectory.")
            self._recorder.save(wait=True)
            self._state = self.State.IDLE

    def _toggle_episode(self) -> None:
        if self._state is self.State.SAVING:
            logger.warning("Still saving the previous episode; ignoring.")
            return

        if self._state is self.State.IDLE:
            self._recorder.new_trajectory(wait=False)
            self._episode_count += 1
            self._state = self.State.RECORDING
        else:
            self._recorder.save(wait=False)
            self._state = self.State.SAVING
        self._print_banner()

    def _discard_last(self) -> None:
        if self._state is not self.State.IDLE:
            logger.warning("Can only discard between episodes; stop the current one first.")
            return
        self._recorder.discard_last()
        self._episode_count = max(self._episode_count - 1, 0)
        self._print_banner(note="discarded previous episode")

    def _print_banner(self, note: str | None = None) -> None:
        if not self._announce:
            return
        name = self._state.name
        colour = _BANNERS.get(name, "")
        suffix = f"  ({note})" if note else ""
        print(f"\n{colour}  {name:<10}  episodes saved: {self._episode_count}  {_RESET}{suffix}\n", flush=True)

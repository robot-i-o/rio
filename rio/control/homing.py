# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Ramped motion to a fixed pose, for resetting the robot between trials."""

import numpy as np
from loguru import logger
from rio_hw import time

# Ease in and out of the motion rather than starting and stopping abruptly.
_MIN_STEPS = 2


def gripper_indices(robot, action_dim: int) -> list[int]:
    """Find which entries of an action vector are gripper commands.

    Recovered by asking the embodiment to parse an action whose every entry holds its own
    index, so the parsed gripper "values" are the indices themselves. This avoids
    hard-coding a layout per embodiment.

    Args:
        robot: The embodiment, which must implement `parse_action`.
        action_dim: Length of the action vector.

    Returns:
        Indices of the gripper entries, ascending. Empty if none could be identified.
    """
    probe = np.arange(action_dim, dtype=np.float32)
    try:
        parsed = robot.parse_action(probe)
    except (AttributeError, IndexError, ValueError) as e:
        logger.warning(f"Could not determine gripper indices from the embodiment: {e}")
        return []

    indices = []
    for key, raw in parsed.items():
        if "gripper" not in key or raw is None:
            continue
        value = np.asarray(raw)
        if value.ndim != 0:
            continue  # a vector-valued gripper command has no single index
        index = round(float(value))
        if 0 <= index < action_dim:
            indices.append(index)
    return sorted(set(indices))


def home_pose(robot, action_dim: int, gripper_open: float = 1.0) -> np.ndarray:
    """Build the default home pose: every joint at zero, every gripper open.

    Args:
        robot: The embodiment, used to locate the gripper entries.
        action_dim: Length of the action vector.
        gripper_open: Value that means fully open for this robot.

    Returns:
        The home pose, in the same layout as an action.
    """
    pose = np.zeros((action_dim,), dtype=np.float32)
    for index in gripper_indices(robot, action_dim):
        pose[index] = gripper_open
    return pose


def ramp_to_pose(
    env,
    target: np.ndarray,
    freq: int,
    duration: float = 5.0,
    arm_latency: float = 0.0,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Move the robot to a pose gradually, blocking until it arrives.

    The robot is driven along a cosine ease-in-out profile from where it currently is to
    `target`, one command per control cycle. Commanding the target directly would ask the
    arm to cross the whole distance in a single cycle, which is what makes an unramped
    "go home" dangerous.

    Args:
        env: The environment to command.
        target: Target pose, in action layout.
        freq: Control frequency in Hz, matching the caller's loop.
        duration: Seconds the motion should take. Longer is gentler.
        arm_latency: Extra lead time added to each command timestamp.
        start: Pose to start from. Defaults to the robot's current proprioception.

    Returns:
        The pose actually commanded at the end of the ramp, for the caller to adopt as
        its current command.

    Raises:
        ValueError: If `target` does not match the robot's proprioception layout.
    """
    if start is None:
        start = np.asarray(env.get_state().observation.proprio, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32).copy()
    target = np.asarray(target, dtype=np.float32)

    if start.shape != target.shape:
        raise ValueError(f"Home pose has shape {target.shape}, but the robot reports {start.shape}.")

    steps = max(round(duration * freq), _MIN_STEPS)
    dt = 1.0 / freq
    logger.info(f"Homing over {duration:.1f}s ({steps} steps) to {np.round(target, 3)}")

    t_start = time.now()
    commanded = start
    for step in range(1, steps + 1):
        # Cosine ease: zero velocity at both ends, so neither departure nor arrival jerks.
        blend = 0.5 - 0.5 * np.cos(np.pi * step / steps)
        commanded = start + blend * (target - start)

        t_cycle_end = t_start + step * dt
        env.move(commanded, t_cmd_target=t_cycle_end + dt + arm_latency)
        time.precise_wait(t_cycle_end)

    logger.info("Homing complete.")
    return commanded

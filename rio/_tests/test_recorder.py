# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import robodm
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager
from tqdm import tqdm

from rio.data.loader import dict_to_step
from rio.envs.factory import make_node
from rio.schema import Camera, Observation, Step

mp.set_start_method("spawn", force=True)

_INSTRUCTION = "Move the arm to the target position"


def create_dummy_step(timestep: int, rng: np.random.Generator | None = None) -> Step:
    if rng is None:
        rng = np.random.default_rng()

    # Create dummy camera data with random values
    rgb_image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    depth_image = rng.integers(0, 5, size=(480, 640), dtype=np.uint16)

    camera = Camera(rgb=rgb_image, depth=depth_image, meta={"camera_name": "wrist_camera", "timestep": timestep})

    # Create observation with random values
    observation = Observation(
        proprio=rng.standard_normal(7, dtype=np.float32),
        cameras={"wrist": camera},
    )

    # Create action with random values
    action = rng.standard_normal(7, dtype=np.float32)
    instruction = _INSTRUCTION
    # Create step
    step = Step(
        timestep=timestep,
        observation=observation,
        action=action,
        instruction=instruction,
        meta={"episode": 1, "step_in_episode": timestep},
    )

    return step


def compare_steps(loaded: Step, original: Step, timestep: int):
    """Compare two Step objects for equality."""
    # Verify timestep
    assert abs(loaded.timestep - original.timestep) < 1e-10, f"Timestep mismatch at index {timestep}"
    assert loaded.instruction == original.instruction, f"Instruction mismatch at index {timestep}"

    # Verify action
    np.testing.assert_array_equal(loaded.action.shape, original.action.shape)
    np.testing.assert_array_almost_equal(loaded.action, original.action, decimal=5)

    # Verify observation fields
    obs_loaded = loaded.observation
    obs_orig = original.observation

    # Check all proprio fields
    for field in ["proprio", "proprio_eef", "proprio_joints", "hand_pose", "hand_joints"]:
        val_loaded = getattr(obs_loaded, field, None)
        val_orig = getattr(obs_orig, field, None)
        if val_loaded is not None and val_orig is not None:
            np.testing.assert_array_equal(
                val_loaded.shape, val_orig.shape, err_msg=f"{field} shape mismatch at timestep {timestep}", verbose=False
            )
            np.testing.assert_array_almost_equal(
                val_loaded, val_orig, decimal=5, err_msg=f"{field} value mismatch at timestep {timestep}", verbose=False
            )

    # Check gripper position
    if obs_loaded.gripper_position is not None and obs_orig.gripper_position is not None:
        assert np.isclose(obs_loaded.gripper_position, obs_orig.gripper_position, rtol=1e-5)

    # Check cameras
    assert obs_loaded.cameras.keys() == obs_orig.cameras.keys()
    for cam_name in obs_orig.cameras:
        # breakpoint()
        cam_loaded = obs_loaded.cameras[cam_name]
        cam_orig = obs_orig.cameras[cam_name]

        # Display images side by side for visual inspection (optional)

        if cam_loaded.rgb is not None and cam_orig.rgb is not None:
            np.testing.assert_array_equal(
                cam_loaded.rgb.shape,
                cam_orig.rgb.shape,
                err_msg=f"RGB image shape mismatch at timestep {timestep}",
                verbose=False,
            )
            np.testing.assert_array_equal(
                cam_loaded.rgb, cam_orig.rgb, err_msg=f"RGB image value mismatch at timestep {timestep}", verbose=False
            )

        if cam_loaded.depth is not None and cam_orig.depth is not None:
            np.testing.assert_array_equal(
                cam_loaded.depth.shape,
                cam_orig.depth.shape,
                err_msg=f"Depth image shape mismatch at timestep {timestep}",
                verbose=False,
            )
            np.testing.assert_array_almost_equal(
                cam_loaded.depth,
                cam_orig.depth,
                decimal=5,
                err_msg=f"Depth image value mismatch at timestep {timestep}",
                verbose=False,
            )


@pytest.fixture
def temp_trajectory_path():
    """Create a temporary path for trajectory file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "test_trajectory.vla"
        yield str(trajectory_path)


@pytest.fixture
def test_trajectory_path():
    """Create a test/data in home directory for trajectory file."""
    home_dir = Path.home()
    data_dir = home_dir / "recontrol_test_data"
    data_dir.mkdir(exist_ok=True)
    trajectory_path = data_dir / "test_trajectory.vla"
    yield str(trajectory_path)


@pytest.fixture
def middleware():
    mw = os.environ.get("MIDDLEWARE", "Thread")
    yield mw


@pytest.mark.integration
@pytest.mark.parametrize(
    "freq,duration,description",
    [
        (10, 2, "Low frequency"),
        (100, 2, "Medium frequency"),
        (250, 2, "Medium frequency"),
        # (500, 2, "High frequency"),
    ],
)
def test_recorder_loader_roundtrip(temp_trajectory_path, middleware, freq, duration, description):
    """
    Test recorder/loader roundtrip with data integrity verification.

    Records data at various frequencies for different durations, then verifies:
    - Correct number of timesteps
    - All timestep values and sequence
    - All array shapes, sizes, and values match original
    """
    dt = 1.0 / freq
    rng = np.random.default_rng(seed=42)

    logger.info(f"{description}: Recording at {freq} Hz for {duration}s (expected ~{int(freq * duration)} steps)...")

    # Store original steps for comparison
    original_steps = []

    # Create recorder configuration
    # Adjust trajectory path to include test parameters
    test_path = f"{temp_trajectory_path.rsplit('.', 1)[0]}_{freq}hz_{duration}s.vla"
    logger.info(f"Trajectory path: {test_path}")

    recorder_cfg = {
        "path": test_path,
        "freq": freq,
        "max_queue_size": 10 * freq,
        "video_codec": "rawvideo",
        "rawvideo_codec": "rawvideo_pyarrow",
    }

    # Phase 1: Record dummy data at specified frequency
    recorder_server, recorder_client = make_node(middleware, "data", "Recorder", recorder_cfg, package="rio")

    with ServerManager(middleware, [recorder_server]):
        with recorder_client() as recorder:
            steps = [create_dummy_step(timestep=i, rng=rng) for i in range(int(freq * duration))]
            t_start = time.now()
            it = 0
            max_iterations = int(freq * duration)

            with tqdm(total=max_iterations, desc=f"{description} - Recording", unit="steps") as pbar:
                while it < max_iterations:
                    t_cycle_end = t_start + (it + 1) * dt

                    step = steps[it]
                    # Create and record step
                    original_steps.append(step)
                    recorder.record_step(step)

                    # Precise timing
                    time.precise_wait(t_cycle_end)
                    it += 1
                    pbar.update(1)

                    # Safety check: don't run forever if timing breaks
                    if time.now() - t_start > duration * 1.5:
                        logger.warning(f"Safety timeout reached at iteration {it}")
                        break

            num_steps = len(original_steps)
            actual_duration = time.now() - t_start
            actual_freq = num_steps / actual_duration

            recorder.save(wait=True, timeout=None)

    logger.info(f"Recorded {num_steps} steps in {actual_duration:.2f}s (target: {freq} Hz, actual: {actual_freq:.1f} Hz)")

    # Verify the file was created
    assert Path(test_path).exists(), "Trajectory file was not created"

    # Phase 2: Load and verify data
    _loaded = robodm.Trajectory(path=test_path, mode="r")
    loaded_data = _loaded.load()

    timesteps = loaded_data.get("timestep", [])
    assert len(timesteps) == num_steps, f"Expected {num_steps} timesteps, got {len(timesteps)}"

    for idx in range(num_steps):
        loaded_step_dict = {feature: loaded_data[feature][idx] for feature in loaded_data}
        loaded_step = dict_to_step(loaded_step_dict)
        original_step = original_steps[idx]

        compare_steps(loaded_step, original_step, idx)
    logger.info("All steps verified successfully!")

    # TODO: Implement loader verification logic


def test_discard_last_removes_trajectory(tmp_path):
    """A saved trajectory can be discarded, and its index reused by the next one."""
    middleware = "Thread"
    recorder_cfg = {
        "path": str(tmp_path),
        "freq": 100,
        "video_codec": "rawvideo",
        "start_recording": False,
    }
    recorder_server, recorder_client = make_node(middleware, "data", "Recorder", recorder_cfg, package="rio")

    with ServerManager(middleware, [recorder_server]):
        with recorder_client() as recorder:
            # First episode, kept.
            recorder.new_trajectory(wait=True)
            for i in range(5):
                recorder.record_step(create_dummy_step(timestep=i))
            recorder.save(wait=True, timeout=None)

            # Second episode, discarded.
            recorder.new_trajectory(wait=True)
            for i in range(5):
                recorder.record_step(create_dummy_step(timestep=i))
            recorder.save(wait=True, timeout=None)

            saved = sorted(f for f in os.listdir(tmp_path) if f.endswith(".vla"))
            assert len(saved) == 2, saved
            doomed = saved[-1]

            recorder.discard_last()
            time.sleep(0.5)

            remaining = sorted(f for f in os.listdir(tmp_path) if f.endswith(".vla"))
            assert doomed not in remaining
            assert len(remaining) == 1, remaining

            # The freed index is reused rather than leaving a gap in the numbering.
            recorder.new_trajectory(wait=True)
            for i in range(5):
                recorder.record_step(create_dummy_step(timestep=i))
            recorder.save(wait=True, timeout=None)

            final = sorted(f for f in os.listdir(tmp_path) if f.endswith(".vla"))
            assert final == saved, final


def test_discard_last_refused_while_recording(tmp_path):
    """Discard must not delete anything while an episode is in progress."""
    middleware = "Thread"
    recorder_cfg = {
        "path": str(tmp_path),
        "freq": 100,
        "video_codec": "rawvideo",
        "start_recording": False,
    }
    recorder_server, recorder_client = make_node(middleware, "data", "Recorder", recorder_cfg, package="rio")

    with ServerManager(middleware, [recorder_server]):
        with recorder_client() as recorder:
            recorder.new_trajectory(wait=True)
            for i in range(5):
                recorder.record_step(create_dummy_step(timestep=i))
            recorder.save(wait=True, timeout=None)
            kept = sorted(f for f in os.listdir(tmp_path) if f.endswith(".vla"))

            # Start a new episode, then try to discard mid-recording.
            recorder.new_trajectory(wait=True)
            recorder.discard_last()
            time.sleep(0.5)

            assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".vla")) == kept
            recorder.save(wait=True, timeout=None)

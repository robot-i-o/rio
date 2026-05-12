# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the data formatters."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import robodm

from rio.data import LeRobotFormatter

lerobot = pytest.importorskip("lerobot", reason="lerobot not installed")

rng = np.random.default_rng()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after test
    if Path(temp_path).exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_robodm_path(temp_dir):
    """Create a sample robodm trajectory for testing."""
    output_path = str(Path(temp_dir) / "test_trajectory.vla")

    # Create trajectory in write mode
    traj = robodm.Trajectory(
        path=output_path,
        mode="w",
        video_codec="libx264",
    )

    # Simulate 100 timesteps at 30 Hz
    num_steps = 100
    fps = 30

    for i in range(num_steps):
        timestamp = i * (1000.0 / fps)  # milliseconds

        # Simulated robot state
        joint_positions = rng.standard_normal(7).astype(np.float32)
        joint_velocities = rng.standard_normal(7).astype(np.float32)
        ee_pose = rng.standard_normal(7).astype(np.float32)
        gripper_position = rng.random(1).astype(np.float32)
        camera_rgb = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        action = rng.standard_normal(7).astype(np.float32)

        # Add data to trajectory
        traj.add("observation/joint_positions", joint_positions, timestamp=timestamp, time_unit="ms")
        traj.add("observation/joint_velocities", joint_velocities, timestamp=timestamp, time_unit="ms")
        traj.add("observation/ee_pose", ee_pose, timestamp=timestamp, time_unit="ms")
        traj.add("observation/gripper_position", gripper_position, timestamp=timestamp, time_unit="ms")
        traj.add("observation/images/camera_rgb", camera_rgb, timestamp=timestamp, time_unit="ms")
        traj.add("action", action, timestamp=timestamp, time_unit="ms")

    traj.close()

    return output_path


@pytest.fixture
def lerobot_output_path(temp_dir):
    """Provide path for LeRobot output."""
    return str(Path(temp_dir) / "lerobot_dataset")


@pytest.mark.integration
class TestLeRobotFormatter:
    """Tests for LeRobotFormatter class."""

    def test_formatter_initialization(self, sample_robodm_path, lerobot_output_path):
        """Test that formatter can be initialized with valid paths."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            verbose=False,
        )

        assert formatter.robodm_path == Path(sample_robodm_path)
        assert formatter.output_path == Path(lerobot_output_path)
        assert formatter.repo_id == "test/sample_dataset"
        assert formatter.fps == 30
        assert formatter.robot_type == "test_robot"

    def test_formatter_invalid_path(self, lerobot_output_path):
        """Test that formatter raises error with invalid robodm path."""
        with pytest.raises(FileNotFoundError):
            LeRobotFormatter(
                robodm_path="/nonexistent/path.vla",
                output_path=lerobot_output_path,
                repo_id="test/dataset",
            )

    def test_convert_creates_directory_structure(self, sample_robodm_path, lerobot_output_path):
        """Test that conversion creates the expected directory structure."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            verbose=False,
        )

        formatter.convert()

        output_dir = Path(lerobot_output_path)

        # Check that main directories exist
        assert (output_dir / "data").exists()
        assert (output_dir / "meta").exists()
        assert (output_dir / "meta" / "episodes").exists()

    def test_convert_creates_metadata_files(self, sample_robodm_path, lerobot_output_path):
        """Test that conversion creates all required metadata files."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            verbose=False,
        )

        formatter.convert()

        output_dir = Path(lerobot_output_path)

        # Check metadata files
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "stats.json").exists()
        assert (output_dir / "meta" / "tasks.parquet").exists()

    def test_info_json_content(self, sample_robodm_path, lerobot_output_path):
        """Test that info.json contains correct metadata."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            verbose=False,
        )

        formatter.convert()

        info_path = Path(lerobot_output_path) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == "test_robot"
        assert info["fps"] == 30
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 100
        assert "features" in info
        assert len(info["features"]) == 6  # All features from sample data

    def test_stats_json_content(self, sample_robodm_path, lerobot_output_path):
        """Test that stats.json contains statistics for features."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            verbose=False,
        )

        formatter.convert()

        stats_path = Path(lerobot_output_path) / "meta" / "stats.json"
        with open(stats_path) as f:
            stats = json.load(f)

        # Check that stats exist for non-video features
        assert "observation/joint_positions" in stats
        assert "action" in stats

        # Check that each stat has required fields
        for feature_stats in stats.values():
            assert "mean" in feature_stats
            assert "std" in feature_stats
            assert "min" in feature_stats
            assert "max" in feature_stats

    def test_data_parquet_files(self, sample_robodm_path, lerobot_output_path):
        """Test that data parquet files are created and contain correct data."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            verbose=False,
        )

        formatter.convert()

        data_dir = Path(lerobot_output_path) / "data" / "chunk-000"
        assert data_dir.exists()

        parquet_files = list(data_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        # Read the parquet file
        table = pq.read_table(parquet_files[0])

        # Check that it has the expected columns
        assert "timestamp" in table.column_names
        assert "episode_index" in table.column_names
        assert "frame_index" in table.column_names
        assert "observation/joint_positions" in table.column_names
        assert "action" in table.column_names

        # Check that it has the correct number of rows
        assert len(table) == 100

    def test_episode_metadata(self, sample_robodm_path, lerobot_output_path):
        """Test that episode metadata is created correctly."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            verbose=False,
        )

        formatter.convert()

        episodes_dir = Path(lerobot_output_path) / "meta" / "episodes" / "chunk-000"
        assert episodes_dir.exists()

        episode_files = list(episodes_dir.glob("*.parquet"))
        assert len(episode_files) > 0

        # Read the episode metadata
        table = pq.read_table(episode_files[0])

        assert "episode_index" in table.column_names
        assert "length" in table.column_names
        assert "task_index" in table.column_names

        # Check episode values
        df = table.to_pandas()
        assert df.iloc[0]["episode_index"] == 0
        assert df.iloc[0]["length"] == 100
        assert df.iloc[0]["task_index"] == 0

    def test_video_key_detection(self, sample_robodm_path, lerobot_output_path):
        """Test that video keys are auto-detected correctly."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            video_keys=None,  # Auto-detect
            verbose=False,
        )

        formatter.convert()

        info_path = Path(lerobot_output_path) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        # Check that camera_rgb was detected as a video key
        assert "observation/images/camera_rgb" in info["video_keys"]

    def test_explicit_video_keys(self, sample_robodm_path, lerobot_output_path):
        """Test that explicitly provided video keys are used."""
        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task="Test task",
            video_keys=["observation/images/camera_rgb"],
            verbose=False,
        )

        formatter.convert()

        info_path = Path(lerobot_output_path) / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        assert info["video_keys"] == ["observation/images/camera_rgb"]

    def test_tasks_parquet(self, sample_robodm_path, lerobot_output_path):
        """Test that tasks.parquet is created correctly."""
        task_description = "Custom test task"

        formatter = LeRobotFormatter(
            robodm_path=sample_robodm_path,
            output_path=lerobot_output_path,
            repo_id="test/sample_dataset",
            fps=30,
            robot_type="test_robot",
            task=task_description,
            verbose=False,
        )

        formatter.convert()

        tasks_path = Path(lerobot_output_path) / "meta" / "tasks.parquet"
        table = pq.read_table(tasks_path)

        assert "task_index" in table.column_names
        assert "task" in table.column_names

        df = table.to_pandas()
        assert df.iloc[0]["task_index"] == 0
        assert df.iloc[0]["task"] == task_description

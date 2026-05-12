"""Example demonstrating robodm to RLDS format conversion."""

import shutil
from pathlib import Path

import numpy as np
import robodm

# RLDS formatter requires optional dependencies
try:
    from rio.data import RLDSFormatter

    RLDS_AVAILABLE = True
except ImportError:
    RLDS_AVAILABLE = False
    print("RLDS formatter not available. Install with: pip install rio[rlds]")

rng = np.random.default_rng()


def create_sample_robodm_dataset(output_path: str):
    """
    Create a sample robodm trajectory for testing.

    This creates a simple dataset with:
    - Robot joint positions
    - End effector pose
    - Gripper state
    - Camera observations (simulated)
    """
    print(f"Creating sample robodm dataset at {output_path}")

    # Create trajectory in write mode
    traj = robodm.Trajectory(
        path=output_path,
        mode="w",
        video_codec="libx264",
    )

    # Simulate 50 timesteps at 30 Hz
    num_steps = 50
    fps = 30

    for i in range(num_steps):
        timestamp = i * (1000.0 / fps)  # milliseconds

        # Simulated robot state
        joint_positions = rng.standard_normal(7).astype(np.float32)
        ee_pose = rng.standard_normal(7).astype(np.float32)
        gripper_position = rng.random(1).astype(np.float32)

        # Simulated camera image (small for testing)
        camera_rgb = (rng.random((64, 64, 3)) * 255).astype(np.uint8)

        # Action
        action = rng.standard_normal(7).astype(np.float32)

        # Add data to trajectory
        traj.add("observation/joint_positions", joint_positions, timestamp=timestamp, time_unit="ms")
        traj.add("observation/ee_pose", ee_pose, timestamp=timestamp, time_unit="ms")
        traj.add("observation/gripper_position", gripper_position, timestamp=timestamp, time_unit="ms")
        traj.add("observation/images/camera", camera_rgb, timestamp=timestamp, time_unit="ms")
        traj.add("action", action, timestamp=timestamp, time_unit="ms")

    # Save the trajectory
    traj.close()
    print(f"✓ Created robodm dataset with {num_steps} timesteps")


def test_rlds_formatter():
    """Test the RLDS formatter with sample data."""

    if not RLDS_AVAILABLE:
        print("\n" + "=" * 60)
        print("RLDS formatter not available")
        print("=" * 60)
        print("\nTo use the RLDS formatter, install the optional dependencies:")
        print("  pip install rio[rlds]")
        print("\nThis will install:")
        print("  - tensorflow>=2.11.0")
        print("  - tensorflow-datasets>=4.9.0")
        return

    # Paths
    robodm_path = "/tmp/test_rlds_trajectory.vla"
    rlds_path = "/tmp/test_rlds_dataset/robot_demo"  # Point directly to dataset directory

    # Clean up any existing files
    if Path(robodm_path).exists():
        Path(robodm_path).unlink()
    if Path("/tmp/test_rlds_dataset").exists():
        shutil.rmtree("/tmp/test_rlds_dataset")

    print("\n" + "=" * 60)
    print("Testing RLDS Formatter")
    print("=" * 60 + "\n")

    # Step 1: Create sample robodm dataset
    create_sample_robodm_dataset(robodm_path)

    # Step 2: Convert to RLDS format
    print("\nConverting to RLDS format...")
    formatter = RLDSFormatter(
        robodm_path=robodm_path,
        output_path=rlds_path,
        dataset_name="robot_demo",
        fps=30,
        robot_type="test_robot",
        task_description="Sample robot manipulation task",
        language_instruction="Pick up the object",
        compress_images=True,
        verbose=True,
    )

    formatter.convert()

    # Step 3: Load dataset using tfds.builder
    print(f"\n{'=' * 60}")
    print("Loading dataset with tfds.builder...")
    print("=" * 60 + "\n")

    try:
        import tensorflow_datasets as tfds

        # Method 1: Load using builder_from_directory (already works)
        print("Method 1: Using tfds.builder_from_directory()")
        builder = tfds.builder_from_directory(rlds_path)

        print(f"✓ Successfully loaded builder: {builder.info.name}")
        print(f"  → Version: {builder.info.version}")
        print(f"  → Description: {builder.info.description}")

        # Method 2: Load using tfds.builder() with data_dir
        print("\nMethod 2: Using tfds.builder() with data_dir")

        # Add the dataset directory to Python path so the builder class can be imported
        import sys

        if str(Path(rlds_path).parent) not in sys.path:
            sys.path.insert(0, str(Path(rlds_path).parent))

        # Register the dataset by importing it
        __import__(Path(rlds_path).name)

        # Now load with tfds.builder
        builder2 = tfds.builder("robot_demo", data_dir=str(Path(rlds_path).parent), version="1.0.1")

        print("✓ Successfully loaded with tfds.builder()")
        print(f"  → Name: {builder2.info.name}")
        print(f"  → Version: {builder2.info.version}")

        # Display dataset info
        print("\n  Dataset Info:")
        print(f"  → Splits: {list(builder.info.splits.keys())}")
        if "train" in builder.info.splits:
            print(f"  → Train examples: {builder.info.splits['train'].num_examples}")

        # Load the dataset
        print("\n  Loading train split...")
        ds = builder.as_dataset(split="train")

        # Iterate through a few examples
        print("\n  Inspecting dataset examples:")
        for i, example in enumerate(ds.take(3)):
            print(f"\n  Step {i + 1}:")
            print(f"    → is_first: {example['is_first'].numpy()}")
            print(f"    → is_last: {example['is_last'].numpy()}")
            print(f"    → is_terminal: {example['is_terminal'].numpy()}")
            print(f"    → reward: {example['reward'].numpy()}")
            print(f"    → language_instruction: {example['language_instruction'].numpy().decode()}")

            # Show observation dict
            if "observation" in example:
                obs = example["observation"]
                obs_keys = list(obs.keys())
                print(f"    → Observation: {obs_keys[:5]}")
                if obs_keys:
                    for key in obs_keys[:3]:
                        val = obs[key]
                        if hasattr(val, "shape"):
                            print(f"       - {key}: shape={val.shape}, dtype={val.dtype}")

            # Show action_dict keys
            if "action_dict" in example:
                action_dict = example["action_dict"]
                action_dict_keys = list(action_dict.keys())
                if action_dict_keys:
                    print(f"    → Action dict: {action_dict_keys[:5]}")

            # Show action
            if "action" in example:
                print(f"    → Action: shape={example['action'].shape}, dtype={example['action'].dtype}")

        # Count total steps
        total_steps = sum(1 for _ in ds)
        print(f"\n✓ Dataset contains {total_steps} total steps")

        print("\n" + "=" * 60)
        print("Test complete! ✓")
        print("=" * 60)
        print(f"\nRLDS dataset created at: {rlds_path}")
        print("\nLoad with either:")
        print(f"  1. tfds.builder_from_directory('{rlds_path}')")
        print(f"  2. tfds.builder('robot_demo', data_dir='{Path(rlds_path).parent}', version='1.0.1')")

    except Exception as e:
        print(f"✗ Error loading dataset with tfds.builder: {e}")
        import traceback

        traceback.print_exc()
        print("\nFalling back to manual verification...")

        # Fallback to manual verification
        print(f"\nChecking directory structure at {rlds_path}:")

        # List what was created
        import os

        for root, _dirs, files in os.walk(Path(rlds_path).parent):
            level = root.replace(str(Path(rlds_path).parent), "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

        expected_files = [
            "dataset_info.json",
            "robot_demo-train.tfrecord-00000-of-00001",
            "1.0.1/features.json",
        ]

        for file_name in expected_files:
            file_path = Path(rlds_path) / file_name
            if file_path.exists():
                print(f"✓ {file_name} exists")
            else:
                print(f"✗ {file_name} missing")

    return rlds_path


if __name__ == "__main__":
    test_rlds_formatter()

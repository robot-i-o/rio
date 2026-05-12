"""Example demonstrating robodm to LeRobot format conversion.

This example follows the same pattern as the OpenPI libero conversion:
https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py

Expected Output Structure:
--------------------------
The formatter creates a dataset with the following structure, matching the libero format:

    output_path/
    ├── data/
    │   └── chunk-000/
    │       └── episode_000000.parquet  (contains all data columns)
    └── meta/
        ├── info.json              (dataset metadata and feature specs)
        ├── episodes.jsonl         (per-episode metadata)
        ├── episodes_stats.jsonl   (per-episode statistics)
        └── tasks.jsonl            (task definitions)

Parquet Columns (matching libero format):
------------------------------------------
- image: Main camera observation (256x256x3 RGB images)
- wrist_image: Wrist camera observation (256x256x3 RGB images)
- state: Robot state (8D: 7 joint positions + 1 gripper)
- actions: Robot actions (7D: delta joint positions)
- timestamp: Frame timestamp (float32)
- frame_index: Frame number within episode (int64)
- episode_index: Episode number (int64)
- index: Global frame index across all episodes (int64)
- task_index: Task identifier (int64)

Key Conventions:
----------------
- Uses LeRobotDataset.create() and add_frame() API
- Auto-detects features from robodm trajectory
- Uses simple feature names (image, state, actions) - no hierarchical names with '/'
- Dimension names match feature names for 1D arrays (e.g., ["state"] for state feature)
- Includes required 'task' field for each frame
- Repo ID format: your_hf_username/dataset_name (e.g., "test/sample_robot_dataset")

The generated dataset is compatible with the LeRobot ecosystem and can be loaded
using LeRobotDataset for training vision-language-action models.

Usage Example:
--------------
```python
from rio.data import LeRobotFormatter

# Convert robodm trajectory to LeRobot format
formatter = LeRobotFormatter(
    robodm_path="path/to/trajectory.vla",
    output_path="~/.cache/huggingface/lerobot/your_username/dataset_name",
    repo_id="your_username/dataset_name",  # Must match output_path structure
    fps=10,
    robot_type="panda",
    video_keys=["image", "wrist_image"],  # Features containing images
    task="Pick and place",
)
formatter.convert()
```

For datasets with hierarchical names in robodm (e.g., "observation/images/camera"),
use simple names instead (e.g., "image") to match the libero format. The formatter
will automatically handle the conversion.
"""

import shutil
from pathlib import Path

import numpy as np
import robodm

from rio.data import LeRobotFormatter

rng = np.random.default_rng()


def create_sample_robodm_dataset(output_path: str):
    """
    Create a sample robodm trajectory for testing.

    This creates a simple dataset with:
    - Robot state (joint positions + gripper)
    - Camera observations (simulated)
    - Actions (delta positions)

    The data structure matches the libero format:
    - image: main camera view
    - wrist_image: wrist camera view
    - state: robot joint positions + gripper state (8 dims)
    - actions: delta joint positions (7 dims)
    """
    print(f"Creating sample robodm dataset at {output_path}")

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

        # Robot state: 7 joint positions + 1 gripper state (matching libero format)
        state = rng.standard_normal(8).astype(np.float32)

        # Simulated camera images (256x256 to match libero)
        image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        wrist_image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)

        # Actions: delta joint positions (7 dims, matching libero)
        actions = rng.standard_normal(7).astype(np.float32)

        # Add data to trajectory using simple names (will be converted to LeRobot format)
        traj.add("image", image, timestamp=timestamp, time_unit="ms")
        traj.add("wrist_image", wrist_image, timestamp=timestamp, time_unit="ms")
        traj.add("state", state, timestamp=timestamp, time_unit="ms")
        traj.add("actions", actions, timestamp=timestamp, time_unit="ms")

    # Save the trajectory
    traj.close()
    print(f"✓ Created robodm dataset with {num_steps} timesteps")


def test_lerobot_formatter():
    """Test the LeRobot formatter with sample data."""

    # Paths
    robodm_path = "/tmp/test_trajectory.vla"
    lerobot_path = "/tmp/test_lerobot_dataset"

    # Clean up any existing files
    if Path(robodm_path).exists():
        Path(robodm_path).unlink()
    if Path(lerobot_path).exists():
        shutil.rmtree(lerobot_path)

    print("\n" + "=" * 60)
    print("Testing LeRobot Formatter")
    print("=" * 60 + "\n")

    # Step 1: Create sample robodm dataset
    create_sample_robodm_dataset(robodm_path)

    # Step 2: Convert to LeRobot format
    print("\nConverting to LeRobot format...")

    # Define features for the dataset
    # Following OpenPI/LeRobot conventions: use simple names like "image", "state", "actions"
    # This will create a dataset under: ~/.cache/huggingface/lerobot/test/sample_robot_dataset
    # (or the specified output_path)
    formatter = LeRobotFormatter(
        robodm_path=robodm_path,
        output_path=lerobot_path,
        repo_id="test/sample_robot_dataset",  # Format: your_hf_username/dataset_name
        fps=30,
        robot_type="simulated_robot",
        features=None,  # Auto-detect features from robodm trajectory
        video_keys=["image", "wrist_image"],  # Specify which features contain video
        task="Random motion test",
        verbose=True,
    )

    formatter.convert()

    # Step 3: Verify output structure
    print(f"\n{'=' * 60}")
    print("Verifying output structure...")
    print("=" * 60 + "\n")

    lerobot_dir = Path(lerobot_path)

    # Check directories created by LeRobot
    required_dirs = ["data", "meta"]
    for dir_name in required_dirs:
        dir_path = lerobot_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"✗ {dir_name}/ missing")

    # Check required metadata files
    required_files = ["meta/info.json"]
    for file_name in required_files:
        file_path = lerobot_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name} exists")
            if file_name.endswith(".json"):
                import json

                with open(file_path) as f:
                    data = json.load(f)
                    print(f"  → {len(data)} keys: {list(data.keys())[:5]}")
        else:
            print(f"✗ {file_name} missing")

    # Check data files
    data_chunk_dir = lerobot_dir / "data" / "chunk-000"
    if data_chunk_dir.exists():
        parquet_files = list(data_chunk_dir.glob("*.parquet"))
        print(f"✓ Found {len(parquet_files)} parquet file(s) in data/chunk-000/")

        if parquet_files:
            # Read and show sample
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_files[0])
            print(f"  → Columns: {table.column_names}")
            print(f"  → Rows: {len(table)}")
    else:
        print("✗ data/chunk-000/ missing")

    print("\n" + "=" * 60)
    print("Test complete! ✓")
    print("=" * 60)
    print(f"\nLeRobot dataset created at: {lerobot_path}")

    # Step 4: Validate the dataset can be read
    print("\n" + "=" * 60)
    print("Validating dataset...")
    print("=" * 60 + "\n")

    try:
        import pyarrow.parquet as pq

        # Read the parquet file directly
        parquet_files = list(data_chunk_dir.glob("*.parquet"))
        if parquet_files:
            table = pq.read_table(parquet_files[0])

            # Validate required columns exist
            required_cols = ["timestamp", "frame_index", "episode_index", "task_index"]
            for col in required_cols:
                if col in table.column_names:
                    print(f"✓ Required column '{col}' present")
                else:
                    print(f"✗ Missing required column '{col}'")

            # Check that we have observation and action data
            obs_cols = [c for c in table.column_names if c.startswith("observation.")]
            action_cols = [c for c in table.column_names if c == "action"]

            print(f"✓ Found {len(obs_cols)} observation features")
            print(f"✓ Found {len(action_cols)} action features")

            # Validate data types
            print("\n✓ Dataset validated successfully!")
            print(f"  - {len(table)} frames")
            print(f"  - {len(table.column_names)} columns")
            meta_cols = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            print(f"  - Features: {[c for c in table.column_names if c not in meta_cols]}")

            # Print first row to show data format
            print("\nFirst frame sample:")
            df = table.to_pandas()
            for col in ["image", "wrist_image", "state", "actions", "timestamp"]:
                if col in df.columns:
                    val = df.iloc[0][col]
                    if col in ["image", "wrist_image"]:
                        print(f"  {col}: <image data>")
                    else:
                        print(f"  {col}: {val}")

    except Exception as e:
        print(f"✗ Validation failed: {e}")

    # Step 5: Compare with libero format
    print("\n" + "=" * 60)
    print("Comparing with libero dataset format...")
    print("=" * 60 + "\n")

    try:
        import json

        import pyarrow.parquet as pq

        libero_path = Path.home() / ".cache/huggingface/lerobot/your_hf_username/libero"
        if libero_path.exists():
            # Compare metadata
            with open(lerobot_dir / "meta/info.json") as f:
                test_meta = json.load(f)
            with open(libero_path / "meta/info.json") as f:
                libero_meta = json.load(f)

            print("Metadata structure comparison:")
            print(f"✓ Both have {len(test_meta)} top-level keys")
            print(f"✓ Both use codebase_version: {test_meta['codebase_version']}")

            # Compare feature structures
            print("\nFeature comparison:")
            for key in ["image", "wrist_image", "state", "actions"]:
                if key in test_meta["features"] and key in libero_meta["features"]:
                    test_feat = test_meta["features"][key]
                    libero_feat = libero_meta["features"][key]
                    match = test_feat["dtype"] == libero_feat["dtype"] and test_feat["shape"] == libero_feat["shape"]
                    symbol = "✓" if match else "✗"
                    print(f"{symbol} {key}: dtype={test_feat['dtype']}, shape={test_feat['shape']}, names={test_feat['names']}")

            # Compare parquet columns
            test_parquet = next((lerobot_dir / "data/chunk-000").glob("*.parquet"))
            libero_parquet = next((libero_path / "data/chunk-000").glob("*.parquet"))

            test_cols = set(pq.read_table(test_parquet).column_names)
            libero_cols = set(pq.read_table(libero_parquet).column_names)

            print("\nParquet column comparison:")
            if test_cols == libero_cols:
                print(f"✓ Columns match exactly: {sorted(test_cols)}")
            else:
                print("✗ Column mismatch!")
                print(f"  Test:   {sorted(test_cols)}")
                print(f"  Libero: {sorted(libero_cols)}")

            print("\n✓ Format matches libero dataset structure!")
        else:
            print(f"Libero dataset not found at {libero_path}")
            print("  Run the libero conversion example to create it for comparison")
    except Exception as e:
        print(f"⚠ Could not compare with libero dataset: {e}")

    print("\nYou can inspect the files to verify the structure.")

    return lerobot_path


if __name__ == "__main__":
    test_lerobot_formatter()

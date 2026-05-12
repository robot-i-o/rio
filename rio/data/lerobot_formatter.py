# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""LeRobot dataset formatter for converting robodm trajectories."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import robodm

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
from loguru import logger

from ._formatter import Formatter


class LeRobotFormatter(Formatter):
    """
    Formatter to convert robodm trajectory files to LeRobot dataset format.

    The LeRobot format is created using the LeRobotDataset API which handles:
    - Creating the required directory structure (data/, meta/, videos/)
    - Generating metadata files (info.json, stats.json)
    - Writing Parquet files for observations and actions
    - Managing video encoding and storage

    This formatter extracts data from robodm trajectories and adds frames
    to the LeRobot dataset using the `add_frame()` and `save_episode()` API.

    Note: LeRobot feature names cannot contain '/'. The formatter will automatically
    convert robodm feature names like "observation/images/camera" to "observation.images.camera".
    """

    def __init__(
        self,
        robodm_path: str | Path,
        output_path: str | Path,
        repo_id: str,
        fps: int = 30,
        robot_type: str = "unknown",
        features: dict[str, dict[str, Any]] | None = None,
        video_keys: list[str] | None = None,
        feature_mapping: dict[str, str] | None = None,
        feature_transforms: dict[str, callable] | None = None,
        verbose: bool = False,
        only_mapped_keys: bool = True,
        override_existing: bool = False,
        task: str | None = None,
    ):
        """
        Initialize the LeRobot formatter.

        Args:
            robodm_path: Path to the robodm trajectory file (.vla) or directory containing multiple .vla files
            output_path: Root directory for the LeRobot dataset (will be created if doesn't exist)
            repo_id: Repository ID for the dataset (e.g., "user/dataset_name")
            fps: Frames per second of the dataset
            robot_type: Type of robot used for data collection
            features: Dictionary defining the dataset features. Format:
                {
                    "feature_name": {
                        "dtype": "image" | "float32" | "int64" etc.,
                        "shape": (height, width, channels),  # for images
                        "names": ["dim1", "dim2", ...],  # dimension names
                    }
                }
                If None, will auto-detect from robodm trajectory.
                Feature names should be the final LeRobot names (after any mapping).
            video_keys: List of feature names that contain video/image data.
                       If None, will auto-detect from features with dtype="image".
                       Use the final LeRobot feature names (after mapping).
            feature_mapping: Dictionary mapping robodm feature names to LeRobot names.
                           Format: {"robodm_name": "lerobot_name", ...}
                           Use this to rename features during conversion (e.g., {"camera1": "image"}).
                           If None, feature names are used as-is.
            feature_transforms: Dictionary of transformation functions for features.
                              Format: {"feature_name": transform_func, ...}
                              Each function takes the raw feature value and returns the transformed value.
                              Use this to modify data (e.g., change dtype, slice dimensions, resize images).
            task: Task description/name for the dataset
            verbose: If True, print progress information
            only_mapped_keys: If True, only include features that are in the feature_mapping
            override_existing: If True, overwrite existing dataset if it exists
        """
        super().__init__(robodm_path, output_path, verbose)
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot package is not installed. Please install it to use LeRobotFormatter.")

        self.repo_id = repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.features = features
        self.video_keys = video_keys
        self.feature_mapping = feature_mapping or {}
        self.feature_transforms = feature_transforms or {}
        self.only_mapped_keys = only_mapped_keys
        self.override_existing = override_existing
        self.task = task

        # Configure logger level based on verbose flag
        if not verbose:
            logger.disable("rio.data.lerobot_formatter")

        # Will be initialized during conversion
        self.dataset: LeRobotDataset | None = None

        # Mapping from robodm feature names to LeRobot names (combining user mapping + '/' conversion)
        self.feature_name_mapping: dict[str, str] = {}

        # List of trajectory files to process
        self.trajectory_files: list[Path] = []

    def _convert_feature_name(self, robodm_name: str) -> str:
        """
        Convert robodm feature name to LeRobot-compatible name.

        This applies:
        1. User-provided feature mapping (if specified)
        2. Conversion of '/' to '.' for LeRobot compatibility

        Args:
            robodm_name: Original feature name from robodm

        Returns:
            LeRobot-compatible feature name
        """
        # First apply user mapping if specified
        if robodm_name in self.feature_mapping:
            return self.feature_mapping[robodm_name]

        # Otherwise just convert '/' to '.'
        return robodm_name.replace("/", ".")

    def _process_data(self) -> dict[str, Any]:
        """
        Process robodm trajectory data and create LeRobot dataset.

        This method now handles multiple trajectory files if robodm_path is a directory.

        Returns:
            Dictionary containing dataset metadata
        """
        # Determine trajectory files to process
        robodm_path = Path(self.robodm_path)

        if robodm_path.is_dir():
            # Find all .vla files in the directory
            self.trajectory_files = sorted(robodm_path.glob("*.vla"))
            if not self.trajectory_files:
                raise ValueError(f"No .vla files found in directory: {robodm_path}")
            logger.info(f"Found {len(self.trajectory_files)} .vla files in directory: {robodm_path}")
        elif robodm_path.is_file():
            # Single file
            self.trajectory_files = [robodm_path]
            logger.info(f"Processing single file: {robodm_path}")
        else:
            raise ValueError(f"Path does not exist: {robodm_path}")

        # Load the first trajectory for feature detection
        first_trajectory = robodm.Trajectory(path=str(self.trajectory_files[0]), mode="r")
        trajectory_data = first_trajectory.load()

        # Extract all feature names (robodm format with '/')
        robodm_feature_names = list(trajectory_data.keys())
        logger.debug(f"Found robodm features: {robodm_feature_names}")

        if self.only_mapped_keys and self.feature_mapping is not None:
            mapped_feature_names = set(self.feature_mapping.keys())
            missing_feature_names = mapped_feature_names - set(robodm_feature_names)
            if missing_feature_names:
                logger.warning(
                    f"Mapped feature names not found in trajectory: {missing_feature_names}\n"
                    f" The available features in the dataset are: {robodm_feature_names}"
                )

            selected_robodm_feature_names = [name for name in robodm_feature_names if name in mapped_feature_names]
            logger.debug(f"Using only mapped features: {selected_robodm_feature_names}")
        else:
            selected_robodm_feature_names = robodm_feature_names

        # Create feature name mapping
        for robodm_name in selected_robodm_feature_names:
            lerobot_name = self._convert_feature_name(robodm_name)
            self.feature_name_mapping[robodm_name] = lerobot_name

        logger.debug(f"Feature name mapping: {self.feature_name_mapping}")

        # Auto-detect features if not provided
        if self.features is None:
            self.features = self._detect_features(trajectory_data, selected_robodm_feature_names)
            logger.debug(f"Auto-detected features: {list(self.features.keys())}")
        # Note: If features are provided, they should already use final LeRobot names

        # Auto-detect video keys from features with dtype="image"
        if self.video_keys is None:
            self.video_keys = [name for name, spec in self.features.items() if spec.get("dtype") == "image"]
            logger.debug(f"Auto-detected video keys: {self.video_keys}")
        else:
            # Convert video_keys to LeRobot format
            self.video_keys = [self._convert_feature_name(k) for k in self.video_keys]
            logger.debug(f"Converted video keys: {self.video_keys}")

        # Check if override_existing is set and dataset exists
        dataset_path = Path(self.output_path) if self.output_path is not None else HF_LEROBOT_HOME / self.repo_id
        if dataset_path.exists():
            logger.warning(f"Dataset already exists at {dataset_path}")
            # Ask the user on whether to delete or not.
            answer = (
                input(f"Dataset already exists at {dataset_path} \n Do you want to delete the existing dataset? (y/N): ")
                if not self.override_existing
                else "y"
            )
            if answer.lower() == "y":
                shutil.rmtree(dataset_path)
                logger.info(f"Deleted existing dataset at {dataset_path}")
            else:
                logger.info("Aborting conversion.")
                return {}

        # Create LeRobot dataset
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
            root=self.output_path,
            image_writer_threads=30,
            image_writer_processes=15,
        )

        logger.info(f"Created LeRobot dataset: {self.repo_id}")

        # Process each trajectory file as an episode
        total_frames = 0
        for episode_idx, trajectory_file in enumerate(self.trajectory_files):
            logger.info(f"Processing episode {episode_idx + 1}/{len(self.trajectory_files)}: {trajectory_file.name}")

            # Load trajectory
            traj = robodm.Trajectory(path=str(trajectory_file), mode="r")
            traj_data = traj.load()
            traj_data = self._enhance_trajectory_data(traj_data)

            # Determine number of frames
            first_feature_data = traj_data[selected_robodm_feature_names[0]]
            num_frames = len(first_feature_data) if hasattr(first_feature_data, "__len__") else 1
            total_frames += num_frames

            logger.debug(f"Episode {episode_idx + 1}: {num_frames} frames")

            # Add frames to the dataset
            for frame_idx in range(num_frames):
                frame_data = self._get_frame_at_index(traj_data, selected_robodm_feature_names, frame_idx)
                # Add task field required by LeRobot
                if "task" not in frame_data:
                    if self.task is None:
                        raise ValueError(
                            "LeRobot dataset requires a 'task' field in each frame or to be passed as a"
                            " parameter. Please include it in the feature mapping or transforms."
                        )
                    else:
                        frame_data["task"] = self.task

                self.dataset.add_frame(frame_data)

            # Save the episode
            self.dataset.save_episode()
            logger.debug(f"Saved episode {episode_idx + 1} to LeRobot dataset")

        return {
            "total_frames": total_frames,
            "total_episodes": len(self.trajectory_files),
            "features": self.features,
        }

    def _enhance_trajectory_data(self, trajectory_data: dict) -> dict:
        """
        Enhance trajectory data with any additional processing if needed.

        This method can be overridden to add custom processing steps.

        Args:
            trajectory_data: Original trajectory data loaded from robodm

        Returns:
            Enhanced trajectory data
        """
        # For now, just return the data as-is
        return trajectory_data

    def _detect_features(self, trajectory_data: dict, feature_names: list[str]) -> dict[str, dict[str, Any]]:
        """
        Auto-detect feature specifications from trajectory data.

        Args:
            trajectory_data: Dictionary of loaded trajectory data
            feature_names: List of all robodm feature names (may contain '/')

        Returns:
            Dictionary of feature specifications in LeRobot format (with '.' in names)
        """
        features = {}

        for robodm_name in feature_names:
            # Convert to LeRobot-compatible name
            lerobot_name = self._convert_feature_name(robodm_name)

            try:
                # Sample first frame to infer feature spec
                feature_data = trajectory_data[robodm_name]
                sample = feature_data[0] if hasattr(feature_data, "__getitem__") else feature_data

                if isinstance(sample, np.ndarray):
                    # Apply transform if specified
                    if lerobot_name in self.feature_transforms:
                        sample = self.feature_transforms[lerobot_name](sample)

                    # Determine if it's an image or array data
                    is_image = False
                    if len(sample.shape) == 3:
                        # Check for image-like shape (H, W, C) or (C, H, W)
                        if sample.shape[-1] in [1, 3, 4]:
                            is_image = True
                        elif sample.shape[0] in [1, 3, 4]:
                            is_image = True
                            # Transpose to (H, W, C) format for LeRobot
                            sample = np.transpose(sample, (1, 2, 0))

                    if is_image:
                        features[lerobot_name] = {
                            "dtype": "image",
                            "shape": sample.shape,
                            "names": ["height", "width", "channel"],
                        }
                    else:
                        # Array data (e.g., joint positions, actions)
                        # Use the feature name itself for 1D arrays to match libero format
                        if len(sample.shape) == 1:
                            names = [lerobot_name]
                        else:
                            names = [f"dim_{i}" for i in range(len(sample.shape))]

                        features[lerobot_name] = {
                            "dtype": str(sample.dtype),
                            "shape": sample.shape,
                            "names": names,
                        }
                else:
                    # Scalar value
                    features[lerobot_name] = {
                        "dtype": type(sample).__name__,
                        "shape": (),
                        "names": [],
                    }
            except Exception as e:
                logger.warning(f"Could not detect feature spec for {robodm_name}: {e}")
                # Add minimal spec as fallback
                features[lerobot_name] = {
                    "dtype": "float32",
                    "shape": (),
                    "names": [],
                }

        return features

    def _get_frame_at_index(self, trajectory_data: dict, feature_names: list[str], frame_idx: int) -> dict[str, Any]:
        """
        Extract all feature values at a specific frame index.

        Args:
            trajectory_data: Dictionary of loaded trajectory data
            feature_names: List of all robodm feature names (may contain '/')
            frame_idx: Frame index to extract data for

        Returns:
            Dictionary mapping LeRobot feature names (with '.') to their values at the frame index
        """
        frame_data = {}

        for robodm_name in feature_names:
            # Convert to LeRobot name
            lerobot_name = self.feature_name_mapping[robodm_name]

            feature_data = trajectory_data[robodm_name]

            # Extract value at frame index
            if hasattr(feature_data, "__getitem__"):
                value = feature_data[frame_idx]
            else:
                value = feature_data

            # Apply transform if specified
            if lerobot_name in self.feature_transforms:
                value = self.feature_transforms[lerobot_name](value)

            # Convert to appropriate format for LeRobot
            if isinstance(value, np.ndarray):
                # Check if we need to transpose images
                if lerobot_name in self.video_keys and len(value.shape) == 3:
                    if value.shape[0] in [1, 3, 4]:  # (C, H, W) -> (H, W, C)
                        value = np.transpose(value, (1, 2, 0))
                if len(value.shape) == 1:
                    # convert to dataset sequence
                    pass

            frame_data[lerobot_name] = value

        return frame_data

    def _write_output(self, data: dict[str, Any]):
        """
        Write the converted data to disk.

        The LeRobotDataset API handles all file writing internally,
        so this method is minimal. The dataset is automatically saved
        when save_episode() is called.

        Args:
            data: Processed data dictionary (contains metadata only)
        """
        if self.dataset is None:
            logger.error("Dataset was not created during processing")
            return

        logger.info(f"LeRobot dataset saved to: {self.output_path}")
        logger.info(f"Total episodes: {data['total_episodes']}")
        logger.info(f"Total frames: {data['total_frames']}")
        logger.info(f"Features: {list(data['features'].keys())}")

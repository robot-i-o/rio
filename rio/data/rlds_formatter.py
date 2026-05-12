# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""RLDS dataset formatter for converting robodm trajectories."""

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import robodm
from loguru import logger

from ._formatter import Formatter

if TYPE_CHECKING:
    import tensorflow as tf

try:
    import tensorflow as tf

    RLDS_AVAILABLE = True
except ImportError:
    RLDS_AVAILABLE = False
    tf = None  # type: ignore


try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RLDSFormatter(Formatter):
    """
    Formatter to convert robodm trajectory files to RLDS (Robot Learning Dataset) format.

    RLDS is a format for storing robot learning datasets using TensorFlow's TFDS.
    The format organizes data into episodes, where each episode contains a sequence
    of steps with observations, actions, rewards, and other metadata.

    Structure:
    - Episodes: Top-level grouping of related steps
    - Steps: Individual timesteps containing observations, actions, etc.
    - Features: Typed data (images, vectors, scalars, etc.)

    This formatter handles:
    - Converting robodm timestamped data to RLDS episode/step structure
    - Mapping features to appropriate TensorFlow types
    - Creating TFRecord files with proper serialization
    - Generating dataset_info.json with metadata

    Install with: uv pip install rio[rlds]
    """

    def __init__(
        self,
        robodm_path: str | Path,
        output_path: str | Path,
        dataset_name: str,
        fps: int = 30,
        robot_type: str = "unknown",
        task_description: str = "Robot manipulation task",
        language_instruction: str = "",
        image_keys: list[str] | None = None,
        compress_images: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the RLDS formatter.

        Args:
            robodm_path: Path to the robodm trajectory file (.vla)
            output_path: Root directory for the RLDS dataset
            dataset_name: Name of the dataset
            fps: Frames per second of the dataset
            robot_type: Type of robot used for data collection
            task_description: Description of the task
            image_keys: List of feature names that contain image data.
                       If None, will auto-detect from robodm features.
            compress_images: If True, compress images as JPEG in the dataset
            verbose: If True, print progress information
        """
        if not RLDS_AVAILABLE:
            raise ImportError("RLDS dependencies not available. Install with: uv pip install rio[rlds]")

        super().__init__(robodm_path, output_path, verbose)

        self.dataset_name = dataset_name
        self.fps = fps
        self.robot_type = robot_type
        self.task_description = task_description
        self.language_instruction = language_instruction
        self.image_keys = image_keys
        self.compress_images = compress_images

        # Configure logger level based on verbose flag
        if not verbose:
            logger.disable("rio.data.rlds_formatter")

    def _process_data(self, trajectory: robodm.Trajectory) -> dict[str, Any]:
        """
        Process robodm trajectory data into RLDS-compatible format.

        Args:
            trajectory: Loaded robodm Trajectory object

        Returns:
            Dictionary containing processed data ready for writing
        """
        # Load all data from trajectory
        trajectory_data = trajectory.load()

        # Extract all feature names
        feature_names = list(trajectory_data.keys())

        logger.debug(f"Found features: {feature_names}")

        # Auto-detect image keys if not provided
        if self.image_keys is None:
            self.image_keys = self._detect_image_keys(trajectory_data, feature_names)
            logger.debug(f"Auto-detected image keys: {self.image_keys}")

        # Collect all data indexed by timestamp
        data_by_timestamp = defaultdict(dict)
        timestamps = set()

        # Read all features
        for feature_name in feature_names:
            feature_data = trajectory_data[feature_name]

            # Get timestamps for this feature
            if hasattr(feature_data, "timestamps"):
                feature_timestamps = feature_data.timestamps
            else:
                # Fallback: create synthetic timestamps
                num_frames = len(feature_data) if hasattr(feature_data, "__len__") else 1
                feature_timestamps = np.arange(num_frames) * (1000.0 / self.fps)

            # Store data by timestamp
            for i, ts in enumerate(feature_timestamps):
                timestamps.add(ts)
                data_by_timestamp[ts][feature_name] = feature_data[i] if hasattr(feature_data, "__getitem__") else feature_data

        # Sort timestamps
        sorted_timestamps = sorted(timestamps)

        # Build episode data using DROID schema
        steps = []
        for i, ts in enumerate(sorted_timestamps):
            step_data = data_by_timestamp[ts]

            # Organize into DROID step format
            step = {
                "is_first": i == 0,
                "is_last": False,
                "is_terminal": False,
                "language_instruction": self.language_instruction,
                "language_instruction_2": self.language_instruction,
                "language_instruction_3": self.language_instruction,
                "observation": {},
                "action_dict": {},
                "action": None,
                "discount": 1.0,
                "reward": 0.0,
            }

            # Map features to DROID schema
            for feature_name, feature_value in step_data.items():
                # Strip "observation/" prefix if present (since we'll add it back when writing)
                obs_key = feature_name.replace("observation/", "") if feature_name.startswith("observation/") else feature_name

                # Check if this is an action feature
                is_action_feature = not feature_name.startswith("observation/") and "action" in feature_name

                # All non-action features go into observations
                if not is_action_feature:
                    step["observation"][obs_key] = feature_value
                else:
                    # Handle action - store as main action if it's just "action"
                    if feature_name == "action":
                        step["action"] = np.array(feature_value, dtype=np.float64)
                    else:
                        # Other action features go into action_dict
                        action_key = feature_name.replace("action/", "")
                        step["action_dict"][action_key] = np.array(feature_value, dtype=np.float64)

            steps.append(step)

        # Mark last step
        if steps:
            steps[-1]["is_last"] = True
            steps[-1]["is_terminal"] = True
            steps[-1]["reward"] = 1.0  # Reward 1 on final step for demos

        # Build feature spec for TensorFlow
        feature_spec = self._build_feature_spec(trajectory_data, feature_names)

        # Episode metadata
        episode_metadata = {
            "recording_folderpath": str(Path(self.robodm_path).parent),
            "file_path": str(self.robodm_path),
        }

        return {
            "episodes": [
                {
                    "episode_metadata": episode_metadata,
                    "steps": steps,
                }
            ],
            "feature_spec": feature_spec,
            "feature_names": feature_names,
            "num_steps": len(steps),
        }

    def _detect_image_keys(self, trajectory_data: dict, feature_names: list[str]) -> list[str]:
        """
        Auto-detect which features contain image data.

        Args:
            trajectory_data: Dictionary of loaded trajectory data
            feature_names: List of all feature names

        Returns:
            List of feature names that contain image data
        """
        image_keys = []

        for feature_name in feature_names:
            try:
                # Sample first frame to check if it's image data
                feature_data = trajectory_data[feature_name]
                sample = feature_data[0] if hasattr(feature_data, "__getitem__") else feature_data

                # Check if it's a numpy array with image-like shape (H, W, C) or (C, H, W)
                if isinstance(sample, np.ndarray):
                    if len(sample.shape) == 3 and sample.shape[-1] in [1, 3, 4]:
                        image_keys.append(feature_name)
                    elif len(sample.shape) == 3 and sample.shape[0] in [1, 3, 4]:
                        image_keys.append(feature_name)
            except Exception:
                # Skip features that can't be read
                pass

        return image_keys

    def _build_feature_spec(self, trajectory_data: dict, feature_names: list[str]) -> dict[str, Any]:
        """
        Build TensorFlow feature specification.

        Args:
            trajectory_data: Dictionary of loaded trajectory data
            feature_names: List of all feature names

        Returns:
            Dictionary mapping feature names to TensorFlow dtypes and shapes
        """
        feature_spec = {}

        for feature_name in feature_names:
            try:
                feature_data = trajectory_data[feature_name]
                sample = feature_data[0] if hasattr(feature_data, "__getitem__") else feature_data

                # Determine TensorFlow dtype
                if isinstance(sample, np.ndarray):
                    dtype = sample.dtype
                    shape = sample.shape

                    # Map numpy dtype to TensorFlow dtype
                    if np.issubdtype(dtype, np.floating):
                        tf_dtype = tf.float32
                    elif np.issubdtype(dtype, np.integer):
                        if feature_name in self.image_keys:
                            tf_dtype = tf.uint8
                        else:
                            tf_dtype = tf.int32
                    else:
                        tf_dtype = tf.float32

                    feature_spec[feature_name] = {
                        "dtype": tf_dtype,
                        "shape": shape,
                        "is_image": feature_name in self.image_keys,
                    }
                elif isinstance(sample, (int, float)):
                    feature_spec[feature_name] = {
                        "dtype": tf.float32 if isinstance(sample, float) else tf.int32,
                        "shape": (),
                        "is_image": False,
                    }
            except Exception:
                # Default to float32 scalar for unknown features
                feature_spec[feature_name] = {
                    "dtype": tf.float32,
                    "shape": (),
                    "is_image": False,
                }

        return feature_spec

    def _write_output(self, data: dict[str, Any]):
        """
        Write the converted data to RLDS format on disk.

        This creates:
        - TFRecord files containing serialized episodes
        - dataset_info.json with TFDS-compatible metadata
        - features.json with feature specifications

        Args:
            data: Processed data dictionary
        """
        # Create output directory structure expected by TFDS
        # TFDS expects: data_dir/dataset_info.json and data_dir/{dataset_name}-{split}.tfrecord-*
        dataset_root = self.output_path
        dataset_root.mkdir(parents=True, exist_ok=True)

        # Also create version dir for features.json (for debugging)
        version_dir = dataset_root / "1.0.1"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write TFRecord files to root (alongside dataset_info.json)
        # TFDS expects: {dataset_name}-{split}.tfrecord-{shard}-of-{num_shards}
        tfrecord_path = dataset_root / f"{self.dataset_name}-train.tfrecord-00000-of-00001"
        self._write_tfrecords(data["episodes"], data["feature_spec"], tfrecord_path)

        logger.debug(f"Wrote TFRecord to {tfrecord_path}")

        # Also write to version directory for tfds.builder() compatibility
        version_tfrecord_path = version_dir / f"{self.dataset_name}-train.tfrecord-00000-of-00001"
        self._write_tfrecords(data["episodes"], data["feature_spec"], version_tfrecord_path)

        logger.debug(f"Wrote TFRecord to {version_tfrecord_path} (for tfds.builder compatibility)")

        # Build feature description for TFDS
        features_dict = self._build_tfds_features(data["feature_spec"])

        # Write TFDS-compatible dataset_info.json at root level
        dataset_info = {
            "name": self.dataset_name,
            "version": "1.0.1",
            "description": self.task_description,
            "citation": "",
            "features": features_dict,
            "splits": [
                {
                    "name": "train",
                    "numShards": 1,
                    "shardLengths": [str(data["num_steps"])],
                    "statistics": {"numExamples": str(data["num_steps"])},
                }
            ],
            "supervisedKeys": None,
            "moduleName": "rio.rlds_dataset",
            "redistributionInfo": {},
            "configName": "",
            "configDescription": "",
            "sizeInBytes": "0",
            "downloadSize": "0",
            "fileFormat": "tfrecord",
        }

        info_path = dataset_root / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        logger.debug(f"Wrote dataset_info.json to {info_path}")

        # Also write to version directory for tfds.builder() compatibility
        version_info_path = version_dir / "dataset_info.json"
        with open(version_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        logger.debug(f"Wrote dataset_info.json to {version_info_path}")

        # Write feature spec for debugging (also in version dir)
        features_path = version_dir / "features.json"
        # Convert TF dtypes to strings for JSON serialization
        serializable_spec = {}
        for key, spec in data["feature_spec"].items():
            serializable_spec[key] = {
                "dtype": str(spec["dtype"].name),
                "shape": list(spec["shape"]),
                "is_image": spec["is_image"],
            }

        with open(features_path, "w") as f:
            json.dump(serializable_spec, f, indent=2)

        logger.debug(f"Wrote features.json to {features_path}")
        logger.debug(f"RLDS dataset created at: {dataset_root}")

        # Write dataset builder class for tfds.builder() compatibility
        self._write_dataset_builder(dataset_root, data)

    def _build_tfds_features(self, feature_spec: dict) -> dict:
        """
        Build TFDS-compatible feature description.

        Args:
            feature_spec: Feature specifications

        Returns:
            Dictionary describing features in TFDS format
        """
        # Map numpy/TF dtypes to TFDS dtype strings
        dtype_map = {
            "float32": "float32",
            "float64": "float64",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "bool": "bool",
        }

        # Build nested feature structure
        features = {
            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
            "featuresDict": {
                "features": {
                    "episode_metadata": {
                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                        "featuresDict": {
                            "features": {
                                "recording_folderpath": {
                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                                },
                                "file_path": {
                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                    "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                                },
                            }
                        },
                    },
                    "is_first": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                    },
                    "is_last": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                    },
                    "is_terminal": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                    },
                    "discount": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                    },
                    "reward": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                    },
                    "language_instruction": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                    },
                    "language_instruction_2": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                    },
                    "language_instruction_3": {
                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                        "tensor": {"shape": {}, "dtype": "string", "encoding": "none"},
                    },
                }
            },
        }

        # Add observation features
        observation_features = {}
        for key, spec in feature_spec.items():
            # Only add features that should be observations (not top-level action)
            if key.startswith("observation/") or key not in ["action"]:
                # Strip "observation/" prefix if present
                obs_key = key.replace("observation/", "") if key.startswith("observation/") else key

                # Skip action from observations
                if obs_key == "action":
                    continue

                dtype_str = dtype_map.get(spec["dtype"].name, "float32")

                # Create tensor feature
                observation_features[obs_key] = {
                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                    "tensor": {
                        "shape": {"dimensions": [str(d) for d in spec["shape"]]},
                        "dtype": dtype_str,
                        "encoding": "bytes",  # All arrays are encoded as bytes in TFRecord
                    },
                }

        # Add observation dict to features
        features["featuresDict"]["features"]["observation"] = {
            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
            "featuresDict": {"features": observation_features},
        }

        # Add action (may not be present in all data)
        features["featuresDict"]["features"]["action"] = {
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
            "tensor": {
                "shape": {"dimensions": ["7"]},
                "dtype": "float64",
                "encoding": "bytes",  # Stored as bytes in TFRecord
            },
        }

        # Add action_dict (empty for now, populated dynamically)
        features["featuresDict"]["features"]["action_dict"] = {
            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
            "featuresDict": {"features": {}},
        }

        return features

    def _write_tfrecords(self, episodes: list[dict], feature_spec: dict, output_path: Path):
        """
        Write episodes to TFRecord format using DROID schema.

        Args:
            episodes: List of episode dictionaries
            feature_spec: Feature specifications
            output_path: Path to write TFRecord file
        """
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for episode in episodes:
                episode_metadata = episode["episode_metadata"]

                # Create episode example
                for step in episode["steps"]:
                    # Build features for this step
                    features = {}

                    # Add episode metadata
                    features["episode_metadata/recording_folderpath"] = self._bytes_feature(
                        [episode_metadata["recording_folderpath"].encode()]
                    )
                    features["episode_metadata/file_path"] = self._bytes_feature([episode_metadata["file_path"].encode()])

                    # Add step metadata
                    features["is_first"] = self._int64_feature([int(step["is_first"])])
                    features["is_last"] = self._int64_feature([int(step["is_last"])])
                    features["is_terminal"] = self._int64_feature([int(step["is_terminal"])])
                    features["discount"] = self._float_feature([step["discount"]])
                    features["reward"] = self._float_feature([step["reward"]])

                    # Add language instructions
                    features["language_instruction"] = self._bytes_feature([step["language_instruction"].encode()])
                    features["language_instruction_2"] = self._bytes_feature([step["language_instruction_2"].encode()])
                    features["language_instruction_3"] = self._bytes_feature([step["language_instruction_3"].encode()])

                    # Add observations
                    for obs_key, obs_value in step["observation"].items():
                        feature_key = f"observation/{obs_key}"
                        features[feature_key] = self._encode_feature(obs_value, feature_spec.get(obs_key, {}))

                    # Add action_dict
                    for action_key, action_value in step["action_dict"].items():
                        feature_key = f"action_dict/{action_key}"
                        features[feature_key] = self._encode_feature(action_value, {})

                    # Add main action
                    if step["action"] is not None:
                        features["action"] = self._encode_feature(step["action"], {})

                    # Create TF example
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

    def _encode_feature(self, value: Any, spec: dict) -> Any:
        """
        Encode a feature value to TensorFlow Feature.

        Args:
            value: Feature value (numpy array, scalar, etc.)
            spec: Feature specification

        Returns:
            TensorFlow Feature
        """
        is_image = spec.get("is_image", False)

        if isinstance(value, np.ndarray):
            # Flatten array for storage
            if is_image and self.compress_images:
                # Encode as JPEG
                if not CV2_AVAILABLE:
                    raise ImportError("cv2 is required for image compression.")

                _, encoded = cv2.imencode(".jpg", cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
                return self._bytes_feature([encoded.tobytes()])
            else:
                # Store as raw bytes
                return self._bytes_feature([value.tobytes()])
        elif isinstance(value, (list, tuple)):
            # Convert to numpy and encode
            arr = np.array(value)
            return self._bytes_feature([arr.tobytes()])
        elif isinstance(value, float):
            return self._float_feature([value])
        elif isinstance(value, int):
            return self._int64_feature([value])
        else:
            # Fallback: try to convert to bytes
            return self._bytes_feature([str(value).encode()])

    def _bytes_feature(self, value):
        """Create a bytes feature."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self, value):
        """Create a float feature."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value):
        """Create an int64 feature."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _write_dataset_builder(self, dataset_root: Path, data: dict[str, Any]):
        """
        Write a TFDS dataset builder class file for the dataset.

        This allows the dataset to be loaded with:
        tfds.builder(dataset_name, data_dir=dataset_root.parent)

        Args:
            dataset_root: Root directory of the dataset
            data: Processed data dictionary
        """
        # Create __init__.py to make it a package
        init_file = dataset_root / "__init__.py"
        with open(init_file, "w") as f:
            f.write(f'"""TFDS dataset: {self.dataset_name}"""\n')
            f.write(f"from .{self.dataset_name} import {self.dataset_name.title().replace('_', '')}\n")

        # Create dataset builder class
        builder_file = dataset_root / f"{self.dataset_name}.py"
        builder_code = self._generate_builder_code(data)

        with open(builder_file, "w") as f:
            f.write(builder_code)

        logger.debug(f"Wrote dataset builder to {builder_file}")

    def _generate_builder_code(self, data: dict[str, Any]) -> str:
        """Generate the Python code for the TFDS dataset builder class."""

        class_name = self.dataset_name.title().replace("_", "")

        # Build feature dict code
        features_code = self._generate_features_code(data["feature_spec"])

        code = f'''"""TFDS dataset builder for {self.dataset_name}."""

import tensorflow_datasets as tfds
import tensorflow as tf


class {class_name}(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for {self.dataset_name} dataset."""

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {{
        '1.0.1': 'Initial release.',
    }}

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="{self.task_description}",
            features=tfds.features.FeaturesDict({{
{features_code}
            }}),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # The data_dir is where the pre-generated TFRecords are stored
        return {{
            'train': self._generate_examples(split='train'),
        }}

    def _generate_examples(self, split):
        """Yields examples from pre-generated TFRecords."""
        # This is a lightweight builder that reads from pre-existing TFRecords
        # The actual data is already in the TFRecord files
        # We just need to parse them

        import glob
        import os

        # Try multiple locations for the TFRecord files
        # 1. In the dataset root (standard for builder_from_directory)
        # 2. In the version directory (for compatibility)
        dataset_root = (
            self.data_dir / self.name
            if hasattr(self.data_dir, "__truediv__")
            else os.path.join(self.data_dir, self.name)
        )

        patterns = [
            str(dataset_root) + f'/{{self.name}}-{{split}}.tfrecord-*',
            str(self.data_dir) + f'/{{self.name}}/{{self.name}}-{{split}}.tfrecord-*',
        ]

        tfrecord_files = []
        for pattern in patterns:
            tfrecord_files = sorted(glob.glob(pattern))
            if tfrecord_files:
                break

        if not tfrecord_files:
            raise FileNotFoundError(f'No TFRecord files found matching patterns: {{patterns}}')

        for tfrecord_file in tfrecord_files:
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            for idx, raw_record in enumerate(dataset):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

                # Parse the example into the expected format
                parsed = self._parse_example(example)
                yield f'{{tfrecord_file}}_{{idx}}', parsed

    def _parse_example(self, example):
        """Parse a TFRecord example into the expected feature format."""
        features = example.features.feature

        # Extract all fields
        result = {{}}

        # Episode metadata
        result['episode_metadata'] = {{
            'recording_folderpath': features['episode_metadata/recording_folderpath'].bytes_list.value[0].decode(),
            'file_path': features['episode_metadata/file_path'].bytes_list.value[0].decode(),
        }}

        # Scalars and flags
        result['is_first'] = bool(features['is_first'].int64_list.value[0])
        result['is_last'] = bool(features['is_last'].int64_list.value[0])
        result['is_terminal'] = bool(features['is_terminal'].int64_list.value[0])
        result['discount'] = features['discount'].float_list.value[0]
        result['reward'] = features['reward'].float_list.value[0]

        # Language instructions
        result['language_instruction'] = features['language_instruction'].bytes_list.value[0].decode()
        result['language_instruction_2'] = features['language_instruction_2'].bytes_list.value[0].decode()
        result['language_instruction_3'] = features['language_instruction_3'].bytes_list.value[0].decode()

        # Observation
        result['observation'] = {{}}
        for key in features.keys():
            if key.startswith('observation/'):
                obs_key = key.replace('observation/', '')
                value = features[key].bytes_list.value[0]
                # Decode based on feature type (this is simplified - production code would need dtype info)
                result['observation'][obs_key] = value

        # Action
        if 'action' in features:
            result['action'] = features['action'].bytes_list.value[0]

        # Action dict
        result['action_dict'] = {{}}
        for key in features.keys():
            if key.startswith('action_dict/'):
                action_key = key.replace('action_dict/', '')
                result['action_dict'][action_key] = features[key].bytes_list.value[0]
        return result
'''
        return code

    def _generate_features_code(self, feature_spec: dict) -> str:
        """Generate Python code for the features dictionary."""
        lines = []

        # Dtype mapping from TF dtype to string
        dtype_map = {
            "float32": "tf.float32",
            "float64": "tf.float64",
            "int32": "tf.int32",
            "int64": "tf.int64",
            "uint8": "tf.uint8",
            "bool": "tf.bool",
        }

        # Episode metadata
        lines.append("                'episode_metadata': tfds.features.FeaturesDict({")
        lines.append("                    'recording_folderpath': tfds.features.Text(),")
        lines.append("                    'file_path': tfds.features.Text(),")
        lines.append("                }),")

        # Standard fields
        lines.append("                'is_first': tf.bool,")
        lines.append("                'is_last': tf.bool,")
        lines.append("                'is_terminal': tf.bool,")
        lines.append("                'discount': tf.float32,")
        lines.append("                'reward': tf.float32,")
        lines.append("                'language_instruction': tfds.features.Text(),")
        lines.append("                'language_instruction_2': tfds.features.Text(),")
        lines.append("                'language_instruction_3': tfds.features.Text(),")

        # Observation
        lines.append("                'observation': tfds.features.FeaturesDict({")
        for key, spec in feature_spec.items():
            # Skip action from observations
            if key == "action":
                continue

            obs_key = key.replace("observation/", "") if key.startswith("observation/") else key
            shape = spec["shape"]
            dtype = spec["dtype"]
            is_image = spec.get("is_image", False)

            # Get dtype string
            dtype_name = dtype.name if hasattr(dtype, "name") else str(dtype)
            dtype_str = dtype_map.get(dtype_name, "tf.float32")

            if is_image:
                lines.append(f"                    '{obs_key}': tfds.features.Image(shape={shape}, dtype=tf.uint8),")
            else:
                lines.append(f"                    '{obs_key}': tfds.features.Tensor(shape={shape}, dtype={dtype_str}),")
        lines.append("                }),")

        # Action
        lines.append("                'action': tfds.features.Tensor(shape=(7,), dtype=tf.float64),")

        # Action dict
        lines.append("                'action_dict': tfds.features.FeaturesDict({}),")

        return "\n".join(lines)

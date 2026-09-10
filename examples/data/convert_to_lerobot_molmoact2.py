# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert recorded bimanual YAM episodes to a LeRobot dataset for fine-tuning MolmoAct2.

The counterpart to `convert_to_lerobot_droid.py`, targeting the schema
`allenai/MolmoAct2-BimanualYAM` was trained on rather than DROID's. It writes LeRobot
**v3.0**, which is what MolmoAct2's trainer reads, so it runs in the training venv rather
than the project one -- `rio.data.LeRobotFormatter` targets v2.1 and cannot be reused.

Almost nothing needs reshaping. `BimanualObs.proprio` is already
`[arm1 joints (6), gripper1, arm2 joints (6), gripper2]`, which is exactly the fourteen
names the `yam_dual_molmoact2` tag declares, left arm before right; `action` is an absolute
joint target in the same units; and the grippers already run 0-1, which is what lets the
mixture set `normalize_gripper=False`. Two things do change: `overhead` becomes the
checkpoint's `top` view, and 50 Hz recordings are resampled onto a 30 Hz grid.

    python examples/data/convert_to_lerobot_molmoact2.py --input <dataset_directory> \
        --repo-id yam_dataset --output "$LEROBOT_DATA_ROOT/yam_dataset"
    python examples/data/convert_to_lerobot_molmoact2.py --verify --output ... --repo-id ...
"""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import robodm
import tyro
from loguru import logger

# This tool only ever reads and writes local datasets. Without this, LeRobot resolves the
# repo id against the Hub and fails on a name that has never been pushed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# The fourteen names `yam_dual_molmoact2` declares, in order.
MOTOR_NAMES = [
    "left_joint_0.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_gripper.pos",
    "right_joint_0.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_gripper.pos",
]

# The checkpoint reads its views *positionally* as top, left, right, so `overhead` fills
# the `top` slot. Fixed rather than inferred: all three cameras are 480x640.
CAMERA_MAPPING = {
    "observation/cameras/overhead/rgb": "observation.images.top",
    "observation/cameras/left/rgb": "observation.images.left",
    "observation/cameras/right/rgb": "observation.images.right",
}

# Proprio and action need no renaming beyond the '/' -> '.' the LeRobot names use.
STATE_MAPPING = {
    "observation/proprio": "observation.state",
    "action": "action",
}
CAMERA_FEATURES = ["observation.images.top", "observation.images.left", "observation.images.right"]

# Matches allenai/31122025-tablebuss-04. Chunk and file sizes are LeRobot's own defaults.
ROBOT_TYPE = "bi_yam_follower"


def resample_indices(timestep: np.ndarray, target_fps: float) -> np.ndarray:
    """Frame indices nearest a uniform grid across an episode's wall clock.

    Used instead of robodm's own `desired_frequency`, whose upsampling branch keys off
    per-stream PTS and tries to allocate 703 GiB on these h264-plus-rawvideo recordings.

    Args:
        timestep: Per-frame timestamps, monotonically non-decreasing.
        target_fps: Grid rate. Deduplicated, so a rate above the source invents no frames.

    Returns:
        Strictly increasing indices into `timestep`.
    """
    t = np.asarray(timestep, dtype=np.float64).reshape(-1)
    if t.size == 0:
        return np.empty(0, dtype=np.int64)
    if t.size == 1:
        return np.zeros(1, dtype=np.int64)

    t = t - t[0]
    targets = np.arange(int(np.floor(t[-1] * target_fps)) + 1, dtype=np.float64) / target_fps
    idx = np.clip(np.searchsorted(t, targets), 1, len(t) - 1)
    take_left = (targets - t[idx - 1]) <= (t[idx] - targets)
    return np.unique(np.where(take_left, idx - 1, idx)).astype(np.int64)


def build_features(height: int, width: int) -> dict:
    """The LeRobot feature spec for this checkpoint's fourteen motors and three views."""
    features = {
        key: {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": MOTOR_NAMES} for key in ("action", "observation.state")
    }
    for name in CAMERA_FEATURES:
        features[name] = {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    return features


def find_trajectories(roots: list[Path], limit: int | None = None) -> list[Path]:
    """Every `.vla` under the given roots, sorted within each and concatenated in root order.

    Episode index follows position, so several recording sessions merge into one dataset
    even when their filenames collide.
    """
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            found.append(root)
        elif root.is_dir():
            in_root = sorted(root.rglob("*.vla"))
            if not in_root:
                raise ValueError(f"No .vla files under {root}")
            found.extend(in_root)
        else:
            raise ValueError(f"Path does not exist: {root}")
    return found[:limit] if limit else found


def infer_frame_size(data: dict, camera_mapping: dict[str, str]) -> tuple[int, int]:
    """Frame height and width read off the recording rather than assumed.

    Args:
        data: A loaded trajectory.
        camera_mapping: robodm camera key -> LeRobot view name.

    Returns:
        (height, width). The three views must agree: the feature spec is written once and
        every later episode is validated against it.
    """
    sizes = {key: tuple(np.asarray(data[key]).shape[1:3]) for key in camera_mapping}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Cameras disagree on frame size: {sizes}")
    height, width = next(iter(sizes.values()))
    logger.info(f"Inferred frame size: {height}x{width}")
    return int(height), int(width)


def episode_task(data: dict, args: "Args", arm_by_file: dict[str, str], path: Path) -> str:
    """The instruction stored on every frame of one episode.

    Args:
        data: A loaded trajectory.
        args: Parsed configuration.
        arm_by_file: Per-episode `arm` labels, keyed by file name.
        path: Trajectory being converted, for the per_arm lookup.

    Returns:
        The task string.
    """
    task = args.task
    if task is None:
        # robodm hands strings back as bytes, which str() would render as "b'pick up...'".
        recorded = np.asarray(data["instruction"]).reshape(-1)[0]
        task = recorded.decode() if isinstance(recorded, bytes) else str(recorded)
    if args.task_mode == "per_arm":
        task = args.task_per_arm.format(task=task, arm=arm_by_file[path.name])
    return task


def warn_if_not_absolute(path: Path, proprio: np.ndarray, action: np.ndarray, keep: np.ndarray):
    """Warn when `action` does not read as an absolute joint pose.

    The checkpoint's control mode is absolute joint pose. A station configured for
    `JOINT_VEL` records velocities into the same column, which converts and trains without
    complaint and yields a policy that commands nonsense.

    Args:
        path: Trajectory being converted, for the message.
        proprio: Recorded joint state, (N, 14).
        action: Recorded action, (N, 14).
        keep: Frame indices being kept.
    """
    gap = float(np.abs(np.asarray(action)[keep] - np.asarray(proprio)[keep]).max())
    if gap > 0.5:
        logger.warning(
            f"{path.name}: action differs from proprio by up to {gap:.2f}, which does not read as an "
            "absolute joint pose. Check the station's action_space."
        )


def convert_to_lerobot(args: "Args"):
    """Write the LeRobot v3.0 dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    roots = [Path(p).expanduser() for p in args.input]
    feature_mapping = (args.camera_mapping or CAMERA_MAPPING) | STATE_MAPPING
    output_path = resolve_output(args)
    trajectories = find_trajectories(roots, args.limit_episodes)
    logger.info(f"Converting {len(trajectories)} episodes -> {output_path} (task mode: {args.task_mode})")

    arm_by_file = load_arm_labels(args, roots[0])
    if args.task_mode == "per_arm" and not arm_by_file:
        raise SystemExit(f"--task-mode per_arm needs a manifest.json with per-episode 'arm' labels near {roots[0]}")

    if output_path.exists():
        if not args.clean:
            raise SystemExit(f"{output_path} already exists; pass --clean to replace it")
        logger.warning(f"Removing {output_path}")
        shutil.rmtree(output_path)

    dataset = None
    for episode_idx, path in enumerate(trajectories):
        # A plain load; one episode is roughly 7 GiB resident and is freed each pass.
        data = robodm.Trajectory(path=str(path), mode="r").load()
        # timestep is dropped by the recorder when a Step carries none, so check it here
        # rather than let resample_indices fail on a bare KeyError.
        missing = [key for key in (*feature_mapping, "timestep") if key not in data]
        if missing:
            raise KeyError(f"{path.name} is missing {missing}; it has {sorted(data)}")

        if dataset is None:
            height, width = infer_frame_size(data, args.camera_mapping or CAMERA_MAPPING)
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=args.fps,
                root=output_path,
                robot_type=args.robot_type,
                features=build_features(height, width),
                use_videos=True,
            )

        keep = resample_indices(data["timestep"], args.fps)
        task = episode_task(data, args, arm_by_file, path)
        warn_if_not_absolute(path, data["observation/proprio"], data["action"], keep)

        for i in keep:
            dataset.add_frame({lerobot: data[key][i] for key, lerobot in feature_mapping.items()} | {"task": task})
        dataset.save_episode()

        logger.info(
            f"[{episode_idx + 1}/{len(trajectories)}] {path.name}: "
            f"{len(data['timestep'])} -> {len(keep)} frames @ {args.fps} fps"
        )
        del data

    if dataset is None:
        raise SystemExit("No episodes were converted")

    # Episode metadata is buffered; without this a short run leaves meta/episodes empty.
    dataset.finalize()

    if args.exact_quantiles:
        write_quantile_stats(args.repo_id, output_path)
    logger.info(f"Done. Verify with:\n  {sys.argv[0]} --verify --output {output_path} --repo-id {args.repo_id}")


def write_quantile_stats(repo_id: str, output_path: Path):
    """Replace the aggregated q01/q99 in meta/stats.json with exact ones.

    LeRobot already writes q01/q99 as each episode is saved, from a 5000-bin histogram, and
    those are what `--norm_mode=q01_q99` reads. This recomputes them exactly, which costs a
    decode of every video in the dataset and so needs torchcodec and the system FFmpeg
    libraries -- neither is otherwise required to convert.

    Computed here rather than through upstream's `augment_dataset_quantile_stats.py`, whose
    final step is an unconditional `push_to_hub()` -- that would upload the demonstrations.

    Args:
        repo_id: Dataset repo id.
        output_path: Dataset root.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import write_stats
    from lerobot.datasets.v30.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

    logger.info("Recomputing exact q01/q99; this decodes every video (no Hub upload)...")
    # Reopened rather than reusing the object create() returned: episode metadata is only
    # populated by a load from disk.
    dataset = LeRobotDataset(repo_id, root=output_path)
    write_stats(compute_quantile_stats_for_dataset(dataset), dataset.meta.root)


def verify_dataset(args: "Args"):
    """Check a converted dataset before spending GPU hours on it.

    A camera-order or state-layout mistake trains perfectly happily and yields a policy
    that reaches for the wrong place, so the checks here are the ones that catch that class
    of error: exact motor names, three distinct camera streams, and actions that read as
    absolute poses near the state they were recorded against.
    """
    output_path = resolve_output(args)
    info = json.loads((output_path / "meta" / "info.json").read_text())
    stats = json.loads((output_path / "meta" / "stats.json").read_text())
    failures = []

    def check(label: str, ok: bool, detail: str = ""):
        logger.info(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(label)

    check("codebase_version is v3.0", info.get("codebase_version") == "v3.0", str(info.get("codebase_version")))
    check(f"fps is {args.fps}", info.get("fps") == args.fps, str(info.get("fps")))
    for key in ("action", "observation.state"):
        check(f"{key} names are the 14 canonical motors", info["features"].get(key, {}).get("names") == MOTOR_NAMES)
        # Training normalizes with q01_q99; a dataset without these fails at startup.
        for quantile in ("q01", "q99"):
            values = stats.get(key, {}).get(quantile)
            check(f"stats.json {key}.{quantile} present, length 14", isinstance(values, list) and len(values) == 14)
    for camera in CAMERA_FEATURES:
        check(f"{camera} present as video", info["features"].get(camera, {}).get("dtype") == "video")
    check("parquet shards written", any((output_path / "data").rglob("*.parquet")))
    for camera in CAMERA_FEATURES:
        check(f"{camera} mp4 written", any((output_path / "videos" / camera).rglob("*.mp4")))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Reading a frame back decodes video, which needs torchcodec and the system FFmpeg
    # libraries. The checks above are all metadata and stay useful without them, so a
    # missing decoder skips the content checks rather than losing the whole report.
    try:
        item = LeRobotDataset(args.repo_id, root=output_path)[0]
    except (OSError, RuntimeError, ImportError) as e:
        logger.warning(f"SKIP  content checks: no video decoder ({type(e).__name__})")
        if failures:
            raise SystemExit(f"{len(failures)} check(s) failed: {failures}") from e
        logger.info("Metadata checks passed; install ffmpeg to also check frame content.")
        return

    state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
    action = np.asarray(item["action"], dtype=np.float32).reshape(-1)
    check(
        "action is an absolute pose near the state",
        float(np.abs(action - state).max()) < 0.5,
        f"max|a-s|={float(np.abs(action - state).max()):.3f}",
    )
    grippers = state[[6, 13]]
    check(
        "grippers within [0, 1], 0.05 tolerance",
        bool(((grippers >= -0.05) & (grippers <= 1.05)).all()),
        str(grippers),
    )
    means = [round(float(np.asarray(item[camera], dtype=np.float32).mean()), 3) for camera in CAMERA_FEATURES]
    check("the three cameras differ", len(set(means)) == 3, str(means))

    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    logger.info("All checks passed. Safe to train.")


def resolve_output(args: "Args") -> Path:
    return Path(args.output).expanduser() if args.output else Path.home() / "lerobot_data" / args.repo_id


def load_arm_labels(args: "Args", input_root: Path) -> dict[str, str]:
    """Per-episode `arm` labels, when the recordings carry a manifest."""
    if args.manifest:
        path = Path(args.manifest).expanduser()
    else:
        root = input_root if input_root.is_dir() else input_root.parent
        path = root / "manifest.json"
    if not path.exists():
        return {}
    episodes = json.loads(path.read_text()).get("episodes", [])
    logger.info(f"Read {len(episodes)} episode labels from {path}")
    return {e["file"]: e["arm"] for e in episodes if "arm" in e}


def main(args: "Args"):
    if args.verify:
        verify_dataset(args)
    else:
        convert_to_lerobot(args)


# Deliberately not a DatasetCfg subclass, unlike convert_to_lerobot_droid.py: importing
# anything from `rio` runs rio/__init__.py, which pulls in rio_hw, and the training venv
# installs rio with --no-deps because rio_hw pins a pyrealsense2 with no cp312 wheel.
@dataclass
class Args:
    input: tuple[str, ...] = ("data/",)
    """Directories (searched recursively) or `.vla` files. Order fixes episode order."""
    output: str | None = None
    """Dataset root. Defaults to ~/lerobot_data/<repo_id>."""
    clean: bool = False
    """Replace an existing dataset at `output` instead of refusing to overwrite it."""

    repo_id: str = "yam_dataset"
    fps: int = 30
    """Rate to resample onto. 30 matches the data the checkpoint was pretrained on."""
    robot_type: str = ROBOT_TYPE
    camera_mapping: dict[str, str] | None = None
    """robodm camera key -> LeRobot view name. None uses CAMERA_MAPPING."""

    task: str | None = None
    """Instruction for every episode. None uses each episode's own `instruction` stream."""
    task_mode: Literal["single", "per_arm"] = "single"
    """`single` gives every episode the same instruction, leaving the model to read which
    arm to use off the images. `per_arm` names the arm instead, which removes that
    ambiguity but means saying which arm at inference time."""
    task_per_arm: str = "{task}, using the {arm} arm"
    """Template for --task-mode per_arm. Gets the episode's task and its `arm` label."""
    manifest: str | None = None
    """Episode metadata carrying `arm` labels. Defaults to <input[0]>/manifest.json."""

    limit_episodes: int | None = None
    """Convert only the first N episodes. For smoke runs."""
    exact_quantiles: bool = False
    """Recompute q01/q99 exactly after conversion. Decodes every video; the approximate
    ones LeRobot writes per episode are what training reads and are normally enough."""
    verify: bool = False
    """Check an already-converted dataset instead of converting."""
    verbose: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

"""Script to convert robodm datasets to lerobot or rlds format."""

import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger


def convert_to_lerobot(
    robodm_path: str,
    output_path: str | None,
    task: str,
    fps: int,
    robot_type: str,
    repo_id: str,
    verbose: bool,
):
    """Convert robodm dataset to LeRobot format.

    robodm_path can be either:
    - A single .vla file
    - A directory containing multiple .vla files (each will be processed as an episode)

    If output_path is None, uses the default LeRobot cache location:
    ~/.cache/huggingface/lerobot/{repo_id}
    """
    from rio.data import LeRobotFormatter

    # Use default LeRobot cache location if output_path is None
    if output_path is None:
        output_path = str(Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id)
        logger.info(f"Using default LeRobot cache location: {output_path}")

    # - image: Main camera view (256x256x3 RGB)
    # - wrist_image: Wrist camera view (256x256x3 RGB)
    # - state: Robot state (8D: 7 joint positions + 1 gripper)
    # - actions: Delta joint positions (7D)

    # LIBERO mapping
    # feature_mapping = {
    #     "observation/cameras/camera1/rgb": "image",
    #     "observation/cameras/camera2/rgb": "wrist_image",
    #     "observation/proprio": "state",
    #     "action": "actions",
    # }

    # DROID mapping
    feature_mapping = {
        "observation/cameras/camera1/rgb": "exterior_image_1_left",
        "observation/cameras/camera2/rgb": "exterior_image_2_left",
        "observation/cameras/camera3/rgb": "wrist_image_left",
        "observation/proprio_joints": "joint_position",
        "observation/gripper_position": "gripper_position",
        "action": "actions",
        "prompt": "prompt",
    }

    formatter = LeRobotFormatter(
        robodm_path=robodm_path,
        output_path=output_path,
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        task=task,
        verbose=verbose,
        feature_mapping=feature_mapping,
        feature_transforms=None,
        only_mapped_keys=True,
    )

    logger.info(f"Converting {robodm_path} to LeRobot format...")
    formatter.convert()
    logger.info(f"✓ Conversion complete! Output at: {output_path}")


def convert_to_rlds(
    robodm_path: str,
    output_path: str,
    task: str,
    fps: int,
    robot_type: str,
    dataset_name: str,
    verbose: bool,
):
    """Convert robodm dataset to RLDS format."""
    from rio.data import RLDSFormatter

    formatter = RLDSFormatter(
        robodm_path=robodm_path,
        output_path=output_path,
        dataset_name=dataset_name,
        fps=fps,
        robot_type=robot_type,
        task_description=task,
        verbose=verbose,
    )

    logger.info(f"Converting {robodm_path} to RLDS format...")
    formatter.convert()
    logger.info(f"✓ Conversion complete! Output at: {output_path}")


def main():
    """Main entry point for the dataset conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert robodm datasets to lerobot or rlds format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/tmp/dummy_data/",
        help=(
            "Path to input robodm dataset. For lerobot: single .vla file or directory with multiple .vla files."
            " For rlds: supports glob patterns."
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Output directory for converted dataset. For lerobot format, if not specified,"
            " uses ~/.cache/huggingface/lerobot/{repo_id}. For rlds format, uses input path with format suffix."
        ),
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["lerobot", "rlds"],
        default="lerobot",
        help="Target dataset format",
    )

    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="make circle",
        help="Task description/name for the dataset",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second of the dataset",
    )

    parser.add_argument(
        "--robot-type",
        type=str,
        default="panda",
        help="Type of robot used for data collection",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for LeRobot format (e.g., 'user/dataset_name'). Required for lerobot format.",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="droid",
        help="Dataset name for RLDS format. Required for rlds format.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory if it exists before conversion",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()

    # Convert based on format
    if args.format == "lerobot":
        # For lerobot, pass the path directly (can be file or directory)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)

        # Determine repo_id
        if args.repo_id:
            repo_id = args.repo_id
        else:
            # Default repo_id based on input path name
            repo_id = f"your_hf_username/{input_path.stem}"

        # Determine output path for lerobot
        if args.output:
            output_path = Path(args.output).expanduser()
        else:
            # For lerobot, use None to trigger default cache location
            output_path = None

        # Clean output directory if requested and output_path is specified
        if args.clean and output_path is not None and output_path.exists():
            logger.warning(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)

        logger.info(f"Processing: {input_path}")
        convert_to_lerobot(
            robodm_path=str(input_path),
            output_path=str(output_path) if output_path else None,
            task=args.task,
            fps=args.fps,
            robot_type=args.robot_type,
            repo_id=repo_id,
            verbose=args.verbose,
        )

        # Show summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Input:  {input_path}")
        if output_path:
            logger.info(f"Output: {output_path}")
        logger.info(f"Format: {args.format}")
        logger.info(f"Task:   {args.task}")
        logger.info("=" * 60)

    elif args.format == "rlds":
        # For rlds, expand glob pattern to handle multiple files
        input_pattern = input_path
        input_files = list(input_pattern.parent.glob(input_pattern.name))

        if not input_files:
            logger.error(f"No input files found matching pattern: {args.input}")
            sys.exit(1)

        # Process each input file
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            # Determine output path for rlds
            if args.output:
                output_path = Path(args.output).expanduser()
            else:
                # Default: use input path with format suffix
                output_base = input_file.parent / input_file.stem
                output_path = Path(f"{output_base}_{args.format}")

            # Clean output directory if requested
            if args.clean and output_path.exists():
                logger.warning(f"Removing existing output directory: {output_path}")
                shutil.rmtree(output_path)

            # Determine dataset_name
            if args.dataset_name:
                dataset_name = args.dataset_name
            else:
                # Default dataset_name based on input filename
                dataset_name = input_file.stem

            convert_to_rlds(
                robodm_path=str(input_file),
                output_path=str(output_path),
                task=args.task,
                fps=args.fps,
                robot_type=args.robot_type,
                dataset_name=dataset_name,
                verbose=args.verbose,
            )

            # Show summary for this file
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Input:  {input_file}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Format: {args.format}")
            logger.info(f"Task:   {args.task}")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()

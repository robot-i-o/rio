#!/usr/bin/env python
"""Launch a MolmoAct2 fine-tune on a rio-converted dataset, without editing third_party.

MolmoAct2's trainer takes a mixture *name* rather than a dataset path, and resolves it
through the `MOLMOACT2_LEROBOT_MIXTURES` registry in its own `data_mixtures.py`. The
documented way to add a dataset is to hand-edit that file, which sits under `third_party/`
and is therefore gitignored: the edit is unversioned and is lost on a re-clone.

`train_plan.py` imports that registry by reference and indexes it at call time, so this
registers the mixture from outside and then hands over to the trainer's own `main()`.
Everything the mixture declares beyond the dataset id is read back out of the base
checkpoint's `norm_stats.json`, so it cannot drift from what the model was trained with.

    torchrun --standalone --nproc-per-node=4 scripts/train_molmoact2.py \
        --dataset-repo-id yam_dataset \
        allenai/MolmoAct2-BimanualYAM yam_dataset \
        --norm_mode=q01_q99 --lora_enable=true --dynamic_seq_len=true
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "third_party" / "molmoact2" / "experiments"
DEFAULT_CHECKPOINT = "allenai/MolmoAct2-BimanualYAM"


def load_tag_metadata(checkpoint: str) -> tuple[str, dict]:
    """The norm tag and its metadata from a checkpoint's `norm_stats.json`.

    Args:
        checkpoint: Local checkpoint directory, or a Hugging Face repo id.

    Returns:
        The tag name and the metadata dict recorded under it.
    """
    local = Path(checkpoint).expanduser() / "norm_stats.json"
    if local.exists():
        path = local
    else:
        # Lazy so a local checkpoint path needs no Hub dependency.
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = Path(hf_hub_download(checkpoint, "norm_stats.json"))

    tags = json.loads(path.read_text())["metadata_by_tag"]
    if len(tags) != 1:
        raise ValueError(f"{checkpoint} declares {len(tags)} norm tags ({sorted(tags)}); expected exactly one")
    return next(iter(tags.items()))


def build_mixture(name: str, dataset_repo_id: str, tag: str, metadata: dict):
    """A one-tag mixture builder for `dataset_repo_id`, shaped by the checkpoint's own tag."""
    # Only importable after add_experiments_to_path().
    from launch_scripts.data_mixtures import build_single_lerobot_mixture  # noqa: PLC0415

    def builder():
        return build_single_lerobot_mixture(
            name=name,
            tag=tag,
            repo_ids=[dataset_repo_id],
            action_key=metadata["action_key"],
            state_keys=[metadata["state_key"]],
            camera_keys=metadata["camera_keys"],
            normalize_gripper=metadata["normalize_gripper"],
            action_horizon=metadata["action_horizon"],
            n_action_steps=metadata["n_action_steps"],
            setup_type=metadata["setup_type"],
            control_mode=metadata["control_mode"],
        )

    return builder


def add_experiments_to_path():
    """Put the vendored trainer and its LeRobot fork on `sys.path`."""
    if not EXPERIMENTS.is_dir():
        raise SystemExit(f"{EXPERIMENTS} not found. Run: bash scripts/setup/vla/molmoact2_train_setup.sh")
    for path in (EXPERIMENTS, EXPERIMENTS / "lerobot" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def import_trainer():
    """The vendored trainer, or a legible error when its environment is missing.

    Returns:
        The `launch_scripts.train_lerobot` module.
    """
    add_experiments_to_path()
    try:
        from launch_scripts import train_lerobot  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            f"the MolmoAct2 trainer is not importable ({e}). Training runs in its own environment, "
            "not the station venv: the vendored LeRobot fork requires Python >=3.12 and rio pins "
            "3.11 for its camera stack. Build it with scripts/setup/vla/molmoact2_train_setup.sh "
            "and run this with .venv-molmoact2-train/bin/python."
        ) from e
    return train_lerobot


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot repo id of the converted dataset")
    parser.add_argument("--mixture-name", default=None, help="Registry name. Defaults to the dataset's basename.")
    parser.add_argument("--base-checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint to read norm_stats.json from")

    # --help lists this wrapper's flags and then the trainer's, which is what people are
    # actually looking up. It has to run before parse_known_args, which would reject the
    # missing --dataset-repo-id first.
    if {"-h", "--help"} & set(sys.argv[1:]):
        parser.print_help()
        train_lerobot = import_trainer()

        print("\n--- flags below are the trainer's ---\n")
        sys.argv = [sys.argv[0], "--help"]
        train_lerobot.main()
        return

    args, passthrough = parser.parse_known_args()

    name = args.mixture_name or args.dataset_repo_id.rsplit("/", 1)[-1]
    tag, metadata = load_tag_metadata(args.base_checkpoint)

    train_lerobot = import_trainer()
    from launch_scripts import data_mixtures  # noqa: PLC0415

    # Imported by reference in train_plan.py and indexed at call time, so mutating it here
    # is visible to the resolver.
    data_mixtures.MOLMOACT2_LEROBOT_MIXTURES[name] = build_mixture(name, args.dataset_repo_id, tag, metadata)
    print(f"registered mixture {name!r}: tag={tag} repo={args.dataset_repo_id}")

    # The trainer parses sys.argv itself, so hand it everything this wrapper did not consume.
    sys.argv = [sys.argv[0], *passthrough]
    train_lerobot.main()


if __name__ == "__main__":
    main()

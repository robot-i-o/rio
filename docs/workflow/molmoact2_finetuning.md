# Finetuning MolmoAct2 with your custom data

The following example demonstrates the tuning of `allenai/MolmoAct2-BimanualYAM` using rollouts collected from a bimanual YAM station.


### Convert the data to LeRobot format

MolmoAct2 trains on LeRobot v3.0, which conflicts with the `lerobot` version the project venv pins for pi0. There are two setup scripts, and they do not overlap: `molmoact2_setup.sh` installs inference and deployment into the project venv, and `molmoact2_train_setup.sh` builds `.venv-molmoact2-train` for conversion and training. The split is not optional — the vendored LeRobot fork uses PEP 695 syntax, so it needs Python 3.12, while rio's camera stack pins a pyrealsense2 with no wheel past 3.11.

```bash
# The trainer's torchcodec preflight shells out to ffmpeg, which is a system package.
sudo apt install ffmpeg
bash scripts/setup/vla/molmoact2_train_setup.sh

export LEROBOT_DATA_ROOT=$HOME/lerobot_data
export HF_HOME=$HOME/.cache/huggingface
export LEROBOT_VIDEO_BACKEND=pyav

# Required by the trainer even for a LeRobot-only run; point them anywhere.
export MOLMO_DATA_DIR=$HOME/molmo_data
export SPATIAL_DATA_HOME=$HOME/spatial_data
export LEROBOT_DEPTH_DATA_ROOT=$HOME/lerobot_depth
# The trainer's config interpolates these with no default. WANDB_MODE=disabled keeps it local.
export WANDB_MODE=disabled WANDB_PROJECT=local WANDB_ENTITY=local
```

Next, convert the collected robodm data to LeRobot format using the `examples/data/convert_to_lerobot_molmoact2.py` script:

```bash
.venv-molmoact2-train/bin/python examples/data/convert_to_lerobot_molmoact2.py \
  --input <dataset_directory> \
  --repo-id yam_dataset \
  --output "$LEROBOT_DATA_ROOT/yam_dataset"
```

This maps `overhead` onto the checkpoint's `top` view, resamples onto the 30 fps grid it was pretrained at, and writes the `q01`/`q99` statistics that `--norm_mode=q01_q99` reads. The dataset stays local, so `--repo-id` is just a name. Use `--limit-episodes 2` for a smoke run, `--clean` to overwrite, and `--task-mode per_arm` to name the moving arm in the instruction.

> Note: LeRobot computes the quantiles as each episode is saved, so there is no second pass. They land in `meta/stats.json`.

Then verify the dataset before training on it:

```bash
.venv-molmoact2-train/bin/python examples/data/convert_to_lerobot_molmoact2.py --verify \
  --output "$LEROBOT_DATA_ROOT/yam_dataset" \
  --repo-id yam_dataset
```

> Note: Dimensions 0-6 are labelled `left_*` because `arm1_cfg` is wired to the left arm, which nothing in software verifies — the CAN interface name is assigned by the host. A swap still runs, but inverts the pretrained checkpoint's per-arm priors, so confirm once by replaying an episode and watching which arm moves: `uv run -m examples.replay_data --loader-cfg.path <dataset_directory>/traj_0000.vla`


### Finetune MolmoAct2 with the converted data

1. Fine tuning the molmo-act2 model:

The trainer takes a mixture name rather than a dataset path, and resolves it through a registry under `third_party/`. `scripts/train_molmoact2.py` registers it from outside instead of editing that gitignored file, deriving every field but the dataset id from the base checkpoint's `norm_stats.json`, then hands over to the trainer:

```bash
.venv-molmoact2-train/bin/torchrun --standalone --nproc-per-node=<num_gpus> \
  scripts/train_molmoact2.py \
  --dataset-repo-id yam_dataset \
  allenai/MolmoAct2-BimanualYAM yam_dataset \
  --norm_mode=q01_q99 \
  --lora_enable=true --lora_rank=64 \
  --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head \
  --global_batch_size=32 --device_batch_size=2 \
  --packing=false --dynamic_seq_len=true \
  --save_folder=$HOME/molmoact2_runs/my_experiment

# Note: LoRA is recommended under ~200 demonstrations
# Note: the ft_* flags default to false; without them the optimizer gets no parameters
# Note: --help lists the argparse flags; --save_folder and --max_duration are config
#       overrides passed through to omegaconf and will not appear there
```

Measured on two RTX PRO 6000 at these settings: ~6.8 s per optimizer step and 22 GB peak per GPU, so roughly two hours per epoch over 30k frames. There is a lot of memory headroom to raise `--device_batch_size`.

Unlike openpi there is no separate norm-stats script; the statistics are read from the dataset's `meta/stats.json` and baked into the checkpoint when you export it.

2. Exporting the final checkpoint:

```bash
python -m third_party.molmoact2.experiments.olmo.hf_model.convert_molmoact2_to_hf /location/to/checkpoint/step4000-merged /location/to/save/exported/checkpoint
# Note: LoRA runs save four directories per step, convert the *-merged one
```


### Deploy the finetuned checkpoint

1. Deploying the checkpoint with RIO:

```bash
STATION=BimanualYamStation POLICY=MolmoAct2BimanualYamCfg uv run -m examples.policy_inference --policy-path location/of/checkpoint --instruction "VLA text instruction (Ex. Pick up the screwdriver)"
```

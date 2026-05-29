# Finetuning a VLA policy with your custom data

The following example demonstrates the tuning of PI0.5 using the rollouts collected from a xArm robot.


### Collect the data using a Recorder Node

Run the script to collect data using a dummy policy. The script will save the collected data in a *robodm* format under `<dataset_directory>`.


```bash
# TODO: ADD CORRECT COMMAND
```

The robodm dataset should contain the following features:
```python
#   - observation/cameras/camera1/rgb: (N, H, W, 3) RGB images from camera 1
#   - observation/cameras/camera2/rgb: (N, H, W, 3) RGB images from camera 2  
#   - observation/proprio_joints: (N, 9) joint positions (we use first 7)
#   - observation/gripper_position: (N, 1) gripper position
#   - action: (N, 7) action array (will be padded to 8D)
```

> Optional: You can visualize the collected data with:
```bash
uv sync --all-extras --group rerun
uv run -m examples.replay_data --loader-cfg.path <dataset_directory>/traj_0000.vla
```


### Convert the data to LeRobot format

Ensure that the correct dependency installed match the versions required by openpi:
```bash
uv sync --all-extras --group openpi
```

Next, convert the collected robodm data to LeRobot format using the `examples/data/convert_to_lerobot_droid.py` script:

```bash
python examples/data/convert_to_lerobot_droid.py --input <dataset_directory> --repo-id 'your_hf_username/droid_xarm' --robot-type xarm
```
No output path is needed since Lerobot uses as default path `~/.cache/huggingface/lerobot/<repo_id>` to store datasets.

If you need to store your dataset in a different location, you can export the variable `HF_LEROBOT_HOME` to point to the desired directory. ```export HF_LEROBOT_HOME=/path/to/your/dataset/directory```. This will ensure that other dataloaders using Lerobot will be able to find the dataset in the alternative location.


### Finetune PI0.5 with the converted data

First we need to configure the new dataset for training in the openpi codebase.
in `third_party/openpi/src/openpi/training/config.py`, add a training configuration for the new dataset that points to the repo_id defined before:

```python
    TrainConfig(
        name="pi05_droid_finetune_xarm",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/droid_xarm",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                asset_id="your_hf_username/droid_xarm",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
```

--- 

The openpi policies also require statistics of the dataset to be precomputed before finetuning.

To compute the statistics, run the following script:

```bash
cd third_party/openpi
uv sync
source .venv/bin/activate
# use the same config name as defined in the training config above
uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune_xarm
```

Finally, you can start the finetuning process using the following command:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_droid_finetune_xarm --exp-name=my_experiment --overwrite
# Note: if using a config name other than pi05_droid_finetune_xarm, change it accordingly
# It training on multiple GPUs, you can also add the flag --fsdp-devices <number_of_gpus>
```
> Note: If your hardware requires a CUDA version higher then 12.6 (e.g. NVIDIA RTX PRO 6000 Blackwell with CUDA 13.x). You can patch the jax dependencies for training: 
```bash
uv sync
uv pip install -U jax[cuda13]
```
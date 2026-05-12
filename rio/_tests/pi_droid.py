# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

try:
    import openpi.models.pi0_config as pi0_config
    import openpi.training.weight_loaders as weight_loaders
    from openpi.policies import policy_config
    from openpi.training.config import AssetsConfig, DataConfig, LeRobotDROIDDataConfig, TrainConfig

    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_ERROR = e

_test_config = TrainConfig(
    # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
    # Here, we use LeRobot data format (like for all other fine-tuning examples)
    # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see
    # examples/droid/convert_droid_data_to_lerobot.py
    name="pi05_droid_finetune_xarm",
    exp_name="pi05_droid_finetune_xarm_exp",
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
            # asset_id="droid",
            asset_id="your_hf_username/droid_xarm",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
    num_train_steps=20_000,
    batch_size=32,
)
policy_path = "/data/checkpoints/pi05_coke_can_30/12000"

if __name__ == "__main__":
    policy = policy_config.create_trained_policy(_test_config, policy_path)

    # Dummy observation for testing
    import numpy as np

    request_data = {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.float32),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.float32),
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": 0.0,
        "prompt": "Pick up the cube and place it in the box.",
    }

    output = policy.infer(request_data)
    breakpoint()

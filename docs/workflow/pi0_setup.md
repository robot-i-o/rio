
# Setup Instructions for VLA Pi0 Policy and variants

Please refer to [openpi](https://github.com/physical-intelligence/openpi.git)  repository for detailed setup instructions of Physical Intelligence policies.

An automated setup script is provided below in [`scripts/setup/vla/pi0_setup.sh`](scripts/setup/vla/pi0_setup.sh).

After setting up OpenPI, you can install the rio specific dependencies by running:

```bash
uv sync --all-extras --group openpi
```

and run the example scripts in [`examples/vla/pi0.py`](examples/vla/pi0.py) as follows, this example uses dummy cameras and robot for testing.

```bash
python examples/vla/pi0.py
```


## Finetuning example Droid XArm with Pi0.5
Please refer to the the [pi0_droid_finetuning.md](./pi0_droid_finetuning.md) for instructions on how to finetune Pi0.5 with your custom data collected from a xArm robot.
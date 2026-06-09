# Humanoid Control

Run tethered full-body humanoid control through a re-implementation of [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl).

> ⚠️ **Disclaimer** ⚠️ 
> 
> Please exercise caution when deploying on real hardware. The code is provided as-is for testing and education purposes. We do not take any responsibility for damage to property or injury to persons that may occur while using this system.

### Hardware Requirements

- A CUDA-enabled PC with **Ubuntu 22.04** or **Ubuntu 24.04**.
    - We have tested this system with NVIDIA RTX 4090 and RTX 5090 graphics cards. Other NVIDIA graphics cards should also work, but please test in simulation first to ensure that the networks run smoothly.
- 29DOF **Unitree G1 EDU** with controller
- Overhead Gantry for G1
- **Pico 4 Ultra** with 2 Motion Trackers
- Ethernet cable (10ft+)
- USB-C cable

## Setup Policy Requirements

The script `scripts/setup/humanoid/gearsonic_setup.sh` automates this section —
git-lfs, the GR00T-WholeBodyControl clone, model download, the `gear_sonic`
module, and onnxruntime. Run it from the repo root:

    # CUDA 12.X
    bash scripts/setup/humanoid/gearsonic_setup.sh

    # CUDA 13.X
    bash scripts/setup/humanoid/gearsonic_setup.sh --cuda13

??? note "Manual policy setup — automated by `gearsonic_setup.sh`"

    - Install `git-lfs` if it's not already installed

            sudo apt install git-lfs && git lfs install

    - Install models and assets from GEAR-SONIC

            mkdir third_party
            cd third_party

            # Download GEAR-SONIC codebase
            git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
            cd GR00T-WholeBodyControl
            git lfs pull

            # Download models
            uv pip install huggingface_hub
            python download_from_hf.py

            # Import gear_sonic module (used for SMPL processing)
            cd gear_sonic
            uv pip install -e .

    - Install ONNX for your CUDA version

            # CUDA 12.X
            uv pip install onnxruntime-gpu
            # CUDA 13.X
            uv pip install coloredlogs flatbuffers numpy packaging protobuf sympy
            uv pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu

            python -c "import onnxruntime as ort; ort.get_available_providers()"

## Setup Teleoperation

### Install XRoboToolkit on your PC

- Download the deb package for [Ubuntu 22.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb) or [Ubuntu 24.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb)
- To install, run

        sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb

    or

        sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_24.04_amd64.deb
- Open the XRoboToolkit-PC-Service application
- Install Python bindings for XRobotToolkit from GEAR-SONIC

        export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
        uv pip install --no-build-isolation -e third_party/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64


### Setup XRoboToolkit on the Pico 4 Ultra

- Install [adb](https://developer.android.com/studio/releases/platform-tools) on your PC if it is not already installed
- [Enable developer mode](https://developer.picoxr.com/ja/document/unreal/test-and-build/) on Pico 4 Ultra
- Download [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk) on your PC
- Connect your headset to your PC using a USB cable and run

        # Install XRoboToolkit on Pico
        adb install -g XRoboToolkit-PICO-1.1.1.apk

        # Find the Pico device ID
        adb devices
        # Setup reverse tethering using ADB
        adb reverse tcp:63901 tcp:63901

- Put on your headset and calibrate your motion trackers
- Open the app, and do the following steps:
    - Next to **PC Service:**, press the **Enter** button and input `127.0.0.1`
    - Under **Tracking**, enable **Head** and **Controller**.
    - Set **Mode** to **Full-body**
    - Next to **Status:**, press the **Connect** button
    - Text should appear next to **Status** that says **WORKING**
        - If the connection failed, double check that the XRoboToolkit PC service is running
    - Under **Data & Control** enable **Send**

## Setup Unitree G1 Interface
- Power on your Unitree G1 and connect it to your PC using an ethernet cable
- Set a static IP on the same subnet as the G1  

        # find the robot-facing ethernet interface (typically enp* or eth0)
        ifconfig

        # assign an IP in the 192.168.123.x range (e.g., .244)
        sudo ip addr flush <INTERFACE>
        sudo ip addr add 192.168.123.244/24 dev <INTERFACE>

        # verify connectivity to robot development computer
        ping 192.168.123.164

- Install [Holosoma](github.com/amazon-far/holosoma) Unitree SDK2 Python bindings

        uv pip install unitree_sdk2 --no-index --find-links "https://github.com/amazon-far/unitree_sdk2/releases/expanded_assets/0.1.3"
        python -c "import unitree_interface.unitree_interface as m; print(m.__file__)"

## Deployment

> ⚠️ **Safety Notes** ⚠️
>
> - We **highly recommend** that you run in simulation first to get used to the controls and make sure the system runs smoothly. If the simulation is lagging, it can mean that your policy is not running on GPU, or your system is not powerful enough.
> - We also recommend two people for initial testing, one to teleoperate through VR and another to run the script and kill the process if necessary.
> - The low-level controller was not trained with varying terrain. Do not attempt to climb stairs, go up slopes, or sit down on chairs.
> - We don't recommend putting clothes on the robot, as it can cause dangerous and erratic behavior from our testing.

### Run the script in simulation

- Run

        STATION=G1Station python -m examples.teleop_humanoid --sim
- Press **Enter** on the keyboard to engage the controller
- Move to mirror the standing position of the robot, then press **B** on the right Pico controller to start full-body tracking
- Press **A** at any time to pause tracking, and **B** to resume
- Press **Control-C** on the keyboard, or **X** on the left controller at any time to kill the process 

### Run the script on real hardware

- Turn on the Unitree G1
- Use the G1 controller to enter **Damping Mode** by simultaneously pressing **R2** and **L2**, then enter **Development Mode** by pressing **L2** and **A**.
- Launch the **XRoboToolkit PC** application
- Lift the **Unitree G1** to standing position using the gantry
- Put on the **Pico 4 Ultra** again, make sure it's connected to the PC via USB-C, and recalibrate/reconnect.
    - Optional, in case `adb` isn't configured yet:

            adb reverse tcp:63901 tcp:63901

- Run

        STATION=G1Station python -m examples.teleop_humanoid
- Wait for the policies to load, then press **Enter** on the keyboard to go to initial position.
- Lower the gantry until the robot leans slightly forward
- Press **Enter** to engage the controller
- Move to mirror the standing position of the robot, then press **B** on the right Pico controller to start full-body tracking
- Press **A** at any time to pause tracking, and **B** to resume
- Press **X** on the left controller or **Control-C** on the keyboard at any time to kill the process. 


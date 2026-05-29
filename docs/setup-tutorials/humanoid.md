# Humanoid Control

Run tethered full-body humanoid control through a re-implementation of [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl).

### Hardware Requirements

- 29DOF **Unitree G1 EDU**
- Overhead Gantry for G1
- **Pico 4 Ultra** with 2 Motion Trackers
- Ethernet cable
- (Optional) USB-C cable

## Setup Policy Requirements
- Install models and assets from GEAR-SONIC

        mkdir third_party
        cd third_party
        git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
        cd GR00T-WholeBodyControl
        # Install git-lfs if it's not already installed
        # sudo apt install git-lfs && git lfs install
        git lfs pull
        uv pip install huggingface_hub
        # Download models
        python download_from_hf.py
        cd gear_sonic
        uv pip install -e .

- Install ONNX for your CUDA version

        # CPU
        uv pip install onnxruntime
        # CUDA 12.X
        uv pip install onnxruntime-gpu # cuda 12.X
        # CUDA 13.X
        uv pip install coloredlogs flatbuffers numpy packaging protobuf sympy
        uv pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu

## Setup Teleoperation

### Install XRoboToolkit on your PC

- Download the deb package for [Ubuntu 22.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb) or [Ubuntu 24.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb)
- To install, run

        sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb

    or

        sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_24.04_amd64.deb
- Install Python bindings for XRobotToolkit from GEAR-SONIC

        export CMAKE_PREFIX_PATH="$(python -m pybind 11 --cmakedir)"
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
    - Text should appear next to **Status:** that says **WORKING*
    - Under **Data & Control** enable **Send**

### Setup Unitree G1 Interface
- Power on your Unitree G1 and connect it to your PC using an ethernet cable
- Set a static IP on the same subnet as the G1  

        # find the robot-facing ethernet interface (typically enp* or eth0)
        ifconfig

        # assign an IP in the 192.168.123.x range (e.g., .244)
        sudo ip addr add 192.168.123.244/24 dev <INTERFACE>

        # verify connectivity to robot development computer
        ping 192.168.123.164

- Install [Holosoma](github.com/amazon-far/holosoma) Unitree SDK2 Python bindings

        uv pip install unitree_sdk2 --no-index --find-links "https://github.com/amazon-far/unitree_sdk2/releases/expanded_assets/0.1.3"
        python -c "import unitree_interface.unitree_interface as m; print(m.__file__)"

### Run the script
- Use the G1 controller to enter development mode by simultaneously pressing and holding **R1** and **L1**
- Launch the **XRoboToolkit PC** application
- Lift the **Unitree G1** to standing position
- Put on the **Pico 4 Ultra** again and recalibrate/reconnect.
- Run the script, wait for the policies to load, then press **Enter** to go to initial position.
- Lower the gantry until the robot leans slightly forward
- Press **Enter** to engage the controller
- Move to mirror the standing position of the robot, then press **B** on the right Pico controller to start full-body tracking
- Press **A** at any time to pause tracking, and **B** to resume
- Press **X** on the left controller at any time to kill the process


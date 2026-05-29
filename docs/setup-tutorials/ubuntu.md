# Real-time Ubuntu

Tested: Ubuntu 22.04 LTS with RT kernel version 5.15.

Reference: [Real-time Ubuntu docs](https://documentation.ubuntu.com/real-time/latest/how-to/enable-real-time-ubuntu/).

> You need an Ubuntu Pro subscription (free for personal use). Get your token at https://ubuntu.com/pro/dashboard.

> This will install a specific RT kernel version, see [RT Ubuntu releases](https://documentation.ubuntu.com/real-time/latest/reference/releases/). If you want to build a real-time kernel patch for your specific kernel version from source, see [Franka docs](https://frankarobotics.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel).

> You will need to re-install NVIDIA CUDA drivers after enabling real-time kernel. Download a [CUDA runfile installer](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu) before proceeding. The runfile installer (with DKMS) is easier for RT Ubuntu setup than Ubuntu's packaged ones, which may not match the RT kernel.

### 1) Install RT kernel
```bash
# Dependencies
sudo apt update && sudo apt install -y \
    python3-dev \
    build-essential \
    mokutil

# Check that secure boot is disabled
sudo mokutil --sb-state

# Attach Ubuntu Pro (free personal plan allows up to 5 machines)
sudo pro attach <YOUR-TOKEN>

# Enable RT kernel
sudo pro enable realtime-kernel

# Set IGNORE_PREEMPT_RT_PRESENCE persistently
sudo grep -qxF 'IGNORE_PREEMPT_RT_PRESENCE=1' /etc/environment || echo 'IGNORE_PREEMPT_RT_PRESENCE=1' | sudo tee -a /etc/environment > /dev/null

# Re-install NVIDIA CUDA drivers
sudo IGNORE_PREEMPT_RT_PRESENCE=1 bash cuda_<VERSION>.run

# Update initramfs (in case RT kernel version is different than current one)
sudo update-initramfs -u -k all
# get /dev/<name> of root directory
df -h /
# update grub on root drive
sudo grub-install /dev/<name>
sudo update-grub

# Add user to realtime group
sudo addgroup realtime
sudo usermod -a -G realtime $(whoami)
# Update realtime group limits
sudo tee -a /etc/security/limits.conf >/dev/null <<'EOF'
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
EOF

# Make sure Performance mode is enabled
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Reboot to load the new RT kernel
sudo reboot
```

### 2) Verify install

```bash
# Verify you're actually on an RT kernel, "-realtime"
uname -r
# should show 1
cat /sys/kernel/realtime

# Check NVIDIA modules are present/loaded
nvidia-smi
lsmod | grep -i nvidia

# Look for DKMS build issues
sudo journalctl -k -b | grep -i -E "nvidia|dkms|module"

# Confirm PREEMPT_RT config
grep PREEMPT_RT /boot/config-$(uname -r)
```

### 3) Additional setup for NUC

See "Step 4: Set up CPU monitoring utilities." from [`deoxys/System Prerequisite`](https://zhuyifengzju.github.io/deoxys_docs/html/installation/system_prerequisite.html).

### Anydesk

Requirements: a dummy HDMI or DisplayPort plug.

```bash
# install anydesk
sudo su root
wget -qO - https://keys.anydesk.com/repos/DEB-GPG-KEY | apt-key add -
echo "deb http://deb.anydesk.com/ all main" > /etc/apt/sources.list.d/anydesk-stable.list
apt update
apt install anydesk
echo <password> | anydesk --set-password
cat /dev/null > ~/.bash_history
anydesk --get-id
anydesk --get-status
# for "ERROR: display_server_not_supported", see https://askubuntu.com/a/1148504

# restart display manager
sudo systemctl restart gdm3

# open a terminal after signing in to anydesk window
sudo xhost +local:
sudo su $USER
tmux new -s anydesk
# detach tmux session

# attach tmux session in another terminal
ssh $USER@$SERVER
tmux attach -t anydesk
xeyes
# should show up in anydesk window
```

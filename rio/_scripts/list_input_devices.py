# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""List USB input devices and watch button presses, for setting up a button pad.

Prints every readable input device with the USB ids and key codes needed to fill in a
station config, then watches for presses so each physical button can be identified.

Usage:
    uv run python -m rio._scripts.list_input_devices
    uv run python -m rio._scripts.list_input_devices --watch 3553:c011
"""

import argparse
import select
import sys
from typing import TYPE_CHECKING

try:
    import evdev
    from evdev import ecodes
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        evdev = None
        ecodes = None

_PERMISSION_HINT = """
No readable input devices found.

This is almost always permissions rather than missing hardware: /dev/input/event* is
owned by root:input, and an SSH session is not granted access the way a local desktop
session is. Install the udev rule:

    sudo cp rio-hw/scripts/setup/buttons/99-rio-buttons.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger

or add yourself to the input group and reconnect the session:

    sudo usermod -aG input "$USER"
"""


def key_name(code: int) -> str:
    """Return a readable name for a Linux input-event code."""
    name = ecodes.KEY.get(code, f"<{code}>")
    # Codes shared by several key names map to a sequence; the first is the common one.
    return name[0] if isinstance(name, (list, tuple)) else name


def parse_usb_id(value: str) -> tuple[int, int]:
    """Parse a "vendor:product" USB id pair given in hex.

    Args:
        value: USB id pair, e.g. "3553:c011".

    Returns:
        The vendor and product ids.
    """
    try:
        vendor, product = value.split(":")
        return int(vendor, 16), int(product, 16)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected a 'vendor:product' hex pair, got {value!r}") from e


def list_devices() -> list:
    """Print a table of readable input devices. Returns the devices found."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    if not devices:
        print(_PERMISSION_HINT)
        return []

    print(f"{'PATH':<20} {'USB ID':<12} NAME")
    print(f"{'-' * 20} {'-' * 12} {'-' * 40}")
    for device in sorted(devices, key=lambda d: d.path):
        usb_id = f"{device.info.vendor:04x}:{device.info.product:04x}"
        keys = ecodes.EV_KEY in device.capabilities()
        marker = "" if keys else "  (no keys)"
        print(f"{device.path:<20} {usb_id:<12} {device.name}{marker}")
    return devices


def watch(devices) -> None:
    """Print key events from the given devices until interrupted."""
    if not devices:
        return
    print("\nPress each button to see its key code. Ctrl-C when done.\n")
    fds = {device.fd: device for device in devices}
    try:
        while True:
            ready, _, _ = select.select(fds, [], [])
            for fd in ready:
                for event in fds[fd].read():
                    if event.type != ecodes.EV_KEY or event.value != 1:
                        continue  # key-down only
                    device = fds[fd]
                    print(f"{device.path:<20} code={event.code:<5} {key_name(event.code)}")
    except KeyboardInterrupt:
        print("\nDone.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch",
        nargs="?",
        const="all",
        metavar="VENDOR:PRODUCT",
        help="Watch for presses. Give a USB id (e.g. 3553:c011) to watch one device.",
    )
    args = parser.parse_args()

    if evdev is None:
        print("evdev is not installed. Install the interfaces extra of rio-hw.")
        return 1

    devices = list_devices()
    if not devices:
        return 1

    if args.watch is None:
        return 0

    if args.watch != "all":
        vendor, product = parse_usb_id(args.watch)
        devices = [d for d in devices if (d.info.vendor, d.info.product) == (vendor, product)]
        if not devices:
            print(f"\nNo device matched USB id {args.watch}.")
            return 1

    watch(devices)
    return 0


if __name__ == "__main__":
    sys.exit(main())

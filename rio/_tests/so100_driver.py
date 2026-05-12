#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Test SO-ARM driver with same waypoints as LeRobot for apples-to-apples comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Import your custom driver
from rio.robots.utils.soarm_driver import SOArmDriver

PORT = "/dev/ttyACM0"
ARM_ID = "leader_L"
URDF_PATH = "./SO101/so101_new_calib.urdf"
OUTPUT_FILE = "soarm_driver_results.json"

# Test positions (all in radians) - SAME AS LEROBOT TEST
TEST_POSITIONS = [
    {
        "name": "Home",
        "joints": [0.0, 0.0, 0.0, 0.0, 0.0],  # [pan, lift, elbow, wrist_flex, wrist_roll]
        "gripper": 0.0,
    },
    {
        "name": "Position 1",
        "joints": [0.0, -45.0, 25.0, 90.0, 0.0],
        "gripper": 100.0,
    },
    # {
    #     "name": "Position 2",
    #     "joints": [-45.0, 30.0, -60.0, 20.0, 90.0],
    #     "gripper": 20.0,
    # },
    # {
    #     "name": "Extended Reach",
    #     "joints": [0.0, -45.0, 90.0, -45.0, 0.0],
    #     "gripper": 50.0,
    # },
]


def compute_ee_pose_dict(ee_pose_array):
    """Convert EEF pose array to dict format matching LeRobot output."""
    position = ee_pose_array[:3]
    orientation = ee_pose_array[3:]

    return {
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "orientation_rotvec": {
            "wx": float(orientation[0]),
            "wy": float(orientation[1]),
            "wz": float(orientation[2]),
        },
        "orientation_degrees": {
            "wx": float(np.rad2deg(orientation[0])),
            "wy": float(np.rad2deg(orientation[1])),
            "wz": float(np.rad2deg(orientation[2])),
        },
    }


def main():
    print("=" * 80)
    print("SO-ARM Driver Position Test (LeRobot Compatible)")
    print("=" * 80)

    # Initialize results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "driver": "SOArmDriver",
            "port": PORT,
            "arm_id": ARM_ID,
            "urdf_path": URDF_PATH,
            "joint_units": "radians",
        },
        "tests": [],
    }

    # Initialize driver
    print("\nInitializing driver...")
    driver = SOArmDriver(
        port="/dev/ttyACM0",
        model="so101",
        baudrate=1_000_000,
        joint_units="radians",
        arm_id=ARM_ID,
    )

    print("Connecting to robot...")
    driver.connect()
    print("✓ Connected\n")
    # breakpoint()

    try:
        for i, test_config in enumerate(TEST_POSITIONS, 1):
            name = test_config["name"]
            commanded_joints = test_config["joints"]
            commanded_gripper = test_config["gripper"]

            print("=" * 80)
            print(f"TEST {i}: {name}")
            print("=" * 80)

            # Initialize test result
            test_result = {
                "test_number": i,
                "name": name,
                "commanded": {},
                "actual": {},
                "errors": {},
            }

            # Build commanded positions dict (matching LeRobot format)
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            for j, joint_name in enumerate(joint_names):
                key = f"{joint_name}.pos"
                test_result["commanded"][key] = float(commanded_joints[j])
            test_result["commanded"]["gripper.pos"] = float(commanded_gripper)

            # Send command
            print("\nSending command:")
            print(f"  Joints: {commanded_joints}")
            print(f"  Gripper: {commanded_gripper}%")

            driver.moveJ(commanded_joints)
            # driver.move_gripper(commanded_gripper)
            time.sleep(2.0)  # Wait for movement

            # Read actual positions
            print("\nReading actual positions...")
            actual_joints = driver.get_joint_positions()
            actual_gripper = driver.get_gripper_position()

            # Store actual positions
            for j, joint_name in enumerate(joint_names):
                key = f"{joint_name}.pos"
                actual_val = float(actual_joints[j])
                commanded_val = commanded_joints[j]
                error = actual_val - commanded_val

                test_result["actual"][key] = actual_val
                test_result["errors"][key] = error

                print(f"  {key}: {actual_val:>7.2f}° (cmd: {commanded_val:>6.1f}°, err: {error:>6.2f}°)")

            # Gripper
            test_result["actual"]["gripper.pos"] = float(actual_gripper)
            test_result["errors"]["gripper.pos"] = float(actual_gripper - commanded_gripper)

            print(
                f"  gripper.pos: {actual_gripper:>7.2f}%"
                f" (cmd: {commanded_gripper:>6.1f}%, err: {actual_gripper - commanded_gripper:>6.2f}%)"
            )

            # Compute end-effector pose
            print("\nComputing end-effector pose...")
            try:
                ee_pose = driver.get_end_effector_pose(as_deg=False)  # Get in radians
                ee_pose_dict = compute_ee_pose_dict(ee_pose)
                test_result["end_effector_pose"] = ee_pose_dict

                print("  Position (m):")
                print(f"    x = {ee_pose_dict['position']['x']:>8.4f}")
                print(f"    y = {ee_pose_dict['position']['y']:>8.4f}")
                print(f"    z = {ee_pose_dict['position']['z']:>8.4f}")
                print("  Orientation (rad):")
                print(f"    wx = {ee_pose_dict['orientation_rotvec']['wx']:>8.4f}")
                print(f"    wy = {ee_pose_dict['orientation_rotvec']['wy']:>8.4f}")
                print(f"    wz = {ee_pose_dict['orientation_rotvec']['wz']:>8.4f}")
                print("  Orientation (deg):")
                print(f"    wx = {ee_pose_dict['orientation_degrees']['wx']:>8.2f}°")
                print(f"    wy = {ee_pose_dict['orientation_degrees']['wy']:>8.2f}°")
                print(f"    wz = {ee_pose_dict['orientation_degrees']['wz']:>8.2f}°")

            except Exception as e:
                print(f"  ⚠️ FK computation failed: {e}")
                test_result["end_effector_pose_error"] = str(e)

            # Add to results
            results["tests"].append(test_result)

            print()
            input("Press ENTER for next position...")
            print()

        print("=" * 80)
        print("All tests complete!")
        print("=" * 80)

    finally:
        print("\nDisconnecting...")
        driver.disconnect()
        print("✓ Disconnected")

    # Save results to JSON
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(results['tests'])}")
    print(f"Output file: {Path(OUTPUT_FILE).absolute()}")

    # Calculate average errors
    if results["tests"]:
        all_errors = []
        for test in results["tests"]:
            for joint, error in test["errors"].items():
                if joint != "gripper.pos":  # Exclude gripper
                    all_errors.append(abs(error))

        if all_errors:
            print("\nJoint Error Statistics:")
            print(f"  Mean:   {np.mean(all_errors):.3f}°")
            print(f"  Median: {np.median(all_errors):.3f}°")
            print(f"  Max:    {np.max(all_errors):.3f}°")
            print(f"  Min:    {np.min(all_errors):.3f}°")


if __name__ == "__main__":
    main()

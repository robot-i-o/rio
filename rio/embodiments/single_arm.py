# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import scipy.spatial.transform as st
from loguru import logger
from rio_hw.middlewares._middleware import Client

from rio.envs.poll import TeleopMode

from ..schema import ActionSpace, Observation
from .base import BaseEmbodiment


@dataclass
class SingleArmObs(Observation):
    proprio_eef: np.ndarray | None = None
    proprio_joints: np.ndarray | None = None
    gripper_position: float | None = None

    hand_pose: np.ndarray | None = None
    hand_joints: np.ndarray | None = None


class SingleArm(BaseEmbodiment):
    def __init__(
        self,
        arm: Client,
        gripper: Client | None = None,
        hand: Client | None = None,
        action_space: str = "TASK_POS",
        **kwargs,
    ):
        self.arm = arm
        self.gripper = gripper
        self.hand = hand
        self.action_space = ActionSpace[action_space]

        # Resolve the node that owns the gripper channel (explicit client, else the
        # arm itself when it exposes moveG). Commanded via moveG in all cases.
        self._gripper = self._resolve_gripper(self.arm, self.gripper)

        self.urdf_path = kwargs.get("urdf_path", "")
        self.__obs_schema__ = SingleArmObs

        # Get number of joints for each arm dynamically
        self.arm_num_joints = self._get_num_joints(self.arm)

        if self.action_space == ActionSpace.TASK_POS:
            self.arm_dim = 6
        else:  # JOINT_POS or JOINT_VEL
            self.arm_dim = self.arm_num_joints

    def move(
        self,
        actions: np.ndarray,
        t_cmd_target: float,
        binarize_gripper: bool = False,
    ):
        parsed = self.parse_action(actions)

        self._dispatch_gripper(self._gripper, parsed["gripper_cmd"], t_cmd_target, binarize=binarize_gripper)
        self._dispatch_arm(self.arm, parsed["arm_cmd"], t_cmd_target)

        if self.hand is not None and parsed["hand_cmd"] is not None:
            self.hand.moveJ(parsed["hand_cmd"].tolist(), t_cmd_target)

    def _get_num_joints(self, arm: Client) -> int:
        if hasattr(arm, "num_joints"):
            return arm.num_joints
        else:
            logger.warning(
                "Arm client does not have 'num_joints' attribute; using state to determine joint count.  \
                    Consider adding 'num_joints' attribute to the client for efficiency."
            )
            state = arm.get_state()
            joint_q = state.get("joint_q", None)
            if joint_q is not None:
                return len(joint_q)
            else:
                raise ValueError("Cannot determine number of joints for the given arm client.")

    def parse_action(self, action):
        # Parse arm command
        arm_cmd = action[: self.arm_dim]
        gripper1_cmd = action[self.arm_dim]

        return {
            "arm_cmd": arm_cmd,
            "gripper_cmd": gripper1_cmd,
            "hand_cmd": None,
        }

    def build_action(self, arm_cmd: np.ndarray, gripper_cmd: np.ndarray | None = None, **kwargs) -> np.ndarray:
        arm_cmd = np.concatenate([arm_cmd, [gripper_cmd]])
        return arm_cmd

    def moveL(self, arm_cmd: np.ndarray, t_cmd_target: float, convert_to_aa: bool = False):
        if convert_to_aa:  # If needed, convert euler angles to axis-angle
            rot = st.Rotation.from_euler("xyz", arm_cmd[3:])
            aa = rot.as_rotvec()
            arm_cmd = np.concatenate([arm_cmd[:3], aa])
        self.arm.moveL(arm_cmd.tolist(), t_cmd_target)

    def moveJ(self, arm_cmd: np.ndarray, t_cmd_target: float):
        self.arm.moveJ(arm_cmd.tolist(), t_cmd_target)

    def get_state(self):
        state = {}
        if self.arm:
            state["arm"] = self.arm.get_state()
        if self.gripper:
            state["gripper"] = self.gripper.get_state()
        if self.hand:
            state["hand"] = self.hand.get_state()
        return state

    def get_obs(self, cams: dict) -> SingleArmObs:
        robot_state = self.get_state()

        # Get proprioceptive states
        eef_pose = robot_state["arm"].get("eef_pose", None)
        joint_q = robot_state["arm"].get("joint_q", None)
        if joint_q is None:
            logger.warning("Robot arm state does not contain joint_q.")

        if "gripper" in robot_state.keys():
            gripper_pos = robot_state["gripper"].get("gripper_position", None)
        else:
            # Integrated gripper: position is published in the arm state.
            gripper_pos = robot_state["arm"].get("gripper_position", 0.0)

        default_proprio = eef_pose if self.action_space == "EEF_POSE" else joint_q
        default_proprio = np.concatenate([default_proprio, [gripper_pos]])

        obs = SingleArmObs(
            proprio=default_proprio,
            proprio_eef=eef_pose,
            proprio_joints=joint_q,
            gripper_position=gripper_pos,
            cameras=cams,
        )
        return obs

    @staticmethod
    def make_teleop_eef_cmd(freq, teleop_mode, delta_pose, target_pose, max_pos_speed, max_rot_speed):
        dpos = delta_pose[:3] * (max_pos_speed / freq)
        drot_xyz = delta_pose[3:] * (max_rot_speed / freq)
        if teleop_mode == TeleopMode.TRANSLATION_2D:
            drot_xyz[:] = 0
            dpos[2] = 0
        elif teleop_mode == TeleopMode.TRANSLATION:
            drot_xyz[:] = 0
        elif teleop_mode == TeleopMode.ROTATION:
            dpos[:] = 0
        elif teleop_mode == TeleopMode.TRANSLATION_ROTATION:
            pass
        else:
            raise RuntimeError(teleop_mode)
        drot = st.Rotation.from_euler("xyz", drot_xyz)
        rot = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
        target_pose[:3] += dpos
        target_pose[3:] = rot
        return target_pose

    @staticmethod
    def make_gello_joint_cmd(freq, gello_joints, target_joints, max_joint_delta=0.02, deadband=0.0):
        command_joints = gello_joints[: len(target_joints)]
        delta = command_joints - target_joints

        if deadband > 0:
            delta = np.where(np.abs(delta) < deadband, 0, delta)

        delta = np.clip(delta, -max_joint_delta, max_joint_delta)
        target_joints[:] = target_joints + delta
        return target_joints

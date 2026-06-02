# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import os
import queue
import traceback

import numpy as np
import scipy.spatial.transform as st
import torch
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory
from rio_hw.node import Node
from threadpoolctl import threadpool_limits

try:
    import onnxruntime as ort

    ORT_IMPORT_ERROR = None
except ImportError as e:
    ORT_IMPORT_ERROR = e

try:
    import gear_sonic
    from gear_sonic.scripts.pico_manager_thread_server import compute_from_body_poses
    from gear_sonic.trl.utils import torch_transform
    from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa

    # compute_human_body_joints inside compute_from_body_poses loads data from a relative path
    # we directly load the data first
    gear_sonic_root = os.path.dirname(os.path.dirname(gear_sonic.__file__))
    f = os.path.join(gear_sonic_root, "gear_sonic/data/human/human_joints_info.pkl")
    torch_transform.human_joints_info = torch.load(f)

    GEAR_SONIC_IMPORT_ERROR = None
except ImportError as e:
    GEAR_SONIC_IMPORT_ERROR = e


# fmt: off
MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
], dtype=np.int32)

ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
    21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
], dtype=np.int32)

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)

G1_ACTION_SCALE = np.array([
    0.3507, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.3507, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.4386, 0.4386,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
], dtype=np.float32)

KP_MJ = np.array([
    99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,
    99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,
    40.1792, 28.5012, 28.5012,
    14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,
    14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,
], dtype=np.float32)

KD_MJ = np.array([
    6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,
    6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,
    2.5579, 1.8144, 1.8144,
    0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
    0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
], dtype=np.float32)
# fmt: on

KPS = tuple(KP_MJ.tolist())
KDS = tuple(KD_MJ.tolist())

SMPL_PARENT_INDICES = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23]


# https://github.com/NVlabs/GR00T-WholeBodyControl/3d3e593369bc1384a8f5d1f22830bf20cb3a6047/gear_sonic/scripts/pico_manager_thread_server.py#L1345-L1403
def _compute_g1_wrist_joints(body_pose: np.ndarray) -> np.ndarray:
    SMPL_L_ELBOW_IDX = 17
    SMPL_L_WRIST_IDX = 19
    SMPL_R_ELBOW_IDX = 18
    SMPL_R_WRIST_IDX = 20

    g1_l_elbow_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _, g1_l_elbow_q_swing = decompose_rotation_aa(body_pose[:, SMPL_L_ELBOW_IDX], g1_l_elbow_axis)

    g1_r_elbow_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _, g1_r_elbow_q_swing = decompose_rotation_aa(body_pose[:, SMPL_R_ELBOW_IDX], g1_r_elbow_axis)

    l_elbow_swing_euler = st.Rotation.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    r_elbow_swing_euler = st.Rotation.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)

    l_wrist_euler = st.Rotation.from_rotvec(body_pose[:, SMPL_L_WRIST_IDX]).as_euler("XYZ", degrees=False)
    r_wrist_euler = st.Rotation.from_rotvec(body_pose[:, SMPL_R_WRIST_IDX]).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    return np.array(
        [
            g1_l_wrist_roll[0],
            g1_r_wrist_roll[0],
            -g1_l_wrist_pitch[0],
            g1_r_wrist_pitch[0],
            g1_l_wrist_yaw[0],
            g1_r_wrist_yaw[0],
        ],
        dtype=np.float32,
    )


class StateBuffer:
    def __init__(self, size: int):
        self.size = size
        self.count = 0
        self.buffers = {
            "q": np.zeros((size, 29), dtype=np.float32),
            "dq": np.zeros((size, 29), dtype=np.float32),
            "last_action": np.zeros((size, 29), dtype=np.float32),
            "ang_vel": np.zeros((size, 3), dtype=np.float32),
            "gravity": np.zeros((size, 3), dtype=np.float32),
            "smpl_joints": np.zeros((size, 72), dtype=np.float32),
            "smpl_root_quat_xyzw": np.zeros((size, 4), dtype=np.float32),
            "wrist_joints": np.zeros((size, 6), dtype=np.float32),
        }
        self.buffers["gravity"][:, 2] = -1.0

    def append(self, state_dict: dict):
        for key, val in state_dict.items():
            if key in self.buffers:
                self.buffers[key] = np.roll(self.buffers[key], -1, axis=0)
                self.buffers[key][-1] = val
        self.count = min(self.size, self.count + 1)

    def get_history(self, key: str, num_frames: int) -> np.ndarray:
        return self.buffers[key][-num_frames:].flatten()


class GearSonic(Node):
    __api__ = [
        "step",
        "get_action_latest",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        model_dir: str = "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release",
        freq: int = 50,
        max_buffer_size: int = 30,
        max_queue_size: int = 50,
        **kwargs,
    ):
        if ORT_IMPORT_ERROR is not None:
            raise ImportError("onnxruntime is required to run GearSonic.") from ORT_IMPORT_ERROR
        self.freq = freq
        self.model_dir = model_dir
        self.last_timestamp = time.now()
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, max_queue_size=max_queue_size, **kwargs)

    def __post_init__(self):
        self.last_action_array = np.zeros(29, dtype=np.float32)

        self.example_request = {
            "q": np.zeros(29, dtype=np.float32),
            "dq": np.zeros(29, dtype=np.float32),
            "quaternion": np.zeros(4, dtype=np.float32),
            "gyroscope": np.zeros(3, dtype=np.float32),
            "mode": "joint",  # joint | smpl
            # SMPL body tracking mode
            "body_tracking": False,
            "raw_body_pos": np.zeros((24, 3), dtype=np.float32),
            "raw_body_quat": np.zeros((24, 4), dtype=np.float32),
            # Joint position mode
            "joint_pos": np.zeros((10, 29), dtype=np.float32),
            "root_quat": np.zeros((10, 4), dtype=np.float32),
            "ref_fi": 0,
        }

        self.example_data = {
            "action": np.zeros(29, dtype=np.float32),
            "timestamp": time.now(),
        }

        self.worker = None
        self.run = self.pubreq

        super().__post_init__()

    def pubreq(self):
        threadpool_limits(1)

        logger.info(f"SonicPolicyInterface: Initializing ONNX models from {self.model_dir}...")

        encoder_path = os.path.join(self.model_dir, "model_encoder.onnx")
        policy_path = os.path.join(self.model_dir, "model_decoder.onnx")

        self.encoder_session = ort.InferenceSession(encoder_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.policy_session = ort.InferenceSession(policy_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        logger.info("SonicPolicyInterface: Initialization complete.")

        self.pub_ready_event.set()

        state_buffer = StateBuffer(size=10)
        heading_initialized = False
        heading_delta_rot = None

        encoder_mode = None
        dt = 1.0 / self.freq
        t_start = None
        it = 0

        try:
            while not self.exit_event.is_set():
                if t_start is None:
                    t_start = time.now()

                t_cycle_end = t_start + (it + 1) * dt
                try:
                    reqs = self.request_queue.get_all()
                    if len(reqs) == 0:
                        time.precise_wait(t_cycle_end)
                        it += 1
                        continue

                    if isinstance(reqs, dict):
                        reqs = [{k: v[-1] for k, v in reqs.items()}]

                    data = reqs[-1]

                    joint_q = (np.array(data["q"]) - DEFAULT_ANGLES)[MUJOCO_TO_ISAACLAB].astype(np.float32)
                    joint_dq = np.array(data["dq"])[MUJOCO_TO_ISAACLAB].astype(np.float32)

                    ang_vel = data.get("gyroscope", np.zeros(3, dtype=np.float32))

                    quat_xyzw = data.get("quaternion", np.array([0, 0, 0, 1], dtype=np.float32))
                    r = st.Rotation.from_quat(quat_xyzw)
                    gravity_dir = r.inv().apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)

                    # reinitialize if encoder mode change
                    if data["encoder_mode"] != encoder_mode:
                        heading_initialized = False
                    encoder_mode = data["encoder_mode"]

                except queue.Empty:
                    time.precise_wait(t_cycle_end)
                    it += 1
                    continue

                if encoder_mode not in ["joint", "smpl"]:
                    logger.warning("encoder_mode not set, skipping")
                    time.precise_wait(t_cycle_end)
                    it += 1
                    continue

                state_dict = {
                    "q": joint_q,
                    "dq": joint_dq,
                    "last_action": self.last_action_array,
                    "ang_vel": ang_vel,
                    "gravity": gravity_dir,
                }

                if encoder_mode == "smpl":
                    # transform smpl data using the pipeline already in GEAR-SONIC
                    # NOTE: currently we don't do any SMPL interpolation
                    if GEAR_SONIC_IMPORT_ERROR is not None:
                        raise ImportError("gear_sonic is required for raw body tracking mode.") from GEAR_SONIC_IMPORT_ERROR

                    positions = np.array(data["raw_body_pos"])
                    global_quats = np.array(data["raw_body_quat"])

                    body_poses_np = np.zeros((24, 7), dtype=np.float32)
                    body_poses_np[:, :3] = positions
                    body_poses_np[:, 3:] = global_quats

                    latest_data = compute_from_body_poses(SMPL_PARENT_INDICES, "cpu", body_poses_np)

                    smpl_joints_local_np = latest_data["smpl_joints_local"].numpy()[0].astype(np.float32).flatten()
                    ref_root_quat_wxyz = latest_data["global_orient_quat"].numpy()[0].astype(np.float32)
                    ref_root_quat_xyzw = ref_root_quat_wxyz[[1, 2, 3, 0]]

                    _body_pose = latest_data["smpl_pose"].numpy()[0, :63].astype(np.float32).reshape(-1, 21, 3)
                    wrist_joints = _compute_g1_wrist_joints(_body_pose)

                    state_dict.update(
                        {
                            "smpl_joints": smpl_joints_local_np,
                            "smpl_root_quat_xyzw": ref_root_quat_xyzw,
                            "wrist_joints": wrist_joints,
                        }
                    )
                    if not heading_initialized:
                        state_buffer.buffers["smpl_joints"][:] = smpl_joints_local_np
                        state_buffer.buffers["smpl_root_quat_xyzw"][:] = ref_root_quat_xyzw
                        state_buffer.buffers["wrist_joints"][:] = wrist_joints
                else:
                    ref_root_quat_xyzw = np.array(data["root_quat"], dtype=np.float32)[0][[1, 2, 3, 0]]
                    ref_fi = data["ref_fi"]

                state_buffer.append(state_dict)

                if not heading_initialized:
                    # Calculate yaw-only delta between state and reference base rotation
                    base_rot = st.Rotation.from_quat(quat_xyzw)
                    base_x, base_y, _ = base_rot.apply([1.0, 0.0, 0.0])
                    base_yaw = np.arctan2(base_y, base_x)

                    ref_rot = st.Rotation.from_quat(ref_root_quat_xyzw)
                    ref_x, ref_y, _ = ref_rot.apply([1.0, 0.0, 0.0])
                    ref_yaw = np.arctan2(ref_y, ref_x)

                    heading_delta_rot = st.Rotation.from_euler("z", base_yaw - ref_yaw)
                    if encoder_mode == "smpl":
                        heading_initialized = True

                if encoder_mode == "joint":
                    encoder_obs = self._build_encoder_input_joint(
                        data=data,
                        quat_xyzw=quat_xyzw,
                        heading_delta_rot=heading_delta_rot,
                    )
                else:
                    encoder_obs = self._build_encoder_input_smpl(
                        quat_xyzw=quat_xyzw,
                        state_buffer=state_buffer,
                        heading_delta_rot=heading_delta_rot,
                    )

                enc_in = np.expand_dims(encoder_obs, axis=0)
                enc_out = self.encoder_session.run(None, {"obs_dict": enc_in})[0]
                token_state = enc_out[0]

                policy_obs = self._build_decoder_input(token_state, state_buffer)

                pol_in = np.expand_dims(policy_obs, axis=0)
                pol_out = self.policy_session.run(None, {"obs_dict": pol_in})[0]
                action = pol_out[0]

                self.last_action_array = action.copy()

                action_mujoco = action[ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE + DEFAULT_ANGLES
                target_dof_pos_actuator_map = action_mujoco

                action = {"action": target_dof_pos_actuator_map, "timestamp": time.now()}
                self.ring_buffer.put(action)

                time.precise_wait(t_cycle_end)
                it += 1

        except KeyboardInterrupt:
            pass
        except Exception as e:
            traceback.print_exc()
            print(f"[SonicPolicyInterface] Interrupted: {e}")

    def _build_encoder_input_joint(
        self,
        data: dict,
        quat_xyzw: np.ndarray,
        heading_delta_rot: st.Rotation,
    ) -> np.ndarray:
        out = np.zeros(1762, dtype=np.float32)

        joint_pos = np.array(data["joint_pos"], dtype=np.float32).copy()
        root_quat = np.array(data["root_quat"], dtype=np.float32)
        ref_fi = data["ref_fi"]

        N = len(joint_pos)
        fi = np.minimum(N - 1, ref_fi + np.arange(10) * 5)
        fi_next = np.minimum(N - 1, fi + 1)

        jp = joint_pos[fi]
        jp1 = joint_pos[fi_next]

        out[4:294] = jp[:, MUJOCO_TO_ISAACLAB].flatten()  # motion_joint_positions_10frame_step5
        out[294:584] = (
            (jp1[:, MUJOCO_TO_ISAACLAB] - jp[:, MUJOCO_TO_ISAACLAB]) * 50
        ).flatten()  # motion_joint_velocities_10frame_step5

        new_ref_rot = heading_delta_rot * st.Rotation.from_quat(root_quat[fi], scalar_first=True)
        base_to_ref = st.Rotation.from_quat(quat_xyzw).inv() * new_ref_rot
        out[601:661] = base_to_ref.as_matrix()[:, :, :2].flatten()  # motion_anchor_orientation_10frame_step5

        return out

    def _build_encoder_input_smpl(
        self,
        quat_xyzw: np.ndarray,
        state_buffer: StateBuffer,
        heading_delta_rot: st.Rotation,
    ) -> np.ndarray:
        out = np.zeros(1762, dtype=np.float32)
        out[0:4] = [2.0, 0.0, 0.0, 0.0]

        smpl_joints_hist = state_buffer.get_history("smpl_joints", 10)
        wrist_joints_hist = state_buffer.get_history("wrist_joints", 10)
        smpl_root_quats = state_buffer.get_history("smpl_root_quat_xyzw", 10).reshape(10, 4)

        new_ref_rot = heading_delta_rot * st.Rotation.from_quat(smpl_root_quats)
        base_to_ref = st.Rotation.from_quat(quat_xyzw).inv() * new_ref_rot

        out[922:1642] = smpl_joints_hist  # smpl_joints_hist
        out[1642:1702] = base_to_ref.as_matrix()[:, :, :2].flatten()  # smpl_anchor_ori
        out[1702:1762] = wrist_joints_hist  # wrist_joints_hist

        return out

    def _build_decoder_input(self, token_state: np.ndarray, state_buffer: StateBuffer) -> np.ndarray:
        """Build the input for the policy decoder session by concatenating history states."""
        his_ang_vel = state_buffer.get_history("ang_vel", 10)
        his_q = state_buffer.get_history("q", 10)
        his_dq = state_buffer.get_history("dq", 10)
        his_last_actions = state_buffer.get_history("last_action", 10)
        his_gravity = state_buffer.get_history("gravity", 10)

        return np.concatenate(
            [
                token_state,
                his_ang_vel,
                his_q,
                his_dq,
                his_last_actions,
                his_gravity,
            ]
        ).astype(np.float32)

    def step(self, encoder_data: dict, decoder_data: dict, time_req: float):
        self.request_queue.put(
            {
                **encoder_data,
                **decoder_data,
                "time": time_req,
            }
        )

    def get_action_latest(self) -> np.ndarray | None:
        data = self.ring_buffer.get()
        if data is None:
            return None

        if not isinstance(data, dict) or data.get("timestamp") is None or data["timestamp"] <= self.last_timestamp:
            return None

        return data["action"]


def GearSonicServer(mw, *args, **kwargs):
    return ServerFactory(mw, GearSonic, *args, **kwargs)


def GearSonicClient(mw, *args, **kwargs):
    return ClientFactory(mw, GearSonic, *args, **kwargs)

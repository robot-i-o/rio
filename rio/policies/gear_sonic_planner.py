# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import os
import queue
from enum import Enum, auto

import numpy as np
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory
from rio_hw.node import Node
from rio_hw.request import Request
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from threadpoolctl import threadpool_limits

try:
    import onnxruntime as ort

    ORT_IMPORT_ERROR = None
except ImportError as e:
    ORT_IMPORT_ERROR = e


class RequestType(Enum):
    INIT_STATE = auto()
    REPLAN = auto()


class SonicPlanner:
    """
    Kinematic planner from GEAR-SONIC.

    Implemented based on https://nvlabs.github.io/GR00T-WholeBodyControl/references/planner_onnx.html
    """

    def __init__(self):
        self.session = None
        self.context_mujoco_qpos = None
        self.active_motion = None
        self.current_frame = 0

        self._interp_pos = None
        self._interp_joints = None
        self._slerp = None
        self._times_30hz = None

    def load(self, model_path: str, providers: list[str] | None = None):
        if ORT_IMPORT_ERROR is not None:
            raise ImportError("onnxruntime is required to run the kinematic planner.") from ORT_IMPORT_ERROR

        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

    def init_state(self, initial_qpos: np.ndarray):
        """Initializes the 4-frame context array from the robot's starting position."""
        self.context_mujoco_qpos = np.tile(initial_qpos, (4, 1)).astype(np.float32)
        self.active_motion = None
        self.current_frame = 0

    def replan(self, inputs: dict):
        if self.session is None:
            raise ValueError("ONNX session not loaded. Call load() first.")
        if self.context_mujoco_qpos is None:
            raise ValueError("Planner context state not initialized. Call init_state() first.")

        inputs["context_mujoco_qpos"] = np.expand_dims(self.context_mujoco_qpos, axis=0).astype(np.float32)
        mujoco_qpos, num_pred_frames = self.session.run(None, inputs)

        num_pred = int(num_pred_frames[0])
        valid_qpos = mujoco_qpos[0, :num_pred, :]

        # interpolate from 30hz to 50hz
        num_frames_50hz = int(num_pred * 50 / 30)
        self._times_30hz = np.arange(num_pred) / 30.0
        times_50hz = np.arange(num_frames_50hz) / 50.0

        self._interp_pos = interp1d(self._times_30hz, valid_qpos[:, 0:3], axis=0, fill_value="extrapolate")
        resampled_pos = self._interp_pos(times_50hz)

        self._interp_joints = interp1d(self._times_30hz, valid_qpos[:, 7:36], axis=0, fill_value="extrapolate")
        resampled_joints = self._interp_joints(times_50hz)

        quats_30hz = valid_qpos[:, 3:7]
        scipy_quats = quats_30hz[:, [1, 2, 3, 0]]

        self._slerp = Slerp(self._times_30hz, R.from_quat(scipy_quats))
        times_50hz_clamped = np.clip(times_50hz, self._times_30hz[0], self._times_30hz[-1])
        resampled_rotations = self._slerp(times_50hz_clamped)
        resampled_quats = resampled_rotations.as_quat()[:, [3, 0, 1, 2]]

        new_motion = np.hstack([resampled_pos, resampled_quats, resampled_joints])

        # blend with previous motion
        if self.active_motion is not None and self.current_frame < len(self.active_motion):
            old_remainder = self.active_motion[self.current_frame :]
            blend_frames = min(8, len(old_remainder), len(new_motion))

            for i in range(blend_frames):
                w_new = i / 7.0 if blend_frames > 1 else 1.0
                w_old = 1.0 - w_new

                new_motion[i, 0:3] = w_old * old_remainder[i, 0:3] + w_new * new_motion[i, 0:3]
                new_motion[i, 7:36] = w_old * old_remainder[i, 7:36] + w_new * new_motion[i, 7:36]

                q_old = old_remainder[i, 3:7][[1, 2, 3, 0]]
                q_new = new_motion[i, 3:7][[1, 2, 3, 0]]
                slerp_blend = Slerp([0, 1], R.from_quat(np.vstack([q_old, q_new])))
                new_motion[i, 3:7] = slerp_blend([w_new])[0].as_quat()[[3, 0, 1, 2]]

        self.active_motion = new_motion
        self.current_frame = 0

    def get_motion_data(self) -> dict:
        if self.active_motion is None:
            raise ValueError("Active motion data is empty. Run replan() first.")
        return {
            "frames": len(self.active_motion),
            "joint_pos": self.active_motion[:, 7:],
            "root_quat": self.active_motion[:, 3:7],
        }

    def update_context(self):
        """Samples the active motion to prepare the 4-frame context for the next replan."""

        gen_frame = min(self.current_frame + 2, len(self.active_motion) - 1)
        t_start = gen_frame / 50.0

        context_times = t_start + np.arange(4) / 30.0
        context_times = np.clip(context_times, self._times_30hz[0], self._times_30hz[-1])

        ctx_pos = self._interp_pos(context_times)
        ctx_joints = self._interp_joints(context_times)
        ctx_quats = self._slerp(context_times).as_quat()[:, [3, 0, 1, 2]]

        self.context_mujoco_qpos = np.hstack([ctx_pos, ctx_quats, ctx_joints])

    def step(self):
        if self.active_motion is not None and self.current_frame < len(self.active_motion) - 1:
            self.current_frame += 1


class GearSonicPlanner(Node):
    __api__ = [
        "init_state",
        "replan",
        "get_motion_data",
        "step",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        model_dir: str = "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2",
        freq: int = 50,
        max_buffer_size: int = 30,
        max_queue_size: int = 50,
        **kwargs,
    ):
        if ORT_IMPORT_ERROR is not None:
            raise ImportError("onnxruntime is required for GearSonicPlanner.") from ORT_IMPORT_ERROR

        self.model_dir = model_dir
        self.freq = freq
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, max_queue_size=max_queue_size, **kwargs)

    def __post_init__(self):
        self.example_request = {
            "type": next(iter(RequestType)).value,
            "target_vel": np.zeros(1, dtype=np.float32),
            "mode": np.zeros(1, dtype=np.int64),
            "movement_direction": np.zeros((1, 3), dtype=np.float32),
            "facing_direction": np.zeros((1, 3), dtype=np.float32),
            "height": np.zeros(1, dtype=np.float32),
            "initial_qpos": np.zeros(36, dtype=np.float32),
        }

        self.example_data = {
            "frames": 0,
            "joint_pos": np.zeros((10, 29), dtype=np.float32),
            "root_quat": np.zeros((10, 4), dtype=np.float32),
            "ref_fi": 0,
            "timestamp": time.now(),
        }

        self.worker = None
        self.run = self.pubreq
        super().__post_init__()

    def init_state(self, initial_qpos: np.ndarray):
        req = {
            "type": RequestType.INIT_STATE.value,
            "initial_qpos": np.array(initial_qpos),
        }
        self.request_queue.put(req)

    def replan(
        self,
        target_vel: np.ndarray,
        mode: np.ndarray,
        movement_direction: np.ndarray,
        facing_direction: np.ndarray,
        height: np.ndarray,
    ):
        req = {
            "type": RequestType.REPLAN.value,
            "target_vel": np.array(target_vel),
            "mode": np.array(mode),
            "movement_direction": np.array(movement_direction),
            "facing_direction": np.array(facing_direction),
            "height": np.array(height),
        }
        self.request_queue.put(req)

    def get_motion_data(self) -> dict | None:
        return self.ring_buffer.get()

    def pubreq(self):
        threadpool_limits(1)

        planner_path = os.path.join(self.model_dir, "planner_sonic.onnx")
        planner = SonicPlanner()
        logger.info(f"GearSonicPlanner: Loading kinematic planner model from {planner_path}...")
        planner.load(planner_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        logger.info("GearSonicPlanner: Kinematic planner initialized.")

        self.pub_ready_event.set()

        # default to standing
        planner_inputs = {
            "target_vel": np.array([-1.0], dtype=np.float32),
            "mode": np.array([0], dtype=np.int64),
            "movement_direction": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "facing_direction": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "height": np.array([-1.0], dtype=np.float32),
            "random_seed": np.array([1234], dtype=np.int64),
            "has_specific_target": np.array([[0]], dtype=np.int64),
            "specific_target_positions": np.zeros([1, 4, 3], dtype=np.float32),
            "specific_target_headings": np.zeros([1, 4], dtype=np.float32),
            "allowed_pred_num_tokens": np.ones([1, 11], dtype=np.int64),
        }

        dt = 1.0 / self.freq
        t_start = None
        it = 0
        last_replan_time = None
        replan_interval = 1.0

        try:
            while not self.exit_event.is_set():
                if t_start is None:
                    t_start = time.now()
                current_time = t_start + it * dt
                t_cycle_end = t_start + (it + 1) * dt

                try:
                    reqs = self.request_queue.get_all()
                    if isinstance(reqs, dict):
                        reqs = [{k: reqs[k][i] for k in reqs.keys()} for i in range(len(reqs["type"]))]
                except queue.Empty:
                    reqs = []

                if len(reqs) > 0:
                    for r in reqs:
                        req = Request(RequestType(r.pop("type")), r)
                        if req.type == RequestType.INIT_STATE:
                            init_qpos = req.params["initial_qpos"]
                            planner.init_state(init_qpos)
                        elif req.type == RequestType.REPLAN:
                            for key in ["target_vel", "mode", "movement_direction", "facing_direction", "height"]:
                                if key in req.params:
                                    planner_inputs[key] = req.params[key]

                    if planner_inputs["mode"] == 2: # runnning
                        replan_interval = 0.1
                    elif planner_inputs["mode"] in [8, 14]: # crawling
                        replan_interval = 0.2
                    else:
                        replan_interval = 1.0 

                    planner.replan(planner_inputs)
                    motion_data = planner.get_motion_data()
                    self.ring_buffer.put({**motion_data, "timestamp": time.now()})

                    last_replan_time = current_time

                if planner.context_mujoco_qpos is None:
                    time.precise_wait(t_cycle_end)
                    it += 1
                    continue

                if last_replan_time is None or (current_time - last_replan_time) >= replan_interval:
                    planner.update_context()
                    planner.replan(planner_inputs)
                    motion_data = planner.get_motion_data()
                    self.ring_buffer.put({**motion_data, "timestamp": time.now()})

                    last_replan_time = current_time

                planner.step()

                time.precise_wait(t_cycle_end)
                it += 1
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.exception(f"GearSonicPlanner background thread error: {e}")


def GearSonicPlannerServer(mw, *args, **kwargs):
    return ServerFactory(mw, GearSonicPlanner, *args, **kwargs)


def GearSonicPlannerClient(mw, *args, **kwargs):
    return ClientFactory(mw, GearSonicPlanner, *args, **kwargs)

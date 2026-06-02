import multiprocessing as mp
from contextlib import nullcontext

import mujoco
import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env


class MujocoSimClient:
    def __init__(
        self,
        xml_path: str = "third_party/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml",
        motor_kp: tuple | None = None,
        motor_kd: tuple | None = None,
        robot_iface: str = "",
        freq: int = 200,
    ):
        self.num_joints = 29
        self.xml_path = xml_path
        self.motor_kp = np.array(motor_kp) if motor_kp is not None else np.array(KPS)
        self.motor_kd = np.array(motor_kd) if motor_kd is not None else np.array(KDS)

    def __enter__(self):
        import mujoco
        import mujoco.viewer
        from rio.policies.gear_sonic import DEFAULT_ANGLES

        logger.info(f"Loading MuJoCo model from {self.xml_path}...")
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mj_model.opt.timestep = 1.0 / 200  # 200 Hz
        quat = self.mj_data.qpos[3:7] + 0.0 * np.random.randn(4)
        self.mj_data.qpos[3:7] = quat / np.sqrt(np.mean(quat**2))
        self.mj_data.qpos[7:] = DEFAULT_ANGLES + 0.2 * np.random.randn(29)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        logger.info("MuJoCo passive viewer launched successfully.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the viewer and stops the simulation."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.mj_model = None
        self.mj_data = None
        logger.info("MuJoCo simulation shut down.")

    def get_state(self) -> dict:
        """Returns the simulated state of the robot."""
        if self.mj_data is None:
            return {}

        qpos_j = self.mj_data.qpos[7 : 7 + self.num_joints].copy()
        qvel_j = self.mj_data.qvel[6 : 6 + self.num_joints].copy()

        qw, qx, qy, qz = self.mj_data.qpos[3:7]
        quaternion = np.array([qx, qy, qz, qw], dtype=np.float32)

        gyroscope = self.mj_data.qvel[3:6].copy()

        return {
            "joint_q": qpos_j,
            "joint_qd": qvel_j,
            "imu_quat": quaternion,
            "imu_gyro": gyroscope,
        }

    def moveJ(self, action: list | np.ndarray, t_cmd_target: float) -> None:
        """Applies targets and steps the simulation.

        Args:
            action: Target joint angles.
            t_cmd_target: Target timestamp for control loop sync.
        """
        if self.mj_data is None:
            return

        targets = np.array(action, dtype=np.float32)

        qpos_j = self.mj_data.qpos[7 : 7 + self.num_joints]
        qvel_j = self.mj_data.qvel[6 : 6 + self.num_joints]

        torque = self.motor_kp * (targets - qpos_j) - self.motor_kd * qvel_j

        self.mj_data.ctrl[:] = torque
        mujoco.mj_step(self.mj_model, self.mj_data)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()


def teleop_humanoid(args, env, controller, planner, teleop=None):
    freq = args.freq
    dt = 1.0 / freq

    # Start Loop
    paused = False
    encoder_mode = "joint"
    smpl_data = None

    try:
        default_dof_pos = np.array(args.default_dof_pos)

        if not args.sim:
            input("Press enter to go to initial position...")

            t_start = time.now()
            env.set_start_time(t_start)
            it = 0

            for _ in range(int(args.duration * args.freq)):
                t_cycle_end = t_start + (it + 1) * dt
                obs = env.get_state().observation
                ratio = np.clip(it / (args.duration * args.freq), 0.0, 1.0)
                action = (1.0 - ratio) * obs.q + ratio * default_dof_pos

                env.move(action, t_cycle_end)

                time.precise_wait(t_cycle_end)
                it += 1

        t_start = time.now()
        env.set_start_time(t_start)

        # Initialize the planner's state from environment
        state = env.get_state()
        # Z-height passed to planner is hardcoded at startup
        planner.init_state(np.concatenate([[0.0, 0.0, 0.789], state.observation.quaternion[[3, 0, 1, 2]], state.observation.q]))

        input("Press enter to start controller...")
        t_start = time.now()
        env.set_start_time(t_start)
        it = 0

        while True:
            t_cycle_end = t_start + (it + 1) * dt

            if teleop:
                teleop_data = teleop.get_state()
            else:
                teleop_data = {}

            if teleop_data.get("left_x"):
                logger.warning("EMERGENCY STOP")
                break

            if encoder_mode == "joint" and teleop_data.get("right_b"):
                encoder_mode = "smpl"
                logger.info("Switched from planner to teleop device tracking")
            else:
                if (not paused) and (smpl_data is not None) and teleop_data.get("right_a"):
                    paused = True
                    logger.info("Paused SMPL tracking")

                if paused and teleop_data.get("right_b"):
                    paused = False
                    logger.info("Resumed SMPL tracking")

            state = env.get_state()
            if state is None or state.observation is None:
                time.precise_wait(t_cycle_end)
                it += 1
                continue

            decoder_data = {
                "q": state.observation.q,
                "dq": state.observation.dq,
                "quaternion": state.observation.quaternion,
                "gyroscope": state.observation.gyroscope,
            }

            if encoder_mode == "joint":
                motion_data = planner.get_motion_data()
                if not motion_data or motion_data.get("frames", 0) == 0:
                    time.precise_wait(t_cycle_end)
                    it += 1
                    continue
                encoder_data = {
                    "encoder_mode": encoder_mode,
                    "frames": motion_data["frames"],
                    "joint_pos": motion_data["joint_pos"],
                    "root_quat": motion_data["root_quat"],
                    "ref_fi": int((time.now() - motion_data["timestamp"]) * 50),
                }
            else:
                if not paused:
                    encoder_data = {
                        "encoder_mode": encoder_mode,
                        "body_tracking": teleop_data.get("body_tracking"),
                        "raw_body_pos": teleop_data.get("raw_body_pos"),
                        "raw_body_quat": teleop_data.get("raw_body_quat"),
                    }

            controller.step(encoder_data=encoder_data, decoder_data=decoder_data, time_req=t_cycle_end)

            action = controller.get_action_latest()

            if action is None:
                time.precise_wait(t_cycle_end)
                it += 1
                continue

            env.move(action, t_cycle_end)

            time.precise_wait(t_cycle_end)
            it += 1

    except KeyboardInterrupt:
        pass


def main(args):
    servers, clients, env = make_env(args)

    if getattr(args, "sim", False):
        clients["humanoid"] = lambda: MujocoSimClient(**args.humanoid_cfg.cfg)
        if "humanoid" in servers:
            del servers["humanoid"]

    with ServerManager(args.mw, list(servers.values())):
        with (
            env,
            clients["teleop"]() if clients.get("teleop") else nullcontext() as teleop,
            clients["controller"]() as controller,
            clients["planner"]() as planner,
        ):
            teleop_humanoid(args, env, controller, planner, teleop)


if __name__ == "__main__":
    from examples import get_station_cfg

    args = tyro.cli(get_station_cfg())
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)

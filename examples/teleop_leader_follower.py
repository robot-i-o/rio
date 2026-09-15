import multiprocessing as mp
from contextlib import nullcontext

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.control import ButtonPoller, Command, RecordingSession, optional_buttons, resolve_bindings
from rio.envs.env import make_env


def check_alignment(teleop_joints, robot_joints, max_joint_delta=0.8):
    """Verify leader device is aligned with robot before starting teleoperation."""
    joint_delta = np.abs(teleop_joints - robot_joints)
    max_delta = joint_delta.max()

    print("\nJoint-by-joint alignment:")
    print("Joint | Leader (°) | Robot (°) | Delta (°)")
    print("------|------------|-----------|----------")
    for i in range(len(teleop_joints)):
        leader_deg = np.rad2deg(teleop_joints[i])
        robot_deg = np.rad2deg(robot_joints[i])
        delta_deg = np.rad2deg(joint_delta[i])
        status = "❌" if joint_delta[i] > max_joint_delta else "✅"
        print(f"  {i + 1}   | {leader_deg:9.1f}  | {robot_deg:8.1f}  | {delta_deg:6.1f} {status}")

    if max_delta > max_joint_delta:
        max_joint_idx = np.argmax(joint_delta)
        print(f"Joint {max_joint_idx + 1} has the largest delta: {np.rad2deg(max_delta):.1f}°")
        raise RuntimeError("Align leader device to match robot initial pose")

    print(f"\n✓ Alignment OK (max delta: {np.rad2deg(max_delta):.1f}°)")


DEFAULT_BINDINGS = {
    "a": Command.TOGGLE_EPISODE,
    "c": Command.DISCARD_LAST,
}


def teleop_leader_follower(args, env, teleop, teleop2=None, buttons=None, visualizer=None):
    """Unified leader-follower teleoperation loop for single or bimanual arms."""
    primary_arm = getattr(env.robot, "arm", None) or getattr(env.robot, "arm1", None)
    arm_target_joint_q = primary_arm.get_state()["joint_q"].copy() if primary_arm else None

    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
        logger.debug(f"Visualizer: set robot model to {env.robot.urdf_path}")

    if getattr(args, "check_alignment", False):
        print("Checking leader alignment...")
        check_alignment(teleop.get_state()["joint_q"], arm_target_joint_q)

    session = RecordingSession(env.recorder)
    poller = ButtonPoller(buttons, resolve_bindings(args, DEFAULT_BINDINGS, "button_bindings"))

    if buttons is not None:
        print(f"\nInstruction: {args.instruction}")
        print("Teleoperation is live. Use the buttons to control recording; Ctrl-C to exit.")
    else:
        # Without buttons there is nothing to start or stop an episode, so record the
        # whole session as one trajectory and save it on exit.
        input(f"Instruction: {args.instruction}\nPress Enter to start")
        session.handle(Command.TOGGLE_EPISODE)
    time.sleep(getattr(args, "startup_delay", 0.0))

    freq = args.freq
    dt = 1.0 / freq
    command_latency = dt / 2
    t_start = time.now()
    it = 0
    env.set_start_time(t_start)
    env.set_instruction(args.instruction)

    try:
        while True:
            t_cycle_end = t_start + (it + 1) * dt
            t_sample = t_cycle_end - command_latency
            t_cmd_target = t_cycle_end + dt

            time.precise_wait(t_sample)

            # Apply operator commands from the physical buttons
            for command in poller.poll():
                session.handle(command)
            session.refresh()

            # Skip control while recorder is actively saving
            if env.recorder and env.recorder.get_state().get("is_saving", False):
                time.precise_wait(t_cycle_end)
                it += 1
                continue

            # Read leader state
            teleop_state = teleop.get_state()
            joint_q = teleop_state["joint_q"]
            gripper_pos = teleop_state["gripper_position"]

            if getattr(args, "invert_gripper", False) and gripper_pos is not None:
                gripper_pos = 1 - gripper_pos

            # Bimanual: concatenate second leader
            if teleop2 is not None:
                teleop2_state = teleop2.get_state()
                joint_q = np.concatenate([joint_q, teleop2_state["joint_q"]])
                gripper_pos = np.array([gripper_pos, teleop2_state["gripper_position"]])

            # Optional joint smoothing (e.g. for Gello-style continuous hand tracking)
            if getattr(args, "use_leader_smoothing", False) and getattr(env.robot, "arm", None):
                arm_target_joint_q = env.robot.make_gello_joint_cmd(freq, joint_q, arm_target_joint_q)
                joint_q = arm_target_joint_q

            if primary_arm and gripper_pos is not None:
                action = env.build_action(joint_q, gripper_pos)
                t_target = t_cmd_target + args.arm_latency
                env.move(action, t_target)

            step = env.get_state(action=action)

            session.record_step(step)
            if visualizer:
                visualizer.log_env_state("env", step)

            time.precise_wait(t_cycle_end)
            it += 1

    except KeyboardInterrupt:
        pass
    finally:
        session.close()


def main(args):
    """Initialize environment and run leader-follower teleoperation."""
    args.action_space = "joint_pos"

    kwargs = {}
    teleop_module = getattr(args, "teleop_module", None)
    teleop2_module = getattr(args, "teleop2_module", None)
    if teleop_module:
        kwargs["teleop_module"] = teleop_module
    if teleop2_module:
        kwargs["teleop2_module"] = teleop2_module

    servers, clients, env = make_env(args, **kwargs)

    with ServerManager(args.mw, list(servers.values())):
        with (
            env,
            clients["teleop"]() as teleop,
            (clients.get("teleop2") or (lambda: nullcontext()))() as teleop2,
            optional_buttons(clients.get("buttons")) as buttons,
            (clients.get("visualizer") or (lambda: nullcontext()))() as visualizer,
        ):
            try:
                teleop_leader_follower(
                    args,
                    env,
                    teleop,
                    teleop2=teleop2,
                    buttons=buttons,
                    visualizer=visualizer,
                )
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    from examples import get_station_cfg

    args = tyro.cli(get_station_cfg())
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)

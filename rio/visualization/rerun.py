# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import queue
from dataclasses import fields
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory
from rio_hw.node import Node
from rio_hw.request import Request

from ..schema import Observation, Step
from .utils import display_arrows_traj, display_frame

try:
    import rerun as rr
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        rr = None  # type: ignore

try:
    import mujoco
    import rerun_loader_mjcf
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        mujoco = None  # type: ignore

try:
    from robot_descriptions.loaders.mujoco import load_robot_description

    ROBOT_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    load_robot_description = None  # type: ignore
    ROBOT_DESCRIPTIONS_AVAILABLE = False


def log_obs(observation: Observation, entity_path: str = "world"):
    # Log all fields
    for field in fields(observation):
        field_name = field.name
        field_value = getattr(observation, field_name)

        if field_value is None:
            continue

        if field_name == "cameras":
            continue  # Cameras handled separately
        else:
            _log_field(field_value, f"{entity_path}/{field_name}")


def _log_field(value, path: str):
    if isinstance(value, np.ndarray):
        # Flatten and log each element
        flat = value.flatten()
        for i, val in enumerate(flat):
            rr.log(f"{path}/{i}", rr.Scalars(float(val)))

    elif isinstance(value, (int, float, np.number)):
        rr.log(path, rr.Scalars(float(value)))


_warned_depth_units: set[str] = set()


def _depth_image_kwargs(
    entity_path: str,
    depth_units: float | None,
    depth_range: tuple[float, float] | None = None,
    colormap: str = "viridis",
) -> dict:
    """Build ``rr.DepthImage`` keyword arguments for a raw depth map.

    Rerun expects both ``meter`` and ``depth_range`` in the native units of the depth
    array, so the metric colormap range is converted using the camera depth units. A
    camera that does not report its depth units gets neither: the depth map is logged
    unscaled and Rerun estimates the colormap range from the data.

    Args:
        entity_path: Entity path the depth map is logged to, used to warn only once per camera
        depth_units: Size of one depth unit in meters (e.g. 0.001 for millimeter depth)
        depth_range: Colormap range in meters, or None to let Rerun estimate it
        colormap: Rerun colormap used to display the depth map

    Returns:
        Keyword arguments to pass to ``rr.DepthImage``.
    """
    if not depth_units:
        if entity_path not in _warned_depth_units:
            _warned_depth_units.add(entity_path)
            logger.warning(f"No depth units reported for {entity_path}, logging depth unscaled.")
        return {"colormap": colormap}

    meter = 1.0 / depth_units
    kwargs: dict = {"colormap": colormap, "meter": meter}
    if depth_range is not None:
        kwargs["depth_range"] = (depth_range[0] * meter, depth_range[1] * meter)
    return kwargs


class RequestType(Enum):
    # Configuration requests
    SET_TIME_SEQUENCE = auto()
    SET_ROBOT_MODEL = auto()
    RESET_RECORDING = auto()
    SAVE_RECORDING = auto()
    CONFIGURE_ENTITY = auto()
    # Data logging requests
    LOG_IMAGE = auto()
    LOG_DEPTH = auto()
    LOG_ARRAY = auto()
    LOG_TRAJECTORY = auto()
    LOG_FRAME = auto()
    LOG_SCALAR = auto()
    LOG_POINTS3D = auto()
    # Aggregated logging requests
    LOG_ENV_STATE = auto()


RR_TYPE_MAP = {
    RequestType.LOG_IMAGE: (lambda: rr.Image, "image"),
    RequestType.LOG_DEPTH: (lambda: rr.DepthImage, "depth"),
    RequestType.LOG_ARRAY: (lambda: rr.Tensor, "array"),
    RequestType.LOG_SCALAR: (lambda: rr.Scalars, "value"),
    RequestType.LOG_POINTS3D: (lambda: rr.Points3D, "points"),
}


class RerunVisualizer(Node):
    __api__ = [
        "log_image",
        "log_depth",
        "log_array",
        "log_trajectory",
        "log_frame",
        "log_scalar",
        "log_points3d",
        "log_env_state",
        "set_time_sequence",
        "set_robot_model",
        "reset_recording",
        "save_recording",
        "configure_entity",
        "is_busy",
    ]
    __pub__ = True
    __req__ = True  # Changed to True - now using request queue

    def __init__(
        self,
        app_id: str = "recontrol_visualization",
        recording_id: str | None = None,
        spawn: bool = True,
        init_logging: bool = True,
        rgb_quality: int = 50,
        *,
        freq: int = 30,
        max_buffer_size: int = 30,
        max_queue_size: int = 1,
        max_request_bytes: int = 50_000_00,  #  100KB default for flexible requests
        **kwargs,
    ):
        """
        Args:
            app_id: Application ID for Rerun recording
            recording_id: Optional recording ID, generated automatically if None
            spawn: Whether to spawn Rerun viewer automatically
            init_logging: Whether to initialize Rerun logging on startup
            freq: Update frequency in Hz
            max_buffer_size: Maximum size of ring buffer for state history
            max_request_bytes: Maximum size for serialized request data
        """
        self.app_id = app_id
        self.recording_id = recording_id
        self.spawn = spawn
        self.init_logging = init_logging
        self.max_request_bytes = max_request_bytes
        self.robot_model = None
        self.mjcf_logger = None
        self.blueprint = None
        self.rgb_quality = rgb_quality
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, max_queue_size=max_queue_size, **kwargs)

        if not self.verbose:
            logger.disable("rio.visualization.rerun")

    def __post_init__(self):
        # Use flexible field for request data
        self.example_request = {
            "type": RequestType.LOG_IMAGE.value,
            "data": ("flexible", self.max_request_bytes),
        }
        self.example_data = {
            "timestamp": time.now(),
            "frame_count": 0,
        }
        self.worker = None
        self.run = self.pubreq

        # Build request handler dispatch table
        self._request_handlers = {}
        for req_type in RequestType:
            handler_name = f"_handle_{req_type.name.lower()}"
            handler = getattr(self, handler_name, None)
            if handler:
                self._request_handlers[req_type] = handler

        super().__post_init__()

    def pubreq(self):
        """Main loop that handles requests and updates visualization."""
        logger.debug("Starting RerunVisualizer...")
        try:
            if self.init_logging:
                self._init_rerun()

            rate = time.Rate(self.freq)
            not_pub_ready = True
            not_req_ready = True
            frame_count = 0

            while not self.exit_event.is_set():
                # Process requests from shared memory queue
                try:
                    reqs = self.request_queue.get_all()
                    if isinstance(reqs, dict):
                        # Convert dict format to list of request dicts
                        num_reqs = len(reqs["type"])
                        reqs = [{"type": RequestType(reqs["type"][i]), "data": reqs["data"][i]} for i in range(num_reqs)]
                except queue.Empty:
                    reqs = []

                # Handle each request
                for req_raw in reqs:
                    req = Request(RequestType(req_raw.pop("type")), req_raw["data"])
                    try:
                        self._handle_request(req)
                    except Exception as e:
                        # Keep visualizing: one malformed request should not take down the node
                        logger.opt(exception=True).error(f"Error handling {req.type.name} request: {e}")

                # Update time
                rr.set_time("frame_idx", sequence=frame_count)
                rr.set_time("time", timestamp=time.now())

                # Publish state
                data = {
                    "timestamp": time.now(),
                    "frame_count": frame_count,
                }
                self.ring_buffer.put(data)

                # Signal readiness
                if not_pub_ready:
                    self.pub_ready_event.set()
                    not_pub_ready = False
                if not_req_ready:
                    self.req_ready_event.set()
                    not_req_ready = False

                frame_count += 1
                rate.precise_sleep()

        except KeyboardInterrupt:
            pass

    def _init_rerun(self):
        """Initialize Rerun recording."""
        if rr is None:
            logger.warning("Rerun not installed. Visualization disabled.")
            return

        rr.init(self.app_id, recording_id=self.recording_id, spawn=self.spawn)
        logger.debug(f"Initialized Rerun with app_id='{self.app_id}'")
        if self.blueprint is not None:
            rr.set_blueprint(self.blueprint)
            logger.debug("Set custom Rerun blueprint.")

    def _handle_request(self, req: Request):
        """Dispatch request to appropriate handler."""
        if rr is None:
            return

        # Set timestamp if provided
        if req.params.get("timestamp") is not None:
            rr.set_time("time", timestamp=req.params["timestamp"])

        # Check if this is a simple rr.log request
        if req.type in RR_TYPE_MAP:
            rr_type_fn, data_param = RR_TYPE_MAP[req.type]
            entity_path = req.params.pop("entity_path")
            data = req.params.pop(data_param)
            self._log_rr(entity_path, data, rr_type_fn(), **req.params)
            return

        # Otherwise dispatch to custom handler
        handler = self._request_handlers.get(req.type)
        if handler:
            handler(**req.params)
        else:
            logger.warning(f"No handler for request type: {req.type}")

    def _log_rr(self, entity_path: str, data, rr_type, **kwargs):
        """Generic rerun logging function."""
        if data is None:
            return
        if rr_type == rr.Image:
            rr.log(entity_path, rr_type(data, color_model=kwargs.get("color_model")).compress(jpeg_quality=self.rgb_quality))
        else:
            rr.log(entity_path, rr_type(data, **kwargs))

    # Configuration request handlers
    def _handle_set_time_sequence(self, time_sequence: int, **kwargs):
        """Handle SET_TIME_SEQUENCE request."""
        rr.set_time("frame_idx", sequence=time_sequence)

    def _handle_set_robot_model(
        self, entity_path: str, model=None, robot_description: str | None = None, variant: str | None = None, **kwargs
    ):
        """Handle SET_ROBOT_MODEL request."""
        if model is None and robot_description is not None:
            model = self._load_robot_model(robot_description, variant)

        if model is not None:
            self.robot_model = model
            if rerun_loader_mjcf is not None:
                try:
                    self.mjcf_logger = rerun_loader_mjcf.MJCFLogger(model, entity_path_prefix=entity_path)
                    self.mjcf_logger.log_model()
                    logger.debug(f"Set and logged robot model at {entity_path}")
                except Exception as e:
                    logger.error(f"Error logging robot model: {e}")
            else:
                logger.warning("rerun_loader_mjcf not installed. Cannot log robot model.")

    def _handle_reset_recording(self, **kwargs):
        """Handle RESET_RECORDING request."""
        rr.reset()
        logger.debug("Reset Rerun recording")

    def _handle_save_recording(self, save_path: str = "/tmp/recording.rrd", **kwargs):
        """Handle SAVE_RECORDING request."""
        rr.save(save_path)
        logger.debug(f"Saved recording to {save_path}")

    def _handle_configure_entity(self, entity_path: str, config: dict, **kwargs):
        """Handle CONFIGURE_ENTITY request."""
        # Entity configuration logic can be added here
        pass

    def _handle_log_trajectory(
        self, entity_path: str, trajectory: np.ndarray, color: list | None = None, radii: float = 0.01, **kwargs
    ):
        """Handle LOG_TRAJECTORY request."""
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        display_arrows_traj(entity_path, trajectory, color, radii=radii)

    def _handle_log_frame(self, entity_path: str, pose: np.ndarray | None = None, axis_length: float = 0.1, **kwargs):
        display_frame(entity_path, pose, axis_length=axis_length)

    def _handle_log_env_state(self, entity_path: str, step: Step, **kwargs):
        """Handle LOG_ENV_STATE request."""
        obs = step.observation
        action = step.action
        robot_path = f"{entity_path}/robot"
        # Image
        for cam_name, camera in obs.cameras.items():
            cam_path = f"{entity_path}/camera/{cam_name}"
            if camera.rgb is not None:
                rr.log(
                    f"{cam_path}/rgb",
                    rr.Image(camera.rgb, color_model=rr.ColorModel.RGB).compress(jpeg_quality=self.rgb_quality),
                )
            if camera.depth is not None:
                depth_path = f"{cam_path}/depth"
                depth_units = (camera.meta or {}).get("depth_units")
                rr.log(depth_path, rr.DepthImage(camera.depth, **_depth_image_kwargs(depth_path, depth_units)))
        # Observation
        log_obs(obs, entity_path=robot_path)

        # Action
        if action is not None:
            _log_field(action, f"{robot_path}/action")
        # Instruction
        if step.instruction is not None:
            rr.log(f"{robot_path}/instruction", rr.TextDocument(step.instruction))

    def _update_robot_visualization(self, entity_path: str, joint_q: np.ndarray):
        """Update robot visualization using MJCF model."""
        if self.robot_model is None or self.mjcf_logger is None:
            return

        data = mujoco.MjData(self.robot_model)
        data.qpos[: len(joint_q)] = joint_q
        mujoco.mj_forward(self.robot_model, data)
        self.mjcf_logger.log_data(data)

    def _load_robot_model(self, description: str, variant: str | None = None):
        """Load a robot model from robot_descriptions."""
        if not ROBOT_DESCRIPTIONS_AVAILABLE:
            logger.warning("robot_descriptions not installed. Cannot load robot model.")
            return None
        try:
            if variant is not None:
                model = load_robot_description(description, variant=variant)
            else:
                model = load_robot_description(description)
            return model
        except ImportError:
            logger.warning(f"{description} NOT FOUND! Check https://github.com/robot-descriptions/robot_descriptions.py")
            return None
        except Exception as e:
            logger.error(f"Error loading robot model: {e}")
            return None

    # Public API methods - now use shared memory request queue
    def log_image(self, entity_path: str, image: np.ndarray, color_model=rr.ColorModel.RGB):
        """
        Log an image.

        Args:
            entity_path: Entity path for the image
            image: Image array (H, W, C) for RGB or (H, W) for grayscale
        """
        req = {
            "type": RequestType.LOG_IMAGE.value,
            "data": {"entity_path": entity_path, "image": image, "color_model": color_model},
        }
        self.request_queue.put(req)

    def log_depth(
        self,
        entity_path: str,
        depth: np.ndarray,
        depth_units: float | None,
        depth_range: tuple[float, float] | None = None,
        colormap: str = "viridis",
    ):
        """
        Log a depth image.

        Args:
            entity_path: Entity path for the depth image
            depth: Depth array (H, W) in raw sensor units
            depth_units: Size of one depth unit in meters (e.g. 0.001 for millimeter depth).
                Cameras that do not report it log the depth unscaled and warn
            depth_range: Colormap range in meters, or None to let Rerun estimate it
            colormap: Rerun colormap used to display the depth map
        """
        req = {
            "type": RequestType.LOG_DEPTH.value,
            "data": {
                "entity_path": entity_path,
                "depth": depth,
                **_depth_image_kwargs(entity_path, depth_units, depth_range, colormap),
            },
        }
        self.request_queue.put(req)

    def log_array(self, entity_path: str, array: np.ndarray):
        """
        Log a generic array/tensor.

        Args:
            entity_path: Entity path for the array
            array: Numpy array of any shape
        """
        req = {
            "type": RequestType.LOG_ARRAY.value,
            "data": {
                "entity_path": entity_path,
                "array": array,
            },
        }
        self.request_queue.put(req)

    def log_trajectory(self, entity_path: str, trajectory: np.ndarray, color: list | None = None, radii: float = 0.01):
        """
        Log a trajectory.

        Args:
            entity_path: Entity path for the trajectory
            trajectory: Array of shape (N, 3) representing 3D points
            color: RGB color as [r, g, b] (0-255)
            radii: Arrow radius
        """
        if color is None:
            color = [255, 0, 0]
        req = {
            "type": RequestType.LOG_TRAJECTORY.value,
            "data": {
                "entity_path": entity_path,
                "trajectory": trajectory,
                "color": color,
                "radii": radii,
            },
        }
        self.request_queue.put(req)

    def log_frame(self, entity_path: str, pose: np.ndarray | None = None, axis_length: float = 0.1):
        """
        Log a coordinate frame.

        Args:
            entity_path: Entity path for the frame
            pose: Pose array [x, y, z, rx, ry, rz] or [x, y, z, qx, qy, qz, qw]
            axis_length: Length of coordinate axes
        """
        req = {
            "type": RequestType.LOG_FRAME.value,
            "data": {
                "entity_path": entity_path,
                "pose": pose,
                "axis_length": axis_length,
            },
        }
        self.request_queue.put(req)

    def log_scalar(self, entity_path: str, value: float | np.ndarray):
        """
        Log a scalar value.

        Args:
            entity_path: Entity path for the scalar
            value: Scalar value to log
        """
        req = {
            "type": RequestType.LOG_SCALAR.value,
            "data": {
                "entity_path": entity_path,
                "value": value,
            },
        }
        self.request_queue.put(req)

    def log_points3d(
        self, entity_path: str, points: np.ndarray, colors: np.ndarray | None = None, radii: float | np.ndarray | None = None
    ):
        """
        Log 3D points.

        Args:
            entity_path: Entity path for the points
            points: Array of shape (N, 3)
            colors: Optional colors array of shape (N, 3) or (N, 4)
            radii: Optional radius/radii for points
        """
        req = {
            "type": RequestType.LOG_POINTS3D.value,
            "data": {
                "entity_path": entity_path,
                "points": points,
                "colors": colors,
                "radii": radii,
            },
        }
        self.request_queue.put(req)

    def log_env_state(self, entity_path: str, step: Step):
        """
        Log environment step data.

        Args:
            entity_path: Base entity path
            step: Step object containing observation, action, and metadata
        """
        req = {
            "type": RequestType.LOG_ENV_STATE.value,
            "data": {
                "entity_path": entity_path,
                "step": step,
            },
        }
        self.request_queue.put(req)

    def set_time_sequence(self, time_sequence: int):
        """Set the time sequence for logging."""
        req = {
            "type": RequestType.SET_TIME_SEQUENCE.value,
            "data": {
                "time_sequence": time_sequence,
            },
        }
        self.request_queue.put(req)

    def set_robot_model(
        self,
        entity_path: str,
        model=None,
        robot_description: str | None = None,
        variant: str | None = None,
    ):
        """
        Args:
            entity_path: Entity path for the robot
            model: Optional MuJoCo model instance (mujoco.MjModel)
            robot_description: Optional robot description name (e.g., "panda_mj_description")
            variant: Optional variant name (e.g., "panda_nohand")
        """
        req = {
            "type": RequestType.SET_ROBOT_MODEL.value,
            "data": {
                "entity_path": entity_path,
                "model": model,
                "robot_description": robot_description,
                "variant": variant,
            },
        }
        self.request_queue.put(req)

    def reset_recording(self):
        """Reset the Rerun recording."""
        req = {
            "type": RequestType.RESET_RECORDING.value,
            "data": {},
        }
        self.request_queue.put(req)

    def save_recording(self, save_path: str = "/tmp/recording.rrd"):
        """
        Save the current recording.

        Args:
            save_path: Path where the recording should be saved
        """
        req = {
            "type": RequestType.SAVE_RECORDING.value,
            "data": {
                "save_path": save_path,
            },
        }
        self.request_queue.put(req)

    def configure_entity(self, entity_path: str, config: dict):
        """
        Configure visualization settings for an entity.

        Args:
            entity_path: Rerun entity path
            config: Configuration dictionary
        """
        req = {
            "type": RequestType.CONFIGURE_ENTITY.value,
            "data": {
                "entity_path": entity_path,
                "config": config,
            },
        }
        self.request_queue.put(req)

    def is_busy(self) -> bool:
        """Check if there are pending requests to process."""
        return not self.request_queue.empty()


def RerunServer(mw, *args, **kwargs):
    return ServerFactory(mw, RerunVisualizer, *args, **kwargs)


def RerunClient(mw, *args, **kwargs):
    return ClientFactory(mw, RerunVisualizer, *args, **kwargs)

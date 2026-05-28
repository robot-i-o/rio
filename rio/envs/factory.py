# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass
from dataclasses import fields as dataclass_fields
from importlib import import_module
from inspect import Parameter, signature

from attrs import fields as attrs_fields
from attrs import has as attrs_has

from ..cfg.node import NodeCfg
from ..embodiments.base import EmbodimentType


def dataclass_to_dict(dc):
    """Convert a dataclass to a dictionary, handling nested dataclasses."""
    if not hasattr(dc, "__dataclass_fields__"):
        return dc
    result = {}
    for field in dc.__dataclass_fields__:
        value = getattr(dc, field)
        result[field] = value
    return result


def make_policy(policy_name, policy_kwargs):
    module = import_module(f"rio.policies.{policy_name.lower()}")
    PolicyClass = getattr(module, policy_name)
    if PolicyClass is None:
        raise ImportError(policy_name)
    return PolicyClass(**policy_kwargs)


def make_node(mw, module, node, node_kwargs, package="rio_hw"):
    if node is None:
        node_server = None
        node_client = None
    else:
        module = import_module(f"{package}.{module}")
        NodeServer = getattr(module, f"{node}Server", None)
        NodeClient = getattr(module, f"{node}Client", None)
        if NodeServer is None or NodeClient is None:
            raise ImportError(node)
        node_server = lambda: NodeServer(mw, **node_kwargs)
        node_client = lambda: NodeClient(mw, **node_kwargs)
    return node_server, node_client


@contextmanager
def init_clients(clients):
    with ExitStack() as stack:
        active_clients = {}
        for name, client_factory in clients.items():
            if client_factory:
                active_clients[name] = stack.enter_context(client_factory())
            else:
                active_clients[name] = None
        yield active_clients


def make_cameras(mw, cameras, **kwargs):
    servers = {}
    clients = {}
    try:
        # Camera
        for cam_name, cam_spec in cameras.items():
            cam_module = kwargs.get(cam_spec.module, "cameras")
            cfg = cam_spec.cfg
            servers[cam_name], clients[cam_name] = make_node(mw, cam_module, cam_spec.cam_type, cfg)
    except AttributeError:
        pass
    return servers, clients


def instantiate_station_cfg(args, **kwargs) -> tuple[dict, dict, dict]:
    servers = {}
    clients = {}

    # Policies
    try:
        policy_module = kwargs.get("policy_module", "policies")
        policy_node_cfg = kwargs.get("policy_node_cfg", asdict(args.policy_node_cfg))
        policy_config = kwargs.get("policy_config", dataclass_to_dict(args.policy_cfg))
        policy = make_policy(args.policy, policy_config)
        policy_node_cfg["policy"] = policy
        servers["policy"], clients["policy"] = make_node(
            args.mw, policy_module, "PolicyInterface", policy_node_cfg, package="rio"
        )
    except AttributeError:
        servers["policy"], clients["policy"] = None, None

    # Get fields from either attrs or dataclass
    if attrs_has(args):
        config_fields = {f.name: getattr(args, f.name) for f in attrs_fields(args)}
    elif is_dataclass(args):
        config_fields = {f.name: getattr(args, f.name) for f in dataclass_fields(args)}
    else:
        raise TypeError(f"Config must be either an attrs class or a dataclass, got {type(args)}")

    # Camera nodes
    cam_servers, cam_clients = make_cameras(args.mw, config_fields.get("cameras", {}), **kwargs)
    servers.update(cam_servers)
    clients.update(cam_clients)
    camera_clients = dict(sorted(cam_clients.items()))

    # Instantiate nodes based on station config
    for field_name, field_value in config_fields.items():
        # Skip config fields - we'll access them via their paired node field
        if field_name.endswith("_cfg"):
            continue

        if field_value is None:
            servers[field_name], clients[field_name] = None, None
            continue

        module_override_key = f"{field_name}_module"

        if isinstance(field_value, str):
            # Standard pattern: field: str = "NodeClass" + field_cfg: <cfg> pairs.
            if field_name in ("cameras", "policy"):
                continue  # handled elsewhere

            cfg_field_name = f"{field_name}_cfg"
            cfg = config_fields.get(cfg_field_name)
            if cfg is None:
                continue

            node_class = field_value
            # NodeCfg instances (Arm, Gripper, Hand) store kwargs in .cfg;
            # plain dataclasses expose them via __dict__.
            if isinstance(cfg, NodeCfg):
                cfg_dict = cfg.cfg
            elif hasattr(cfg, "__dict__"):
                cfg_dict = cfg.__dict__
            else:
                cfg_dict = cfg
            package = "rio_hw"
            if hasattr(args, module_override_key):
                module = getattr(args, module_override_key)
                package = "rio"
            elif module_override_key in kwargs:
                module = kwargs[module_override_key]
                package = "rio"
            elif field_name.startswith("teleop"):
                module = "interfaces"
            elif field_name == "visualizer":
                module = "visualization"
                package = "rio"
            elif field_name == "recorder":
                module = "data"
                package = "rio"
            else:
                module = "robots"

        else:
            continue

        # Create server and client
        servers[field_name], clients[field_name] = make_node(
            args.mw,
            module,
            node_class,
            cfg_dict,
            package=package,
        )

    return servers, clients, camera_clients


def get_mbody_components(embodiment_type, clients):
    # TODO: improve matching logic
    def find_client(param_name):
        if param_name in clients:
            return clients[param_name]

        # Try variations: arm_1 -> arm1, arm
        if "_" in param_name:
            base, suffix = param_name.rsplit("_", 1)
            if f"{base}{suffix}" in clients:
                return clients[f"{base}{suffix}"]
            if suffix == "1" and base in clients:
                return clients[base]

        return None

    _skip_params = {"self", "action_space", "kwargs", "args"}

    if isinstance(embodiment_type, str):
        embodiment_type = EmbodimentType[embodiment_type.upper()]

    module = import_module(f"rio.embodiments.{embodiment_type.name.lower()}")
    EmbodimentClass = getattr(module, embodiment_type.name.title().replace("_", ""))

    sig = signature(EmbodimentClass.__init__)
    param_specs = {
        name: param
        for name, param in sig.parameters.items()
        if name not in _skip_params and not name.startswith("**") and not name.startswith("*")
    }

    components = {}
    missing = []
    for param_name, param in param_specs.items():
        client = find_client(param_name)
        is_required = param.default == Parameter.empty

        if client is not None:
            components[param_name] = client
        elif is_required:
            missing.append(param_name)

    if missing:
        raise RuntimeError(
            f"Missing required components for {embodiment_type.name}: {missing}. Available: {list(clients.keys())}"
        )

    return components, EmbodimentClass

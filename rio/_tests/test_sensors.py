# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pytest

from rio._scripts import list_available
from rio.cfg import NodeCfg
from rio.data.loader import dict_to_step
from rio.data.recorder import step_to_dict
from rio.envs import factory
from rio.envs.env import Env
from rio.schema import Observation, Step

pytestmark = pytest.mark.unit


@dataclass
class SensorStation:
    mw: str = "Thread"
    arm: str | None = None
    ft_sensor: str | None = "AtiFt"
    ft_sensor_cfg: NodeCfg = field(default_factory=lambda: NodeCfg(host="192.168.1.1", freq=100))


class StubArm:
    num_joints = 7

    def get_state(self):
        return {
            "eef_pose": np.zeros(6, dtype=np.float32),
            "joint_q": np.zeros(self.num_joints, dtype=np.float32),
            "gripper_position": 0.0,
        }


class StubSensor:
    def get_state(self):
        return {
            "force": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "torque": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }


def _client_factory(client):
    @contextmanager
    def _client():
        yield client

    return _client


def test_sensor_station_fields_resolve_to_rio_hw_sensors(monkeypatch):
    calls = []

    def fake_make_node(mw, module, node, node_kwargs, package="rio_hw"):
        calls.append(
            {
                "mw": mw,
                "module": module,
                "node": node,
                "node_kwargs": node_kwargs,
                "package": package,
            }
        )
        return lambda: None, lambda: None

    monkeypatch.setattr(factory, "make_node", fake_make_node)

    servers, clients, camera_clients = factory.instantiate_station_cfg(SensorStation())

    assert servers["ft_sensor"] is not None
    assert clients["ft_sensor"] is not None
    assert camera_clients == {}
    assert calls == [
        {
            "mw": "Thread",
            "module": "sensors",
            "node": "AtiFt",
            "node_kwargs": {"host": "192.168.1.1", "freq": 100},
            "package": "rio_hw",
        }
    ]


def test_env_adds_sensor_state_to_observation():
    env = Env(
        mw="Thread",
        clients={
            "arm": _client_factory(StubArm()),
            "ft_sensor": _client_factory(StubSensor()),
        },
        embodiment_type="SINGLE_ARM",
        action_space="TASK_POS",
    )

    with env:
        step = env.get_state(use_relative_time=False)

    assert set(step.observation.sensors) == {"ft_sensor"}
    np.testing.assert_array_equal(step.observation.sensors["ft_sensor"]["force"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(step.observation.sensors["ft_sensor"]["torque"], np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_sensor_observation_roundtrip_through_flat_step_dict():
    step = Step(
        timestep=1,
        observation=Observation(
            proprio=np.zeros(7, dtype=np.float32),
            sensors={
                "ft_sensor": {
                    "force": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    "torque": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                }
            },
        ),
        instruction="test",
        action=np.zeros(7, dtype=np.float32),
    )

    loaded = dict_to_step(step_to_dict(step))

    assert set(loaded.observation.sensors) == {"ft_sensor"}
    np.testing.assert_array_equal(
        loaded.observation.sensors["ft_sensor"]["force"],
        step.observation.sensors["ft_sensor"]["force"],
    )
    np.testing.assert_array_equal(
        loaded.observation.sensors["ft_sensor"]["torque"],
        step.observation.sensors["ft_sensor"]["torque"],
    )


def test_list_sensors_uses_rio_hw_sensors(monkeypatch):
    calls = []
    monkeypatch.setattr(
        list_available,
        "_list",
        lambda module_path, label, env_var=None: calls.append((module_path, label, env_var)),
    )

    list_available.list_sensors()

    assert calls == [("rio_hw.sensors", "sensors", None)]

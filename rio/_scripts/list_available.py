# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import importlib


def _list(module_path: str, label: str, env_var: str | None = None) -> None:
    mod = importlib.import_module(module_path)
    names = mod.__all__
    hint = f"  (set with {env_var}=<name>)" if env_var else ""
    print(f"Available {label}{hint}:\n")
    for name in names:
        print(f"  {name}")


def list_stations() -> None:
    _list("examples.cfg", "stations", env_var="STATION")


def list_robots() -> None:
    _list("rio_hw.robots", "robots")


def list_cameras() -> None:
    _list("rio_hw.cameras", "cameras")


def list_sensors() -> None:
    _list("rio_hw.sensors", "sensors")


def list_interfaces() -> None:
    _list("rio_hw.interfaces", "interfaces")

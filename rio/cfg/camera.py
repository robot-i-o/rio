# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from .node import NodeCfg


class Camera(NodeCfg):
    """Camera node configuration.

    `cam_type` selects the backend class in `rio_hw.cameras` (e.g. "Realsense" resolves to
    `RealsenseServer`/`RealsenseClient`). All remaining keyword arguments are stored in `cfg`
    and forwarded to that backend.
    """

    def __init__(self, **kwargs):
        if "cam_type" not in kwargs:
            raise TypeError("Camera() missing required keyword argument: 'cam_type'")
        self.cam_type = kwargs.pop("cam_type")
        self.module = kwargs.pop("module", "cameras")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in {"cam_type": self.cam_type, **self.cfg}.items())
        return f"{type(self).__name__}({params})"

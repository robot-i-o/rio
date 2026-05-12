# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0


class Camera:
    def __init__(self, cam_type: str, module: str = "cameras", **kwargs):
        self.cam_type = cam_type
        self.module = module
        self.cfg = kwargs

# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0


class NodeCfg:
    """Flexible hardware node configuration container.

    Stores keyword arguments for a hardware node and provides attribute access
    forwarding.
    """

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def __getattr__(self, name: str):
        try:
            cfg = object.__getattribute__(self, "cfg")
        except AttributeError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

        if name in cfg:
            return cfg[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.cfg.items())
        return f"{type(self).__name__}({params})"

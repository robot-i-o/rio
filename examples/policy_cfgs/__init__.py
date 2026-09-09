__all__ = ["MolmoAct2BimanualYamCfg", "MolmoAct2Cfg", "Pi05Cfg", "SmolVLACfg"]


def __getattr__(name):
    if name == "Pi05Cfg":
        from .pi05 import Pi05Cfg

        return Pi05Cfg

    if name == "SmolVLACfg":
        from .smolvla import SmolVLACfg

        return SmolVLACfg

    if name == "MolmoAct2Cfg":
        from .molmoact2 import MolmoAct2Cfg

        return MolmoAct2Cfg

    if name == "MolmoAct2BimanualYamCfg":
        from .molmoact2_bimanual_yam import MolmoAct2BimanualYamCfg

        return MolmoAct2BimanualYamCfg

    raise AttributeError(f"module 'examples.policy_cfgs' has no attribute {name!r}")

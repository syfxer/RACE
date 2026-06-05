"""Reward decomposition and RACE reward modules."""

from .race_reward import RARRReward, RARRRewardConfig
from .race_vib import RARRVIBRewardDecomposer
from .race_residual import (
    RARRDiasterRewardDecomposer,
    RARRRDRewardDecomposer,
    RARRRRDRewardDecomposer,
)
from .stepsoc_reward import StepSOCReward, StepSOCRewardConfig, StepSOCVIBRewardDecomposer


def import_reward_model(args):
    """Factory used by RL algorithms to instantiate the selected decomposer."""
    reward_model = None
    if args.rd_method in {"RRD", "RRD_unbiased"}:
        from .rrd import RRDRewardDecomposer

        reward_model = RRDRewardDecomposer(args)
    elif args.rd_method == "VIB":
        from .VIB import IBRewardDecomposer

        reward_model = IBRewardDecomposer(args)
    elif args.rd_method == "RD":
        from .RD import RDRewardDecomposer

        reward_model = RDRewardDecomposer(args)
    elif args.rd_method == "Diaster":
        from .Diaster import DiasterRewardDecomposer

        reward_model = DiasterRewardDecomposer(args)
    elif args.rd_method == "RARR_VIB":
        reward_model = RARRVIBRewardDecomposer(args)
    elif args.rd_method == "RARR_RD":
        reward_model = RARRRDRewardDecomposer(args)
    elif args.rd_method == "RARR_RRD":
        reward_model = RARRRRDRewardDecomposer(args)
    elif args.rd_method == "RARR_Diaster":
        reward_model = RARRDiasterRewardDecomposer(args)
    elif args.rd_method == "VIB_StepSOC":
        reward_model = StepSOCVIBRewardDecomposer(args)
    return reward_model


__all__ = [
    "RARRReward",
    "RARRRewardConfig",
    "RARRVIBRewardDecomposer",
    "StepSOCReward",
    "StepSOCRewardConfig",
    "StepSOCVIBRewardDecomposer",
    "RARRRDRewardDecomposer",
    "RARRRRDRewardDecomposer",
    "RARRDiasterRewardDecomposer",
    "import_reward_model",
]

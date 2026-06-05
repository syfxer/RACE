from dataclasses import dataclass

from race.rewards.reward_deadband import deadband_penalty_value, deadband_violation

from .race_vib import RARRVIBRewardDecomposer


@dataclass
class StepSOCRewardConfig:
    lambda_stepsoc: float
    soc_target: float
    soc_deadband: float
    voltage_weight: float = 1.0
    stepsoc_mode: str = "linear"


class StepSOCReward:
    """Step-wise SOC dense shaping baseline."""

    def __init__(self, config: StepSOCRewardConfig):
        self.config = config

    def compute(self, soc_next, voltage_cost_t):
        cfg = self.config
        raw_error, violation = deadband_violation(soc_next, cfg.soc_target, cfg.soc_deadband)
        penalty = deadband_penalty_value(violation, cfg.lambda_stepsoc, cfg.stepsoc_mode)
        r_voltage = -float(cfg.voltage_weight) * float(voltage_cost_t)
        r_stepsoc = -float(penalty)
        return {
            "reward": float(r_voltage + r_stepsoc),
            "r_voltage": float(r_voltage),
            "r_stepsoc": float(r_stepsoc),
            "step_soc_error": float(raw_error),
            "step_soc_deadband_violation": float(violation),
        }


class StepSOCVIBRewardDecomposer(RARRVIBRewardDecomposer):
    """VIB terminal-SOC residual used with the StepSOC base reward."""

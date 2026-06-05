from dataclasses import dataclass
from typing import Optional

from race.rewards.reward_deadband import deadband_penalty_value


@dataclass
class RARRRewardConfig:
    T: int
    gamma: float
    lambda_soc: float
    soc_target: float
    soc_deadband: float
    soc_min: float
    soc_max: float
    delta_soc_ch_max: float
    delta_soc_dis_max: float
    voltage_weight: float = 1.0
    use_gamma: bool = True
    penalty_power: float = 1.0
    lambda_rarr: Optional[float] = None
    rarr_mode: str = "linear"


class RARRReward:
    """Returnability-aware reward redistribution for terminal SOC credit."""

    def __init__(self, config: RARRRewardConfig):
        self.config = config

    def returnability_bounds(self, t):
        cfg = self.config
        t = max(0, min(int(t), int(cfg.T)))
        remain = cfg.T - t
        target_low = cfg.soc_target - max(0.0, float(cfg.soc_deadband))
        target_high = cfg.soc_target + max(0.0, float(cfg.soc_deadband))
        lower = target_low - remain * cfg.delta_soc_ch_max
        upper = target_high + remain * cfg.delta_soc_dis_max
        return max(cfg.soc_min, lower), min(cfg.soc_max, upper)

    @staticmethod
    def distance_to_interval(soc, lower, upper):
        soc = float(soc)
        if soc < lower:
            return lower - soc
        if soc > upper:
            return soc - upper
        return 0.0

    def potential(self, t, soc):
        cfg = self.config
        lower, upper = self.returnability_bounds(t)
        distance = self.distance_to_interval(soc, lower, upper)
        return -float(cfg.lambda_soc) * (distance ** float(cfg.penalty_power))

    def rarr_penalty(self, distance):
        cfg = self.config
        lambda_rarr = float(cfg.lambda_soc if cfg.lambda_rarr is None else cfg.lambda_rarr)
        if str(cfg.rarr_mode).lower() in {"linear", "squared"}:
            return -deadband_penalty_value(distance, lambda_rarr, cfg.rarr_mode)
        return -lambda_rarr * (float(distance) ** float(cfg.penalty_power))

    def compute(self, t, soc_t, soc_next, voltage_cost_t):
        cfg = self.config
        phi_t = self.potential(t, soc_t)
        phi_next = self.potential(t + 1, soc_next)
        r_voltage = -float(cfg.voltage_weight) * float(voltage_cost_t)

        lower_t, upper_t = self.returnability_bounds(t)
        lower_next, upper_next = self.returnability_bounds(t + 1)
        d_t = self.distance_to_interval(soc_t, lower_t, upper_t)
        d_next = self.distance_to_interval(soc_next, lower_next, upper_next)
        r_rarr = self.rarr_penalty(d_next)
        reward = r_voltage + r_rarr

        return {
            "reward": float(reward),
            "r_voltage": float(r_voltage),
            "r_rarr": float(r_rarr),
            "phi_t": float(phi_t),
            "phi_next": float(phi_next),
            "lower_t": float(lower_t),
            "upper_t": float(upper_t),
            "lower_next": float(lower_next),
            "upper_next": float(upper_next),
            "d_t": float(d_t),
            "d_next": float(d_next),
            "recoverability_violation": float(d_next),
        }

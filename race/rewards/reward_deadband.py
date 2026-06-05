from __future__ import annotations


def deadband_violation(value: float, target: float, deadband: float) -> tuple[float, float]:
    raw_error = abs(float(value) - float(target))
    violation = max(0.0, raw_error - max(0.0, float(deadband)))
    return raw_error, violation


def deadband_penalty_value(violation: float, weight: float, mode: str = "linear") -> float:
    mode = str(mode).lower()
    violation = max(0.0, float(violation))
    if mode == "linear":
        return float(weight) * violation
    if mode == "squared":
        return float(weight) * (violation ** 2)
    raise ValueError(f"Unknown deadband penalty mode: {mode}")


def compute_terminal_soc_penalty(
    final_soc: float,
    soc_target: float,
    soc_deadband: float,
    lambda_soc: float,
    mode: str = "linear",
) -> dict[str, float]:
    raw_error, violation = deadband_violation(final_soc, soc_target, soc_deadband)
    penalty_cost = deadband_penalty_value(violation, lambda_soc, mode)
    return {
        "terminal_soc_penalty": -float(penalty_cost),
        "terminal_soc_penalty_cost": float(penalty_cost),
        "raw_terminal_soc_error": float(raw_error),
        "deadband_soc_violation": float(violation),
    }

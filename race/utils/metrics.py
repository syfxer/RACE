from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


CORE_EPISODE_METRIC_KEYS = [
    "train_reward",
    "original_reward",
    "avg_voltage_deviation",
    "voltage_deviation_max_pu",
    "terminal_soc_error",
    "voltage_violation_rate",
    "success",
    "soc_success",
    "voltage_success",
]


CORE_AGGREGATED_METRIC_KEYS = [
    "train_reward",
    "original_reward",
    "avg_voltage_deviation",
    "voltage_deviation_max_pu",
    "terminal_soc_error",
    "voltage_violation_rate",
    "success_rate",
    "soc_success_rate",
    "voltage_success_rate",
]


def compute_avg_voltage_deviation(voltage_matrix) -> float:
    voltage_matrix = np.asarray(voltage_matrix, dtype=float)
    if voltage_matrix.size == 0:
        return 0.0
    return float(np.mean(np.abs(voltage_matrix - 1.0)))


def compute_terminal_soc_error(final_soc: float, soc_target: float) -> float:
    return float(abs(float(final_soc) - float(soc_target)))


def compute_voltage_violation_rate(voltage_matrix, v_min: float = 0.95, v_max: float = 1.05) -> float:
    voltage_matrix = np.asarray(voltage_matrix, dtype=float)
    if voltage_matrix.size == 0:
        return 0.0
    violation = (voltage_matrix < float(v_min)) | (voltage_matrix > float(v_max))
    return float(np.mean(violation.astype(float)))


def compute_episode_success(
    soc_error: float,
    avgvd: float,
    vvm: float,
    no_control_avgvd: float,
    no_control_vvm: float,
    epsilon_soc_success: float = 0.03,
    avgvd_tolerance_ratio: float = 0.15,
    vvm_tolerance_ratio: float = 0.15,
) -> Dict[str, int]:
    soc_success = float(soc_error) <= float(epsilon_soc_success)
    voltage_success = (
        float(avgvd) <= (1.0 + float(avgvd_tolerance_ratio)) * max(float(no_control_avgvd), 1e-12)
        and float(vvm) <= (1.0 + float(vvm_tolerance_ratio)) * max(float(no_control_vvm), 1e-12)
    )
    success = bool(soc_success and voltage_success)
    return {
        "success": int(success),
        "soc_success": int(soc_success),
        "voltage_success": int(voltage_success),
    }


def compute_success_rate(success_list: Sequence[float]) -> float:
    if len(success_list) == 0:
        return 0.0
    return float(np.mean(np.asarray(success_list, dtype=float)))


@dataclass
class EpisodeMetricAccumulator:
    """Compact metric accumulator for three-stage sparse 33-bus training.

    It uses the per-step values already produced by ``new_env_ES.BatteryManagementEnv``.
    No legacy-only metrics such as grid loss or equivalent cycles are exposed here.
    """

    soc_target: float
    num_buses: int = 33
    epsilon_soc_success: float = 0.03
    avgvd_tolerance_ratio: float = 0.15
    vvm_tolerance_ratio: float = 0.15
    no_control_avgvd: float = 1.0
    no_control_vvm: float = 1.0
    train_rewards: List[float] = field(default_factory=list)
    original_rewards: List[float] = field(default_factory=list)
    voltage_deviation_mean_pu: List[float] = field(default_factory=list)
    voltage_deviation_max_pu: List[float] = field(default_factory=list)
    voltage_violation_bus_count: List[float] = field(default_factory=list)
    soc: List[float] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)

    def reset(self, initial_soc: float) -> None:
        self.train_rewards.clear()
        self.original_rewards.clear()
        self.voltage_deviation_mean_pu.clear()
        self.voltage_deviation_max_pu.clear()
        self.voltage_violation_bus_count.clear()
        self.soc.clear()
        self.actions.clear()
        self.soc.append(float(initial_soc))

    def update(
        self,
        train_reward: float,
        original_reward: float,
        info: Mapping[str, float],
        action: float,
    ) -> None:
        self.train_rewards.append(float(train_reward))
        self.original_rewards.append(float(original_reward))
        self.voltage_deviation_mean_pu.append(float(info.get("voltage_deviation_mean_pu", 0.0)))
        self.voltage_deviation_max_pu.append(float(info.get("voltage_deviation_max_pu", 0.0)))
        self.voltage_violation_bus_count.append(float(info.get("voltage_violation_bus_count", 0.0)))
        self.soc.append(float(info.get("soc", self.soc[-1] if self.soc else 0.0)))
        self.actions.append(float(action))

    def compute(self) -> Dict[str, float]:
        steps = max(len(self.train_rewards), 1)
        avg_voltage_deviation = (
            float(np.mean(self.voltage_deviation_mean_pu)) if self.voltage_deviation_mean_pu else 0.0
        )
        voltage_deviation_max_pu = (
            float(np.max(self.voltage_deviation_max_pu)) if self.voltage_deviation_max_pu else 0.0
        )
        voltage_violation_rate = (
            float(np.sum(self.voltage_violation_bus_count) / (steps * max(int(self.num_buses), 1)))
            if self.voltage_violation_bus_count
            else 0.0
        )
        final_soc = float(self.soc[-1]) if self.soc else 0.0
        terminal_soc_error = compute_terminal_soc_error(final_soc, self.soc_target)
        success_info = compute_episode_success(
            soc_error=terminal_soc_error,
            avgvd=avg_voltage_deviation,
            vvm=voltage_deviation_max_pu,
            no_control_avgvd=self.no_control_avgvd,
            no_control_vvm=self.no_control_vvm,
            epsilon_soc_success=self.epsilon_soc_success,
            avgvd_tolerance_ratio=self.avgvd_tolerance_ratio,
            vvm_tolerance_ratio=self.vvm_tolerance_ratio,
        )
        return {
            "train_reward": float(np.sum(self.train_rewards)),
            "original_reward": float(np.sum(self.original_rewards)),
            "avg_voltage_deviation": avg_voltage_deviation,
            "voltage_deviation_max_pu": voltage_deviation_max_pu,
            "terminal_soc_error": terminal_soc_error,
            "voltage_violation_rate": voltage_violation_rate,
            "success": float(success_info["success"]),
            "soc_success": float(success_info["soc_success"]),
            "voltage_success": float(success_info["voltage_success"]),
        }


def aggregate_metrics(metric_list: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {key: 0.0 for key in CORE_AGGREGATED_METRIC_KEYS}

    aggregated: Dict[str, float] = {}
    source_keys = [key for key in CORE_EPISODE_METRIC_KEYS if key in metric_list[0]]
    for key in source_keys:
        values = np.asarray([float(metrics[key]) for metrics in metric_list], dtype=float)
        rate_keys = {
            "success": "success_rate",
            "soc_success": "soc_success_rate",
            "voltage_success": "voltage_success_rate",
        }
        output_key = rate_keys.get(key, key)
        aggregated[output_key] = float(np.mean(values))
        aggregated[f"{output_key}_std"] = float(np.std(values))
    return aggregated


def build_wandb_eval_metrics(
    difficulty_metrics: Mapping[str, Mapping[str, float]],
    stage_id: Optional[int] = None,
) -> Dict[str, float]:
    log_dict: Dict[str, float] = {}
    difficulties = [difficulty for difficulty in ("easy", "medium", "hard") if difficulty in difficulty_metrics]

    for difficulty in difficulties:
        metrics = difficulty_metrics[difficulty]
        for key in CORE_AGGREGATED_METRIC_KEYS:
            if key in metrics:
                log_dict[f"eval/{difficulty}/{key}"] = float(metrics[key])

    if difficulties:
        for key in CORE_AGGREGATED_METRIC_KEYS:
            values = [float(difficulty_metrics[difficulty][key]) for difficulty in difficulties if key in difficulty_metrics[difficulty]]
            if values:
                log_dict[f"eval/all/{key}"] = float(np.mean(values))

    if stage_id is not None:
        log_dict["train/stage_id"] = float(stage_id)
    return log_dict


def compute_episodes_to_convergence(
    eval_log: Sequence[Mapping[str, float]],
    metric_reward: str = "eval/all/original_reward",
    metric_avgvd: str = "eval/all/avg_voltage_deviation",
    metric_soc: str = "eval/all/terminal_soc_error",
    episode_key: str = "global_episode",
    max_episode: Optional[int] = None,
    window_size: int = 5,
    consecutive_windows: int = 3,
    reward_rel_threshold: float = 0.05,
    avgvd_rel_threshold: float = 0.01,
    soc_abs_threshold: float = 0.01,
    soc_target_threshold: float = 0.05,
    eps: float = 1e-8,
):
    if len(eval_log) < window_size + consecutive_windows:
        return f">{max_episode}" if max_episode is not None else None

    stable_count = 0
    for i in range(window_size, len(eval_log)):
        current_window = eval_log[i - window_size + 1 : i + 1]
        previous_window = eval_log[i - window_size : i]

        r_cur = float(np.mean([float(row[metric_reward]) for row in current_window]))
        r_prev = float(np.mean([float(row[metric_reward]) for row in previous_window]))
        v_cur = float(np.mean([float(row[metric_avgvd]) for row in current_window]))
        v_prev = float(np.mean([float(row[metric_avgvd]) for row in previous_window]))
        s_cur = float(np.mean([float(row[metric_soc]) for row in current_window]))
        s_prev = float(np.mean([float(row[metric_soc]) for row in previous_window]))

        stable = (
            abs(r_cur - r_prev) / (abs(r_prev) + eps) < reward_rel_threshold
            and abs(v_cur - v_prev) / (abs(v_prev) + eps) < avgvd_rel_threshold
            and abs(s_cur - s_prev) < soc_abs_threshold
            and s_cur < soc_target_threshold
        )
        stable_count = stable_count + 1 if stable else 0
        if stable_count >= consecutive_windows:
            return int(eval_log[i].get(episode_key, i))

    return f">{max_episode}" if max_episode is not None else None

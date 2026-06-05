import csv
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_ROOT / "result" / "experiment5_2_baselines"
RL_DIR = PROJECT_ROOT / "result" / "sparse_33bus_main" / "electric_metric"
OUTPUT_DIR = PROJECT_ROOT / "result" / "final_test_tables"

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4, 5)
TARGET_EPISODE = 1200
DAY_COUNT_PER_DIFFICULTY = 3

METHODS = [
    "NoControl",
    "Rule",
    "MPC_true_forecast",
    "RARR_VIB_SAC",
    "SAC_VIB",
]

METRICS = [
    "original_reward",
    "avg_voltage_deviation",
    "terminal_soc_error",
    "voltage_violation_rate",
]

BASELINE_FILES = {
    "NoControl": "Nocontrol.csv",
    "Rule": "Rule.csv",
    "MPC_true_forecast": "MPC_true_forecast.csv",
}

RL_METHOD_TO_FILE_METHOD = {
    "RARR_VIB_SAC": "RARR_VIB",
    "SAC_VIB": "VIB",
}


def clean_text(value):
    return str(value or "").strip()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {clean_text(key): clean_text(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def relative_percent(value, baseline_value):
    baseline = float(baseline_value)
    if abs(baseline) < 1e-12:
        return 0.0
    return (float(value) - baseline) / abs(baseline) * 100.0


def sample_stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance),
    }


def pooled_stats(groups):
    """Pool grouped means/stds into one sample mean and variance.

    Each group is represented as (n, mean, std). Input std is treated as a
    within-group sample standard deviation.
    """
    clean_groups = [
        (int(n), float(mean), float(std))
        for n, mean, std in groups
        if n is not None and int(n) > 0 and mean is not None and std is not None
    ]
    if not clean_groups:
        return None

    total_n = sum(n for n, _, _ in clean_groups)
    mean = sum(n * group_mean for n, group_mean, _ in clean_groups) / total_n
    if total_n <= 1:
        variance = 0.0
    else:
        ss_within = sum((n - 1) * (group_std ** 2) for n, _, group_std in clean_groups)
        ss_between = sum(n * ((group_mean - mean) ** 2) for n, group_mean, _ in clean_groups)
        variance = (ss_within + ss_between) / (total_n - 1)
    return {
        "n": total_n,
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(max(variance, 0.0)),
    }


def format_mean_std(stats, integer=False):
    if not stats:
        return ""
    if integer:
        return f"{int(round(stats['mean']))}±{int(round(stats['std']))}"
    return f"{stats['mean']:.2f}±{stats['std']:.2f}"


def build_no_control_reference():
    rows = read_csv(BASELINE_DIR / "Nocontrol.csv")
    reference = {}
    for row in rows:
        if clean_text(row.get("scope")) != "test_split":
            continue
        difficulty = clean_text(row.get("difficulty"))
        if difficulty in DIFFICULTIES:
            reference[difficulty] = {metric: to_float(row.get(metric)) for metric in METRICS}
    missing = [difficulty for difficulty in DIFFICULTIES if difficulty not in reference]
    if missing:
        raise RuntimeError(f"Missing NoControl reference rows for: {missing}")
    return reference


def get_test_split_rows(path):
    rows = read_csv(path)
    return {
        clean_text(row.get("difficulty")): row
        for row in rows
        if clean_text(row.get("scope")) == "test_split"
        and clean_text(row.get("difficulty")) in DIFFICULTIES
    }


def baseline_metric_stats(method, metric, reference):
    if method == "NoControl":
        return pooled_stats([(DAY_COUNT_PER_DIFFICULTY, 0.0, 0.0) for _ in DIFFICULTIES])

    rows_by_difficulty = get_test_split_rows(BASELINE_DIR / BASELINE_FILES[method])
    groups = []
    for difficulty in DIFFICULTIES:
        row = rows_by_difficulty[difficulty]
        mean = relative_percent(to_float(row.get(metric)), reference[difficulty][metric])
        std = abs(to_float(row.get(f"{metric}_std"))) / max(abs(reference[difficulty][metric]), 1e-12) * 100.0
        groups.append((DAY_COUNT_PER_DIFFICULTY, mean, std))
    return pooled_stats(groups)


def final_rl_rows(file_method, seed):
    path = RL_DIR / f"SAC_Sparse33BusEnv_{file_method}_nonllm_seed{seed}.csv"
    rows = read_csv(path)
    final_rows = {}
    for row in rows:
        if int(float(row.get("global_episode", -1))) != TARGET_EPISODE:
            continue
        difficulty = clean_text(row.get("test_difficulty"))
        if difficulty in DIFFICULTIES:
            final_rows[difficulty] = row
    return final_rows


def rl_metric_stats(method, metric, reference):
    file_method = RL_METHOD_TO_FILE_METHOD[method]
    groups = []
    for seed in SEEDS:
        rows_by_difficulty = final_rl_rows(file_method, seed)
        for difficulty in DIFFICULTIES:
            if difficulty not in rows_by_difficulty:
                continue
            row = rows_by_difficulty[difficulty]
            mean = relative_percent(to_float(row.get(metric)), reference[difficulty][metric])
            std = abs(to_float(row.get(f"{metric}_std"))) / max(abs(reference[difficulty][metric]), 1e-12) * 100.0
            groups.append((DAY_COUNT_PER_DIFFICULTY, mean, std))
    return pooled_stats(groups)


def convergence_stats(method):
    if method not in RL_METHOD_TO_FILE_METHOD:
        return None
    file_method = RL_METHOD_TO_FILE_METHOD[method]
    values = []
    for seed in SEEDS:
        path = RL_DIR / f"SAC_Sparse33BusEnv_{file_method}_nonllm_seed{seed}_convergence.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        if not rows:
            continue
        value = rows[-1].get("episodes_to_convergence")
        if value:
            values.append(to_float(value))
    return sample_stats(values)


def collect_method_stats():
    reference = build_no_control_reference()
    records = {}
    for method in METHODS:
        method_stats = {}
        for metric in METRICS:
            if method in BASELINE_FILES:
                method_stats[metric] = baseline_metric_stats(method, metric, reference)
            else:
                method_stats[metric] = rl_metric_stats(method, metric, reference)
        method_stats["episode_convergence"] = convergence_stats(method)
        records[method] = method_stats
    return records


def write_compact_csv(records):
    path = OUTPUT_DIR / "combined_9day_selected.csv"
    columns = [
        "method",
        "original_reward",
        "avg_voltage_deviation",
        "terminal_soc_error",
        "voltage_violation_rate",
        "episode_convergence",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method in METHODS:
            row = {"method": method}
            for metric in METRICS:
                row[metric] = format_mean_std(records[method][metric])
            row["episode_convergence"] = format_mean_std(records[method]["episode_convergence"], integer=True)
            writer.writerow(row)
    print(f"saved: {path}")


def write_stats_csv(records):
    path = OUTPUT_DIR / "combined_9day_selected_stats.csv"
    columns = ["method"]
    for metric in [*METRICS, "episode_convergence"]:
        columns.extend([f"{metric}_n", f"{metric}_mean", f"{metric}_std", f"{metric}_variance"])

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method in METHODS:
            row = {"method": method}
            for metric in [*METRICS, "episode_convergence"]:
                stats = records[method].get(metric)
                row[f"{metric}_n"] = "" if not stats else stats["n"]
                row[f"{metric}_mean"] = "" if not stats else f"{stats['mean']:.6f}"
                row[f"{metric}_std"] = "" if not stats else f"{stats['std']:.6f}"
                row[f"{metric}_variance"] = "" if not stats else f"{stats['variance']:.6f}"
            writer.writerow(row)
    print(f"saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_method_stats()
    write_compact_csv(records)
    write_stats_csv(records)


if __name__ == "__main__":
    main()

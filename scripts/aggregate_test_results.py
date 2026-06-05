import csv
import os
import re
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

from race.utils.metrics import compute_episodes_to_convergence


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_ROOT / "result" / "experiment5_2_baselines"
RL_DIR = PROJECT_ROOT / "result" / "sparse_33bus_main" / "electric_metric"
ORACLE_SUMMARY_PATH = PROJECT_ROOT / "result" / "oracle_sac_rarr_dp" / "oracle_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "result" / "final_test_tables"

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4, 5)
TARGET_EPISODE = 1200

RELATIVE_METRICS = [
    "original_reward",
    "avg_voltage_deviation",
    "voltage_deviation_max_pu",
    "terminal_soc_error",
    "voltage_violation_rate",
]

SUCCESS_METRICS = [
    "soc_success_rate",
]

EXTRA_RAW_METRICS = [
    "episode_convergence",
]

OUTPUT_COLUMNS = [
    "method",
    "seed_count",
    *RELATIVE_METRICS,
    *SUCCESS_METRICS,
    *EXTRA_RAW_METRICS,
]

BASELINE_FILES = OrderedDict([
    ("NoControl", "Nocontrol.csv"),
    ("Rule", "Rule.csv"),
    ("MPC_short", "MPC_short.csv"),
    ("MPC_medium", "MPC_medium.csv"),
    ("MPC_long", "MPC_long.csv"),
    ("MPC_true_forecast", "MPC_true_forecast.csv"),
])

RL_PATTERN = re.compile(r"^SAC_Sparse33BusEnv_(?P<method>.+)_nonllm_seed(?P<seed>\d+)\.csv$")
RL_METHOD_ORDER = OrderedDict([
    ("RARR", "RARR_SAC"),
    ("RARR_VIB", "RARR_VIB_SAC"),
    ("RD", "SAC_RD"),
    ("RRD", "SAC_RRD"),
    ("VIB", "SAC_VIB"),
    ("Diaster", "SAC_Diaster"),
])


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({clean_text(key): clean_text(value) for key, value in row.items()})
        return rows


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(value):
    return str(value or "").strip()


def mean_std(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def format_mean_std(mean, std):
    if mean is None:
        return ""
    return f"{mean:.2f}±{(0.0 if std is None else std):.2f}"


def format_integer_mean_std(mean, std):
    if mean is None:
        return ""
    return f"{int(round(mean))}±{int(round(0.0 if std is None else std))}"


def convergence_to_number(value, max_episode=TARGET_EPISODE):
    if value in (None, ""):
        return None
    text = clean_text(value)
    censored = text.startswith(">")
    if censored:
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return float(max_episode) if censored else None


def get_baseline_reference():
    rows = read_csv(BASELINE_DIR / "Nocontrol.csv")
    reference = {}
    for row in rows:
        if clean_text(row.get("scope")) != "test_split":
            continue
        difficulty = clean_text(row.get("difficulty"))
        if difficulty in DIFFICULTIES:
            reference[difficulty] = {metric: to_float(row.get(metric)) for metric in RELATIVE_METRICS}
    missing = [difficulty for difficulty in DIFFICULTIES if difficulty not in reference]
    if missing:
        raise FileNotFoundError(f"Missing NoControl test_split baseline rows for: {missing}")
    return reference


def relative_percent(value, baseline_value):
    baseline = float(baseline_value)
    if abs(baseline) < 1e-12:
        return 0.0
    return (float(value) - baseline) / abs(baseline) * 100.0


def build_baseline_records(reference):
    records = defaultdict(dict)
    for method, file_name in BASELINE_FILES.items():
        rows = read_csv(BASELINE_DIR / file_name)
        for row in rows:
            if clean_text(row.get("scope")) != "test_split":
                continue
            difficulty = clean_text(row.get("difficulty"))
            if difficulty not in DIFFICULTIES:
                continue
            record = {"method": method, "seed_count": "1"}
            for metric in RELATIVE_METRICS:
                mean = relative_percent(to_float(row.get(metric)), reference[difficulty][metric])
                std = abs(to_float(row.get(f"{metric}_std"))) / max(abs(reference[difficulty][metric]), 1e-12) * 100.0
                if method == "NoControl":
                    mean = 0.0
                    std = 0.0
                record[metric] = format_mean_std(mean, std)
            for metric in SUCCESS_METRICS:
                mean = to_float(row.get(metric)) * 100.0
                std = to_float(row.get(f"{metric}_std")) * 100.0
                record[metric] = format_mean_std(mean, std)
            record["episode_convergence"] = ""
            records[difficulty][method] = record
    return records


def compute_convergence_episode(method, seed, difficulty):
    eval_path = RL_DIR / f"SAC_Sparse33BusEnv_{method}_nonllm_seed{seed}.csv"
    rows = [
        row for row in read_csv(eval_path)
        if clean_text(row.get("test_difficulty")) == difficulty
    ]
    if rows:
        rows = sorted(rows, key=lambda row: int(float(row.get("global_episode", -1))))
        value = compute_episodes_to_convergence(
            rows,
            metric_reward="original_reward",
            metric_avgvd="avg_voltage_deviation",
            metric_soc="terminal_soc_error",
            max_episode=TARGET_EPISODE,
            window_size=3,
            consecutive_windows=2,
            reward_rel_threshold=0.05,
            avgvd_rel_threshold=0.01,
            soc_abs_threshold=0.01,
            soc_target_threshold=0.05,
        )
        return convergence_to_number(value)

    path = RL_DIR / f"SAC_Sparse33BusEnv_{method}_nonllm_seed{seed}_convergence.csv"
    convergence_rows = read_csv(path)
    if not convergence_rows:
        return None
    key = f"{difficulty}_episodes_to_convergence"
    return convergence_to_number(
        convergence_rows[-1].get(key) or convergence_rows[-1].get("episodes_to_convergence")
    )


def build_rl_records(reference):
    method_seed_rows = defaultdict(lambda: defaultdict(dict))
    for path in RL_DIR.glob("SAC_Sparse33BusEnv_*_nonllm_seed*.csv"):
        if path.name.endswith("_train.csv") or path.name.endswith("_convergence.csv"):
            continue
        match = RL_PATTERN.match(path.name)
        if not match:
            continue
        method = match.group("method")
        seed = int(match.group("seed"))
        if method not in RL_METHOD_ORDER or seed not in SEEDS:
            continue
        rows = read_csv(path)
        for row in rows:
            if int(float(row.get("global_episode", -1))) != TARGET_EPISODE:
                continue
            difficulty = clean_text(row.get("test_difficulty"))
            if difficulty in DIFFICULTIES:
                method_seed_rows[method][seed][difficulty] = row

    records = defaultdict(dict)
    for method, label in RL_METHOD_ORDER.items():
        for difficulty in DIFFICULTIES:
            seed_rows = [
                method_seed_rows[method][seed][difficulty]
                for seed in SEEDS
                if difficulty in method_seed_rows[method].get(seed, {})
            ]
            if not seed_rows:
                continue

            record = {"method": label, "seed_count": str(len(seed_rows))}
            for metric in RELATIVE_METRICS:
                values = [
                    relative_percent(to_float(row.get(metric)), reference[difficulty][metric])
                    for row in seed_rows
                ]
                record[metric] = format_mean_std(*mean_std(values))
            for metric in SUCCESS_METRICS:
                values = [to_float(row.get(metric)) * 100.0 for row in seed_rows]
                record[metric] = format_mean_std(*mean_std(values))
            convergence_values = [
                compute_convergence_episode(method, seed, difficulty)
                for seed in SEEDS
                if difficulty in method_seed_rows[method].get(seed, {})
            ]
            convergence_values = [value for value in convergence_values if value is not None]
            record["episode_convergence"] = format_integer_mean_std(*mean_std(convergence_values))
            records[difficulty][label] = record
    return records


def build_oracle_records(reference):
    rows = read_csv(ORACLE_SUMMARY_PATH)
    if not rows:
        return defaultdict(dict)
    by_difficulty = defaultdict(list)
    for row in rows:
        difficulty = clean_text(row.get("difficulty"))
        if difficulty in DIFFICULTIES:
            by_difficulty[difficulty].append(row)

    records = defaultdict(dict)
    for difficulty, difficulty_rows in by_difficulty.items():
        record = {"method": "Oracle_DP", "seed_count": str(len(difficulty_rows))}
        for metric in RELATIVE_METRICS:
            source_metric = "oracle_reward" if metric == "original_reward" else metric
            values = [
                relative_percent(to_float(row.get(source_metric)), reference[difficulty][metric])
                for row in difficulty_rows
            ]
            record[metric] = format_mean_std(*mean_std(values))
        for metric in SUCCESS_METRICS:
            if metric == "success_rate":
                values = [to_float(row.get("success")) * 100.0 for row in difficulty_rows]
            else:
                values = []
            record[metric] = format_mean_std(*mean_std(values)) if values else ""
        record["episode_convergence"] = ""
        records[difficulty]["Oracle_DP"] = record
    return records


def merge_records(*record_groups):
    merged = defaultdict(dict)
    for group in record_groups:
        for difficulty, records in group.items():
            merged[difficulty].update(records)
    return merged


def write_outputs(records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preferred_order = [
        *BASELINE_FILES.keys(),
        *RL_METHOD_ORDER.values(),
        "Oracle_DP",
    ]
    for difficulty in DIFFICULTIES:
        path = OUTPUT_DIR / f"{difficulty}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for method in preferred_order:
                if method in records[difficulty]:
                    writer.writerow({column: records[difficulty][method].get(column, "") for column in OUTPUT_COLUMNS})
        print(f"saved: {path}")


def main():
    reference = get_baseline_reference()
    records = merge_records(
        build_baseline_records(reference),
        build_rl_records(reference),
        build_oracle_records(reference),
    )
    write_outputs(records)


if __name__ == "__main__":
    main()

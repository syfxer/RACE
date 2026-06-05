from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "result" / "sparse_33bus_main" / "electric_metric"
BASELINE_DIR = PROJECT_ROOT / "result" / "experiment5_2_baselines"
OUTPUT_DIR = PROJECT_ROOT / "figure" / "evaluator_sensitivity"

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4, 5)
TARGET_EPISODE = 1200

METHOD_PAIRS = [
    ("RD", "RARR_RD", "RD"),
    ("RRD", "RARR_RRD", "RRD"),
    ("Diaster", "RARR_Diaster", "Diaster"),
    ("VIB", "RARR_VIB", "VIB"),
]

METRICS = [
    ("original_reward", "Reward ↑"),
    ("terminal_soc_error", "SOC error ↓"),
    ("avg_voltage_deviation", "AvgVD ↓"),
    ("voltage_violation_rate", "Violation rate ↓"),
    ("episode_convergence", "Episodes to\nconvergence ↓"),
]

# Colors sampled to match the reference bar palette. Edit here for future tuning.
BAR_COLORS = {
    "RD": "#D9E3EC",
    "RRD": "#9EB4CE",
    "Diaster": "#6A89B0",
    "VIB": "#3D7FB6",
}


def clean_text(value) -> str:
    return str(value or "").strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {clean_text(key): clean_text(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def reconstruct_three_samples(mean: float, std: float) -> list[float]:
    """Reconstruct three test-day samples from stored mean/std.

    The evaluation files store difficulty-level mean/std over three test days.
    The triplet [mean-std, mean, mean+std] preserves that mean and sample std,
    which lets the plot include both seed variation and multi-day variation.
    """
    return [float(mean) - float(std), float(mean), float(mean) + float(std)]


def final_rows(method: str, seed: int) -> dict[str, dict[str, str]]:
    path = DATA_DIR / f"SAC_Sparse33BusEnv_{method}_nonllm_seed{seed}.csv"
    rows = read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if int(float(row.get("global_episode", -1))) != TARGET_EPISODE:
            continue
        difficulty = clean_text(row.get("test_difficulty"))
        if difficulty in DIFFICULTIES:
            out[difficulty] = row
    return out


def convergence_value(method: str, seed: int) -> float | None:
    path = DATA_DIR / f"SAC_Sparse33BusEnv_{method}_nonllm_seed{seed}_convergence.csv"
    rows = read_csv(path)
    if not rows:
        return None
    value = rows[-1].get("episodes_to_convergence")
    if value in (None, ""):
        return None
    return to_float(value)


def metric_samples(row: dict[str, str], metric: str) -> list[float]:
    mean = to_float(row.get(metric))
    std = abs(to_float(row.get(f"{metric}_std")))
    return reconstruct_three_samples(mean, std)


def no_control_reference_rows() -> dict[str, dict[str, str]]:
    rows = read_csv(BASELINE_DIR / "Nocontrol.csv")
    return {
        clean_text(row.get("difficulty")): row
        for row in rows
        if clean_text(row.get("scope")) == "test_split"
        and clean_text(row.get("difficulty")) in DIFFICULTIES
    }


def relative_to_no_control(value: float, no_control_value: float) -> float:
    denominator = abs(float(no_control_value))
    if denominator < 1e-12:
        return 0.0
    return (float(value) - float(no_control_value)) / denominator * 100.0


def paired_no_control_improvement_delta(
    base_method: str,
    evaluator_method: str,
    metric: str,
    no_control_rows: dict[str, dict[str, str]],
) -> list[float]:
    """Return evaluator-vs-baseline delta under the NoControl-relative scale.

    For each reconstructed seed-day sample, compute:
        rel_evaluator = (evaluator - NoControl) / |NoControl| * 100
        rel_baseline = (baseline - NoControl) / |NoControl| * 100
        delta = rel_evaluator - rel_baseline

    This keeps the sensitivity plot on the same scale as the final 9-day table
    and avoids exploding ratios when the baseline SOC error is near zero.
    """
    changes: list[float] = []
    for seed in SEEDS:
        base_rows = final_rows(base_method, seed)
        evaluator_rows = final_rows(evaluator_method, seed)
        for difficulty in DIFFICULTIES:
            if (
                difficulty not in base_rows
                or difficulty not in evaluator_rows
                or difficulty not in no_control_rows
            ):
                continue
            base_samples = metric_samples(base_rows[difficulty], metric)
            evaluator_samples = metric_samples(evaluator_rows[difficulty], metric)
            no_control_samples = metric_samples(no_control_rows[difficulty], metric)
            for sample_index, (base_value, evaluator_value) in enumerate(zip(base_samples, evaluator_samples)):
                # The baseline CSV stores three reconstructed day samples in
                # the same order, so align by sample index within each difficulty.
                no_control_value = no_control_samples[sample_index]
                evaluator_relative = relative_to_no_control(evaluator_value, no_control_value)
                base_relative = relative_to_no_control(base_value, no_control_value)
                changes.append(evaluator_relative - base_relative)
    return changes


def paired_convergence_relative_delta(base_method: str, evaluator_method: str) -> list[float]:
    changes: list[float] = []
    for seed in SEEDS:
        base_value = convergence_value(base_method, seed)
        evaluator_value = convergence_value(evaluator_method, seed)
        if base_value is None or evaluator_value is None or abs(base_value) < 1e-12:
            continue
        changes.append((evaluator_value - base_value) / abs(base_value) * 100.0)
    return changes


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def collect_stats() -> dict[str, dict[str, tuple[float, float, int]]]:
    stats: dict[str, dict[str, tuple[float, float, int]]] = {}
    no_control_rows = no_control_reference_rows()
    for metric, _ in METRICS:
        stats[metric] = {}
        for base_method, evaluator_method, label in METHOD_PAIRS:
            if metric == "episode_convergence":
                values = paired_convergence_relative_delta(base_method, evaluator_method)
            else:
                values = paired_no_control_improvement_delta(
                    base_method,
                    evaluator_method,
                    metric,
                    no_control_rows,
                )
            mean, std = sample_mean_std(values)
            stats[metric][label] = (mean, std, len(values))
    return stats


def percent_formatter(value, _position) -> str:
    if abs(value) < 1e-9:
        return "0%"
    return f"{value:.0f}%"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_stats_csv(stats: dict[str, dict[str, tuple[float, float, int]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "evaluator_sensitivity_stats.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "method",
                "n",
                "no_control_relative_delta_mean_percent",
                "no_control_relative_delta_std_percent",
            ],
        )
        writer.writeheader()
        for metric, _ in METRICS:
            for _, _, label in METHOD_PAIRS:
                mean, std, n = stats[metric][label]
                writer.writerow(
                    {
                        "metric": metric,
                        "method": label,
                        "n": n,
                        "no_control_relative_delta_mean_percent": f"{mean:.6f}",
                        "no_control_relative_delta_std_percent": f"{std:.6f}",
                    }
                )
    print(f"saved: {path}")


def plot(stats: dict[str, dict[str, tuple[float, float, int]]]) -> None:
    labels = [label for _, _, label in METHOD_PAIRS]
    metric_labels = [metric_label for _, metric_label in METRICS]
    group_centers = np.arange(len(METRICS), dtype=float) * 0.86
    bar_width = 0.13
    offsets = (np.arange(len(labels), dtype=float) - (len(labels) - 1) / 2.0) * (bar_width + 0.018)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=False)
    for method_index, label in enumerate(labels):
        means = []
        stds = []
        for metric, _ in METRICS:
            mean, std, _ = stats[metric][label]
            means.append(mean)
            stds.append(std)
        ax.bar(
            group_centers + offsets[method_index],
            means,
            width=bar_width,
            label=label,
            color=BAR_COLORS[label],
            edgecolor="#222222",
            linewidth=0.9,
            yerr=stds,
            error_kw={"elinewidth": 1.0, "ecolor": "#222222", "capsize": 3, "capthick": 1.0},
            zorder=3,
        )

    for separator in (group_centers[:-1] + np.diff(group_centers) / 2.0):
        ax.axvline(separator, color="#C20C0C", linestyle="--", linewidth=1.1, alpha=0.9, zorder=2)

    ax.axhline(0.0, color="#333333", linewidth=0.85, zorder=2)
    ax.set_ylabel("Evaluator gain over baseline")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels, rotation=0, ha="center")
    ax.set_ylim(-30.0, 20.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#9a9a9a", alpha=0.55, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.14, top=0.86)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "evaluator_sensitivity_bars.png"
    pdf_path = OUTPUT_DIR / "evaluator_sensitivity_bars.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    configure_matplotlib()
    stats = collect_stats()
    write_stats_csv(stats)
    plot(stats)


if __name__ == "__main__":
    main()

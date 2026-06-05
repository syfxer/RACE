from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "result" / "pre_ex" / "electric_metric"
OUTPUT_DIR = PROJECT_ROOT / "figure" / "pre_ex_learning_curves"
SEEDS = tuple(range(40, 45))
ENV_NAME = "SparseHardcodedDayEnv"
MODEL_TAG = "nonllm"
SMOOTHING_WINDOW = 5
REQUIRE_ALL_SEEDS = True

# Add tuples such as ("SAC", "RARR_VIB") here when a series should be temporarily hidden.
SKIP_SERIES = set()


# Method colors are centralized here for frequent manual tuning.
METHOD_COLORS = {
    "None": "#8aa6c8",      # sparse reward without evaluator
    "VIB": "#d98c1f",       # VIB evaluator
    "RARR_VIB": "#8b1e77",  # RACE + VIB evaluator
}

METHOD_LINESTYLES = {
    "None": "--",
    "VIB": "-",
    "RARR_VIB": "-",
}

POLICIES = ["SAC", "TD3", "DDPG", "PPO"]


@dataclass(frozen=True)
class MethodSpec:
    label: str
    rd_method: str
    color: str
    linestyle: str


METHODS = [
    MethodSpec("Sparse", "None", METHOD_COLORS["None"], METHOD_LINESTYLES["None"]),
    MethodSpec("VIB", "VIB", METHOD_COLORS["VIB"], METHOD_LINESTYLES["VIB"]),
    MethodSpec("RACE", "RARR_VIB", METHOD_COLORS["RARR_VIB"], METHOD_LINESTYLES["RARR_VIB"]),
]

PLOTS = [
    {
        "metric": "total_reward",
        "ylabel": "Total reward",
    },
    {
        "metric": "terminal_soc_error",
        "ylabel": "Terminal SOC error",
    },
    {
        "metric": "average_voltage_deviation",
        "ylabel": "Average voltage deviation",
    },
]


def file_path(policy: str, rd_method: str, seed: int) -> Path:
    return DATA_DIR / f"{policy}_{ENV_NAME}_{rd_method}_{MODEL_TAG}_seed{seed}.csv"


def read_metric_rows(path: Path, metric: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {
                (key.strip() if key else ""): (value.strip() if value else value)
                for key, value in row.items()
            }
            raw_episode = normalized.get("global_episode")
            raw_value = normalized.get(metric)
            if raw_episode in (None, "") or raw_value in (None, ""):
                continue
            try:
                episode = int(float(raw_episode))
                value = float(raw_value)
            except ValueError:
                continue
            rows[episode] = value
    return rows


def moving_average(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values

    smoothed = np.empty_like(values, dtype=float)
    half = window // 2
    for i in range(len(values)):
        left = max(0, i - half)
        right = min(len(values), i + half + 1)
        smoothed[i] = float(np.nanmean(values[left:right]))
    return smoothed


def aggregate_series(policy: str, rd_method: str, metric: str):
    if (policy, rd_method) in SKIP_SERIES:
        print(f"[skip] explicitly skipped while running: {policy}_{rd_method}")
        return None

    seed_rows = []
    missing = []
    for seed in SEEDS:
        path = file_path(policy, rd_method, seed)
        if not path.exists():
            missing.append(path.name)
            continue
        seed_rows.append(read_metric_rows(path, metric))

    if missing:
        print(f"[skip] missing {policy}_{rd_method}: {', '.join(missing)}")
        if REQUIRE_ALL_SEEDS:
            return None

    if not seed_rows:
        return None

    common_episodes = sorted(set.intersection(*(set(row) for row in seed_rows)))
    if not common_episodes:
        return None

    matrix = np.asarray(
        [[row[episode] for episode in common_episodes] for row in seed_rows],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
    return {
        "episodes": np.asarray(common_episodes, dtype=float),
        "mean": moving_average(mean),
        "std": moving_average(std),
        "seed_count": matrix.shape[0],
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13.5,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 14,
            "axes.linewidth": 1.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.55, color="#9a9a9a", alpha=0.48)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.tick_params(axis="x", labelbottom=True)


def plot_panel() -> None:
    fig, axes = plt.subplots(
        nrows=len(PLOTS),
        ncols=len(POLICIES),
        figsize=(15.2, 7.2),
        sharex=True,
        constrained_layout=False,
    )

    legend_handles = {}
    for row_idx, plot_cfg in enumerate(PLOTS):
        for col_idx, policy in enumerate(POLICIES):
            ax = axes[row_idx, col_idx]
            style_axis(ax)

            if row_idx == 0:
                ax.set_title(policy, pad=4)
            if col_idx == 0:
                ax.set_ylabel(plot_cfg["ylabel"])
            if row_idx == len(PLOTS) - 1:
                ax.set_xlabel("Episodes")

            for method in METHODS:
                aggregated = aggregate_series(policy, method.rd_method, plot_cfg["metric"])
                if aggregated is None:
                    continue

                x = aggregated["episodes"]
                mean = aggregated["mean"]
                std = aggregated["std"]
                line_label = f"{policy}_{method.label}"
                (line,) = ax.plot(
                    x,
                    mean,
                    label=line_label,
                    color=method.color,
                    linestyle=method.linestyle,
                    linewidth=2.2,
                )
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=method.color,
                    alpha=0.12,
                    linewidth=0.0,
                )
                legend_handles.setdefault(method.label, line)

            if not ax.lines:
                ax.text(
                    0.5,
                    0.5,
                    "No stable data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                    fontsize=11,
                )

    fig.legend(
        [legend_handles[m.label] for m in METHODS if m.label in legend_handles],
        [m.label for m in METHODS if m.label in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        handlelength=3.0,
        columnspacing=2.8,
        borderpad=0.2,
    )
    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.065, right=0.995, hspace=0.42, wspace=0.11)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "pre_ex_3x4_learning_curves.png"
    pdf_path = OUTPUT_DIR / "pre_ex_3x4_learning_curves.pdf"
    legacy_png_path = OUTPUT_DIR / "pre_ex_4x3_learning_curves.png"
    legacy_pdf_path = OUTPUT_DIR / "pre_ex_4x3_learning_curves.pdf"
    for path in (png_path, pdf_path, legacy_png_path, legacy_pdf_path):
        save_kwargs = {"dpi": 450, "bbox_inches": "tight"} if path.suffix.lower() == ".png" else {"bbox_inches": "tight"}
        try:
            fig.savefig(path, **save_kwargs)
        except PermissionError:
            fallback_path = path.with_name(f"{path.stem}_new{path.suffix}")
            fig.savefig(fallback_path, **save_kwargs)
            print(f"locked, saved fallback: {fallback_path}")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    print(f"saved: {legacy_png_path}")
    print(f"saved: {legacy_pdf_path}")


def print_color_codes() -> None:
    print("Method color codes:")
    for rd_method, code in METHOD_COLORS.items():
        print(f"  {rd_method}: {code}")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    configure_matplotlib()
    print_color_codes()
    plot_panel()


if __name__ == "__main__":
    main()

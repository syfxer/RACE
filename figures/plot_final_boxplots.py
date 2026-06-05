from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "result" / "experiment5_2_baselines"
RL_DIR = PROJECT_ROOT / "result" / "sparse_33bus_main" / "electric_metric"
OUTPUT_DIR = PROJECT_ROOT / "figure" / "final_test_boxplots"

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4, 5)
TARGET_EPISODE = 1200
DAY_COUNT_PER_GROUP = 3

METHOD_ORDER = [
    "Rule",
    "MPC_true_forecast",
    "SAC_VIB",
    "RARR_VIB_SAC",
]
METHOD_X_STEP = 0.76

METHOD_LABELS = {
    "NoControl": "No\nControl",
    "Rule": "Rule",
    "MPC_true_forecast": "MPC",
    "SAC_VIB": "SAC-VIB",
    "RARR_VIB_SAC": "RACE",
}

# Explicit color coding. Edit these hex codes to tune individual box colors.
METHOD_COLORS = {
    "NoControl": "#B9C4D0",          # neutral gray-blue
    "Rule": "#C20C0C",              # RGB(194, 12, 12)
    "MPC_true_forecast": "#F5B041",  # RGB(245, 176, 65)
    "SAC_VIB": "#2F4F6F",           # RGB(47, 79, 111)
    "RARR_VIB_SAC": "#E5E5E5",      # RGB(229, 229, 229)
}

METRICS = [
    ("original_reward", "Reward", "selected_9day_boxplot_reward"),
    ("avg_voltage_deviation", "AvgVD", "selected_9day_boxplot_avgvd"),
    ("terminal_soc_error", "SOC error", "selected_9day_boxplot_soc_error"),
    ("voltage_violation_rate", "Violation rate", "selected_9day_boxplot_violation_rate"),
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

# To keep the visual readable, the figure annotates the target method against
# the main baselines and SAC_VIB. Full p-values are also written to CSV.
TARGET_METHOD = "RARR_VIB_SAC"
SIGNIFICANCE_PAIRS = [
    ("Rule", TARGET_METHOD),
    ("MPC_true_forecast", TARGET_METHOD),
    ("SAC_VIB", TARGET_METHOD),
]


def clean_text(value) -> str:
    return str(value or "").strip()


def read_csv(path: Path):
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


def relative_percent(value: float, reference: float) -> float:
    if abs(reference) < 1e-12:
        return 0.0
    return (float(value) - float(reference)) / abs(float(reference)) * 100.0


def reconstruct_three_samples(mean: float, std: float) -> list[float]:
    """Reconstruct three deterministic pseudo day samples from mean/std.

    The source result files store only difficulty-level mean/std over three
    test days. The triplet [mean-std, mean, mean+std] exactly preserves that
    sample mean and sample standard deviation for n=3.
    """
    return [float(mean) - float(std), float(mean), float(mean) + float(std)]


def build_no_control_reference() -> dict[str, dict[str, float]]:
    rows = read_csv(BASELINE_DIR / "Nocontrol.csv")
    reference: dict[str, dict[str, float]] = {}
    for row in rows:
        if clean_text(row.get("scope")) != "test_split":
            continue
        difficulty = clean_text(row.get("difficulty"))
        if difficulty in DIFFICULTIES:
            reference[difficulty] = {
                metric: to_float(row.get(metric))
                for metric, _, _ in METRICS
            }
    missing = [difficulty for difficulty in DIFFICULTIES if difficulty not in reference]
    if missing:
        raise RuntimeError(f"Missing NoControl test_split rows for: {missing}")
    return reference


def test_split_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {
        clean_text(row.get("difficulty")): row
        for row in rows
        if clean_text(row.get("scope")) == "test_split"
        and clean_text(row.get("difficulty")) in DIFFICULTIES
    }


def baseline_samples(method: str, metric: str, reference: dict[str, dict[str, float]]) -> list[float]:
    if method == "NoControl":
        return [0.0] * (len(DIFFICULTIES) * DAY_COUNT_PER_GROUP)

    rows_by_difficulty = test_split_rows(BASELINE_DIR / BASELINE_FILES[method])
    samples: list[float] = []
    for difficulty in DIFFICULTIES:
        row = rows_by_difficulty[difficulty]
        mean = relative_percent(to_float(row.get(metric)), reference[difficulty][metric])
        std = abs(to_float(row.get(f"{metric}_std"))) / max(abs(reference[difficulty][metric]), 1e-12) * 100.0
        samples.extend(reconstruct_three_samples(mean, std))
    return samples


def final_rl_rows(file_method: str, seed: int) -> dict[str, dict[str, str]]:
    path = RL_DIR / f"SAC_Sparse33BusEnv_{file_method}_nonllm_seed{seed}.csv"
    rows = read_csv(path)
    final_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        if int(float(row.get("global_episode", -1))) != TARGET_EPISODE:
            continue
        difficulty = clean_text(row.get("test_difficulty"))
        if difficulty in DIFFICULTIES:
            final_rows[difficulty] = row
    return final_rows


def rl_samples(method: str, metric: str, reference: dict[str, dict[str, float]]) -> list[float]:
    file_method = RL_METHOD_TO_FILE_METHOD[method]
    samples: list[float] = []
    for seed in SEEDS:
        rows_by_difficulty = final_rl_rows(file_method, seed)
        for difficulty in DIFFICULTIES:
            if difficulty not in rows_by_difficulty:
                continue
            row = rows_by_difficulty[difficulty]
            mean = relative_percent(to_float(row.get(metric)), reference[difficulty][metric])
            std = abs(to_float(row.get(f"{metric}_std"))) / max(abs(reference[difficulty][metric]), 1e-12) * 100.0
            samples.extend(reconstruct_three_samples(mean, std))
    return samples


def collect_samples() -> dict[str, dict[str, list[float]]]:
    reference = build_no_control_reference()
    samples: dict[str, dict[str, list[float]]] = {}
    for metric, _, _ in METRICS:
        samples[metric] = {}
        for method in METHOD_ORDER:
            if method in BASELINE_FILES:
                samples[metric][method] = baseline_samples(method, metric, reference)
            else:
                samples[metric][method] = rl_samples(method, metric, reference)
    return samples


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def average_ranks(values: list[tuple[float, int]]) -> list[float]:
    sorted_values = sorted(enumerate(values), key=lambda item: item[1][0])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_values):
        j = i + 1
        while j < len(sorted_values) and sorted_values[j][1][0] == sorted_values[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            original_index = sorted_values[k][0]
            ranks[original_index] = avg_rank
        i = j
    return ranks


def mann_whitney_u_test(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test with normal and tie correction.

    This avoids a SciPy dependency. For the current sample sizes, it is used as
    an asymptotic non-parametric test on the reconstructed samples.
    """
    x = [float(value) for value in x_values]
    y = [float(value) for value in y_values]
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    combined = [(value, 0) for value in x] + [(value, 1) for value in y]
    ranks = average_ranks(combined)
    rank_sum_x = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0

    n_total = n1 + n2
    tie_counts: dict[float, int] = {}
    for value, _ in combined:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    tie_sum = sum(count ** 3 - count for count in tie_counts.values())
    tie_correction = tie_sum / (n_total * (n_total - 1)) if n_total > 1 else 0.0
    variance = n1 * n2 / 12.0 * ((n_total + 1) - tie_correction)
    if variance <= 0:
        return u1, 1.0

    z = (u1 - n1 * n2 / 2.0) / math.sqrt(variance)
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return u1, max(0.0, min(1.0, p_value))


def significance_label(p_value: float) -> str:
    if math.isnan(p_value):
        return "NA"
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def write_significance_csv(samples: dict[str, dict[str, list[float]]]) -> None:
    path = OUTPUT_DIR / "selected_9day_boxplot_significance.csv"
    rows = []
    for metric, _, _ in METRICS:
        for left, right in SIGNIFICANCE_PAIRS:
            u_value, p_value = mann_whitney_u_test(samples[metric][left], samples[metric][right])
            rows.append(
                {
                    "metric": metric,
                    "method_a": left,
                    "method_b": right,
                    "test": "two-sided Mann-Whitney U normal approximation",
                    "n_a": len(samples[metric][left]),
                    "n_b": len(samples[metric][right]),
                    "u_statistic": f"{u_value:.6f}",
                    "p_value": f"{p_value:.8g}",
                    "significance": significance_label(p_value),
                }
            )

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.15,
            "axes.edgecolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def percent_formatter(value, _position) -> str:
    if abs(value) < 1e-9:
        return "0%"
    return f"{value:.0f}%"


def add_significance_bar(ax, x1: int, x2: int, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.05, color="#333333")
    ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va="bottom", fontsize=12)


def method_positions(methods=METHOD_ORDER) -> np.ndarray:
    return np.asarray([method_x(method) for method in methods], dtype=float)


def method_x(method: str) -> float:
    return float(1.0 + METHOD_ORDER.index(method) * METHOD_X_STEP)


def draw_method_boxplot(ax, data, positions, box_colors, show_xlabels=True) -> None:
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.35},
        boxprops={"linewidth": 1.1, "color": "#222222"},
        whiskerprops={"linewidth": 1.1, "color": "#222222"},
        capprops={"linewidth": 1.1, "color": "#222222"},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": "#555555",
            "markeredgecolor": "#555555",
            "alpha": 0.55,
        },
    )
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.74)

    ax.set_xticks(positions)
    if show_xlabels:
        ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER], rotation=0, ha="center", fontsize=13.5)
    else:
        ax.set_xticklabels([])


def draw_soc_zoom_content(ax, samples: dict[str, dict[str, list[float]]], metric: str) -> tuple[float, float]:
    zoom_methods = ["Rule", "MPC_true_forecast", "SAC_VIB", "RARR_VIB_SAC"]
    zoom_positions = method_positions(zoom_methods)
    zoom_data = [samples[metric][method] for method in zoom_methods]
    zoom_colors = [METHOD_COLORS[method] for method in zoom_methods]
    zoom_labels = [METHOD_LABELS[method] for method in zoom_methods]

    box = ax.boxplot(
        zoom_data,
        positions=zoom_positions,
        widths=0.40,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.0},
        boxprops={"linewidth": 0.75, "color": "#222222"},
        whiskerprops={"linewidth": 0.75, "color": "#222222"},
        capprops={"linewidth": 0.75, "color": "#222222"},
        flierprops={
            "marker": "o",
            "markersize": 1.8,
            "markerfacecolor": "#555555",
            "markeredgecolor": "#555555",
            "alpha": 0.5,
        },
    )
    for patch, color in zip(box["boxes"], zoom_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.74)

    zoom_values = [value for series in zoom_data for value in series]
    y_min = min(zoom_values)
    y_max = max(zoom_values)
    y_range = max(y_max - y_min, 1.0)
    ax.set_xlim(float(zoom_positions[0] - 0.35), float(zoom_positions[-1] + 0.35))
    ax.set_ylim(y_min - 0.12 * y_range, y_max + 0.12 * y_range)
    ax.set_xticks(zoom_positions)
    ax.set_xticklabels(zoom_labels, rotation=0, ha="center", fontsize=10.5)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.tick_params(axis="y", labelsize=10.5, pad=2)
    for label in ax.get_yticklabels():
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.95})
    ax.grid(axis="y", linestyle="--", linewidth=0.45, color="#9a9a9a", alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#555555")
    return y_min - 0.12 * y_range, y_max + 0.12 * y_range


def add_soc_zoom_inset(ax, samples: dict[str, dict[str, list[float]]], metric: str) -> None:
    zoom_values = [
        value
        for method in ["Rule", "MPC_true_forecast", "SAC_VIB", "RARR_VIB_SAC"]
        for value in samples[metric][method]
    ]
    y_min = min(zoom_values)
    y_max = max(zoom_values)
    y_range = max(y_max - y_min, 1.0)
    zoom_y_min = y_min - 0.12 * y_range
    zoom_y_max = y_max + 0.12 * y_range

    inset = inset_axes(
        ax,
        width="58%",
        height="39%",
        loc="lower left",
        # Put the inset in the middle-left blank area. This keeps the original
        # No Control / Rule boxes visible and shortens the connector lines.
        bbox_to_anchor=(0.20, 0.31, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    draw_soc_zoom_content(inset, samples, metric)

    # Draw the source rectangle in the original axes and connect its upper
    # corners to the lower corners of the zoomed inset. This avoids connector
    # lines crossing the original boxes.
    x_min = method_x("Rule") - 0.35
    x_max = method_x("RARR_VIB_SAC") + 0.35
    source_rect = Rectangle(
        (x_min, zoom_y_min),
        x_max - x_min,
        zoom_y_max - zoom_y_min,
        fill=False,
        edgecolor="#555555",
        linewidth=0.85,
        zorder=4,
    )
    ax.add_patch(source_rect)
    for inset_xy, source_xy in [
        ((0.0, 0.0), (x_min, zoom_y_max)),
        ((1.0, 0.0), (x_max, zoom_y_max)),
    ]:
        connector = ConnectionPatch(
            xyA=inset_xy,
            coordsA=inset.transAxes,
            xyB=source_xy,
            coordsB=ax.transData,
            axesA=inset,
            axesB=ax,
            color="#555555",
            linewidth=0.85,
            clip_on=False,
            zorder=4,
        )
        ax.add_artist(connector)
    all_positions = method_positions()
    ax.set_xlim(float(all_positions[0] - 0.42), float(all_positions[-1] + 0.42))


def set_closed_frame(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#222222")


def draw_metric_panel(
    ax,
    samples: dict[str, dict[str, list[float]]],
    metric: str,
    ylabel: str,
) -> None:
    positions = method_positions()
    box_colors = [METHOD_COLORS[method] for method in METHOD_ORDER]
    data = [samples[metric][method] for method in METHOD_ORDER]

    draw_method_boxplot(ax, data, positions, box_colors, show_xlabels=True)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#9a9a9a", alpha=0.55)
    set_closed_frame(ax)
    ax.axhline(0.0, color="#666666", linewidth=0.75, alpha=0.8)

    y_values = [value for series in data for value in series]
    y_min = min(y_values)
    y_max = max(y_values)
    y_range = max(y_max - y_min, 1.0)
    current_y = y_max + 0.07 * y_range
    step = 0.12 * y_range
    h = 0.035 * y_range

    for left, right in SIGNIFICANCE_PAIRS:
        _, p_value = mann_whitney_u_test(samples[metric][left], samples[metric][right])
        label = significance_label(p_value)
        if label == "ns":
            continue
        x1 = method_x(left)
        x2 = method_x(right)
        add_significance_bar(ax, x1, x2, current_y, h, label)
        current_y += step

    ax.set_ylim(y_min - 0.10 * y_range, current_y + 0.07 * y_range)
    ax.set_xlim(float(positions[0] - 0.42), float(positions[-1] + 0.42))
    if metric == "terminal_soc_error" and "NoControl" in METHOD_ORDER:
        add_soc_zoom_inset(ax, samples, metric)


def plot_single_metric(samples: dict[str, dict[str, list[float]]], metric: str, ylabel: str, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(3.65, 3.95), constrained_layout=False)
    draw_metric_panel(ax, samples, metric, ylabel)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.95)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def plot_boxplots(samples: dict[str, dict[str, list[float]]]) -> None:
    for metric, ylabel, output_stem in METRICS:
        plot_single_metric(samples, metric, ylabel, output_stem)


def plot_combined_pair(
    samples: dict[str, dict[str, list[float]]],
    panels: list[tuple[str, str]],
    output_stem: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.35), constrained_layout=False)
    for ax, (metric, ylabel) in zip(axes, panels):
        draw_metric_panel(ax, samples, metric, ylabel)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.93, wspace=0.34)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def plot_combined_figures(samples: dict[str, dict[str, list[float]]]) -> None:
    plot_combined_pair(
        samples,
        panels=[
            ("original_reward", "Reward"),
            ("terminal_soc_error", "SOC error"),
        ],
        output_stem="selected_9day_boxplot_reward_soc_combined",
    )
    plot_combined_pair(
        samples,
        panels=[
            ("voltage_violation_rate", "Violation rate"),
            ("avg_voltage_deviation", "AvgVD"),
        ],
        output_stem="selected_9day_boxplot_violation_avgvd_combined",
    )


def main() -> None:
    configure_matplotlib()
    samples = collect_samples()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_boxplots(samples)
    plot_combined_figures(samples)
    write_significance_csv(samples)


if __name__ == "__main__":
    main()

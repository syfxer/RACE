import csv
import os

from race.utils.metrics import CORE_AGGREGATED_METRIC_KEYS, CORE_EPISODE_METRIC_KEYS

TRAIN_EPISODE_METRIC_COLUMNS = [
    "global_episode",
    "stage",
    "stage_episode",
    "scenario_id",
    "difficulty",
    "split",
    "steps",
    *CORE_EPISODE_METRIC_KEYS,
]


EVAL_METRIC_COLUMNS = [
    "global_episode",
    "stage",
    "stage_episode",
    "test_difficulty",
    *CORE_AGGREGATED_METRIC_KEYS,
    *[f"{key}_std" for key in CORE_AGGREGATED_METRIC_KEYS],
]


ELECTRIC_METRIC_COLUMNS = [
    "global_episode",
    "stage",
    "stage_episode",
    "difficulty",
    "split",
    *CORE_AGGREGATED_METRIC_KEYS,
]


def build_electric_metric_dir(result_dir):
    electric_metric_dir = os.path.join(result_dir, "electric_metric")
    os.makedirs(electric_metric_dir, exist_ok=True)
    return electric_metric_dir


def build_electric_metric_path(result_dir, file_name):
    return os.path.join(build_electric_metric_dir(result_dir), f"{file_name}.csv")


def get_last_logged_step(csv_path):
    if not os.path.exists(csv_path):
        return None

    last_step = None
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_step = row.get("global_episode", row.get("step"))
            if raw_step in (None, ""):
                continue
            try:
                last_step = int(float(raw_step))
            except ValueError:
                continue
    return last_step


class ElectricMetricLogger:
    def __init__(self, result_dir, file_name, fieldnames=None, overwrite=False):
        self.path = build_electric_metric_path(result_dir, file_name)
        self.fieldnames = fieldnames or ELECTRIC_METRIC_COLUMNS
        if overwrite and os.path.exists(self.path):
            os.remove(self.path)

    def append_row(self, row_data):
        row = {}
        for field in self.fieldnames:
            value = row_data.get(field)
            row[field] = "" if value is None else value

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        file_exists = os.path.exists(self.path)
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not file_exists or os.path.getsize(self.path) == 0:
                writer.writeheader()
            writer.writerow(row)

    def append(self, step, metrics):
        row = dict(metrics)
        row.setdefault("global_episode", int(step))
        self.append_row(row)

    def append_train_episode(self, global_episode, stage, stage_episode, metrics):
        row = dict(metrics)
        row.update({
            "global_episode": int(global_episode),
            "stage": stage,
            "stage_episode": int(stage_episode),
        })
        self.append_row(row)

    def append_eval(self, global_episode, stage, stage_episode, test_difficulty, metrics):
        row = dict(metrics)
        row.update({
            "global_episode": int(global_episode),
            "stage": stage,
            "stage_episode": int(stage_episode),
            "test_difficulty": test_difficulty,
        })
        self.append_row(row)

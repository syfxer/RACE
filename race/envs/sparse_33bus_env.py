from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from race.utils.metrics import EpisodeMetricAccumulator
from race.envs.battery_env import BatteryManagementEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EASY_CSV = PROJECT_ROOT / "data" / "easy.csv"
DEFAULT_MEDIUM_CSV = PROJECT_ROOT / "data" / "medium.csv"
DEFAULT_HARD_CSV = PROJECT_ROOT / "data" / "hard.csv"
DIFFICULTIES = ("easy", "medium", "hard")
SUPPORTED_THREE_STAGE_POLICIES = ("SAC", "TD3", "DDPG", "PPO")
SUPPORTED_RARR_RESIDUAL_METHODS = ("RARR_RD", "RARR_RRD", "RARR_Diaster")
TEST_RANKS_BY_DIFFICULTY = {
    "easy": {4, 8, 12},
    "medium": {16, 20, 24},
    "hard": {28, 32, 36},
}


@dataclass(frozen=True)
class StageConfig:
    name: str
    quota: Dict[str, int]
    train_difficulties: Sequence[str]


DEFAULT_THREE_STAGE_CURRICULUM = (
    StageConfig(
        name="stage1_easy",
        quota={"easy": 100, "medium": 0, "hard": 0},
        train_difficulties=("easy",),
    ),
    StageConfig(
        name="stage2_easy_medium",
        quota={"easy": 80, "medium": 120, "hard": 0},
        train_difficulties=("easy", "medium"),
    ),
    StageConfig(
        name="stage3_all",
        quota={"easy": 100, "medium": 150, "hard": 250},
        train_difficulties=("easy", "medium", "hard"),
    ),
)


def build_balanced_three_stage_curriculum(total_episodes: int) -> tuple[StageConfig, ...]:
    total = max(int(total_episodes), 1)
    target_easy = total // 3 + (1 if total % 3 > 0 else 0)
    target_medium = total // 3 + (1 if total % 3 > 1 else 0)
    target_hard = total // 3

    stage1_easy = min(max(total // 8, 1), target_easy)
    stage2_total = min(max(total // 4, 1), max(total - stage1_easy - target_hard, 0))
    stage2_medium = min(max(stage2_total * 3 // 5, 0), target_medium)
    stage2_easy = min(stage2_total - stage2_medium, max(target_easy - stage1_easy, 0))

    used_easy = stage1_easy + stage2_easy
    used_medium = stage2_medium
    stage3_easy = max(target_easy - used_easy, 0)
    stage3_medium = max(target_medium - used_medium, 0)
    stage3_hard = max(total - stage1_easy - stage2_easy - stage2_medium - stage3_easy - stage3_medium, 0)

    return (
        StageConfig(
            name="stage1_easy",
            quota={"easy": stage1_easy, "medium": 0, "hard": 0},
            train_difficulties=("easy",),
        ),
        StageConfig(
            name="stage2_easy_medium",
            quota={"easy": stage2_easy, "medium": stage2_medium, "hard": 0},
            train_difficulties=("easy", "medium"),
        ),
        StageConfig(
            name="stage3_all",
            quota={"easy": stage3_easy, "medium": stage3_medium, "hard": stage3_hard},
            train_difficulties=("easy", "medium", "hard"),
        ),
    )


class ScenarioDataset:
    """Loads easy/medium/hard CSV scenario files and exposes split/difficulty ids."""

    def __init__(
        self,
        easy_csv: Path | str = DEFAULT_EASY_CSV,
        medium_csv: Path | str = DEFAULT_MEDIUM_CSV,
        hard_csv: Path | str = DEFAULT_HARD_CSV,
    ):
        self.csv_paths = {
            "easy": Path(easy_csv),
            "medium": Path(medium_csv),
            "hard": Path(hard_csv),
        }
        for difficulty, path in self.csv_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{difficulty} scenario CSV not found: {path}")

        self.L, self.PV, self.WT, self.metadata = self._load_csv_scenarios()
        self.index = self._build_index()

    @staticmethod
    def _normalize_profiles(l_raw: np.ndarray, pv_raw: np.ndarray, wt_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pv = np.asarray(pv_raw, dtype=np.float32) / 100.0
        wt = np.asarray(wt_raw, dtype=np.float32) / 100.0
        l = np.asarray(l_raw, dtype=np.float32) / 100.0
        l = np.clip(l, 0.0, None)
        peak_mask = l >= 0.8
        l[peak_mask] = np.minimum(l[peak_mask] * 1.12, 1.15)
        return l, pv, wt

    def _load_one_csv(self, path: Path, difficulty: str, start_rank: int, start_scenario_id: int):
        rows_by_day: Dict[int, List[Dict[str, str]]] = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                normalized_row = {str(key).strip(): str(value).strip() for key, value in row.items()}
                rows_by_day.setdefault(int(normalized_row["day_id"]), []).append(normalized_row)

        scenarios = []
        metadata: Dict[int, Dict[str, str]] = {}
        for local_rank, day_id in enumerate(rows_by_day.keys(), start=1):
            day_rows = sorted(rows_by_day[day_id], key=lambda item: int(item["time_step"]))
            if len(day_rows) != 96:
                raise ValueError(f"{path} day_id={day_id} has {len(day_rows)} rows, expected 96.")
            time_steps = [int(row["time_step"]) for row in day_rows]
            if time_steps != list(range(1, 97)):
                raise ValueError(f"{path} day_id={day_id} must contain time_step 1..96.")

            l_raw = np.asarray([float(row["L"]) for row in day_rows], dtype=np.float32)
            pv_raw = np.asarray([float(row["PV"]) for row in day_rows], dtype=np.float32)
            wt_raw = np.asarray([float(row["WT"]) for row in day_rows], dtype=np.float32)
            l, pv, wt = self._normalize_profiles(l_raw, pv_raw, wt_raw)
            scenarios.append((l, pv, wt))

            scenario_id = start_scenario_id + local_rank - 1
            global_rank = start_rank + local_rank - 1
            split = "test" if global_rank in TEST_RANKS_BY_DIFFICULTY[difficulty] else "train"
            metadata[scenario_id] = {
                "scenario_id": str(scenario_id),
                "source_day_id": str(day_id),
                "rank": str(global_rank),
                "local_rank": str(local_rank),
                "difficulty_label": difficulty,
                "split": split,
            }
        if len(scenarios) != 13:
            raise ValueError(f"{path} contains {len(scenarios)} days, expected 13.")
        return scenarios, metadata

    def _load_csv_scenarios(self):
        all_scenarios = []
        metadata: Dict[int, Dict[str, str]] = {}
        rank_start = {"easy": 1, "medium": 14, "hard": 27}
        scenario_start = {"easy": 0, "medium": 13, "hard": 26}
        for difficulty in DIFFICULTIES:
            scenarios, part_metadata = self._load_one_csv(
                self.csv_paths[difficulty],
                difficulty=difficulty,
                start_rank=rank_start[difficulty],
                start_scenario_id=scenario_start[difficulty],
            )
            all_scenarios.extend(scenarios)
            metadata.update(part_metadata)

        l = np.stack([scenario[0] for scenario in all_scenarios], axis=0)
        pv = np.stack([scenario[1] for scenario in all_scenarios], axis=0)
        wt = np.stack([scenario[2] for scenario in all_scenarios], axis=0)
        return l, pv, wt, metadata

    def _build_index(self) -> Dict[str, Dict[str, List[int]]]:
        index = {
            "train": {difficulty: [] for difficulty in DIFFICULTIES},
            "test": {difficulty: [] for difficulty in DIFFICULTIES},
        }
        for scenario_id, row in self.metadata.items():
            split = row.get("split")
            difficulty = row.get("difficulty_label")
            if split not in index:
                raise ValueError(f"Unsupported split={split!r} for scenario {scenario_id}")
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"Unsupported difficulty={difficulty!r} for scenario {scenario_id}")
            index[split][difficulty].append(scenario_id)
        for split in index:
            for difficulty in index[split]:
                index[split][difficulty].sort()
        return index

    def get_ids(self, split: str, difficulty: Optional[str] = None) -> List[int]:
        if split not in self.index:
            raise ValueError(f"split must be one of {sorted(self.index)}, got {split!r}")
        if difficulty is None:
            ids: List[int] = []
            for item in self.index[split].values():
                ids.extend(item)
            return sorted(ids)
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
        return list(self.index[split][difficulty])

    def get_train_pool(self, difficulties: Iterable[str]) -> Dict[str, List[int]]:
        return {difficulty: self.get_ids("train", difficulty) for difficulty in difficulties}

    def get_scenario(self, scenario_id: int) -> Dict[str, np.ndarray]:
        if scenario_id not in self.metadata:
            raise ValueError(f"Unknown scenario_id={scenario_id}")
        return {
            "L": self.L[scenario_id].copy(),
            "PV": self.PV[scenario_id].copy(),
            "WT": self.WT[scenario_id].copy(),
        }

    def get_metadata(self, scenario_id: int) -> Dict[str, str]:
        if scenario_id not in self.metadata:
            raise ValueError(f"Unknown scenario_id={scenario_id}")
        return dict(self.metadata[scenario_id])


class DifficultySampler:
    """Round-robin sampler with reshuffling to balance usage inside one difficulty."""

    def __init__(self, scenario_ids: Sequence[int], seed: int = 0):
        if not scenario_ids:
            raise ValueError("DifficultySampler requires at least one scenario id.")
        self.scenario_ids = list(scenario_ids)
        self.rng = np.random.default_rng(seed)
        self.pointer = 0
        self.order = self._new_order()

    def _new_order(self) -> List[int]:
        order = self.scenario_ids.copy()
        self.rng.shuffle(order)
        return order

    def sample(self) -> int:
        if self.pointer >= len(self.order):
            self.order = self._new_order()
            self.pointer = 0
        scenario_id = self.order[self.pointer]
        self.pointer += 1
        return int(scenario_id)


def build_stage_schedule(quota: Dict[str, int], seed: int = 0) -> List[str]:
    schedule: List[str] = []
    for difficulty in DIFFICULTIES:
        schedule.extend([difficulty] * int(quota.get(difficulty, 0)))
    rng = np.random.default_rng(seed)
    rng.shuffle(schedule)
    return schedule


def build_samplers(train_pool: Dict[str, Sequence[int]], seed: int = 0) -> Dict[str, DifficultySampler]:
    return {
        difficulty: DifficultySampler(ids, seed=seed + i)
        for i, (difficulty, ids) in enumerate(train_pool.items())
        if ids
    }


class Sparse33BusEnv(BatteryManagementEnv):
    """33-bus BESS environment backed by generated multi-day sparse-reward scenarios.

    It inherits the cached pandapower network from ``BatteryManagementEnv``:
    the 33-bus topology is built once, while every step only overwrites load,
    PV, WT, and BESS power injections. The class then replaces the single
    embedded PV/WT/L day with a selected generated scenario at reset time.
    """

    def __init__(
        self,
        easy_csv: Path | str = DEFAULT_EASY_CSV,
        medium_csv: Path | str = DEFAULT_MEDIUM_CSV,
        hard_csv: Path | str = DEFAULT_HARD_CSV,
        seed: int = 0,
        epsilon_soc_success: float = 0.03,
        avgvd_tolerance_ratio: float = 0.15,
        vvm_tolerance_ratio: float = 0.15,
        success_reference_by_difficulty: Optional[Dict[str, Dict[str, float]]] = None,
        _skip_success_reference: bool = False,
    ):
        self.dataset = ScenarioDataset(easy_csv=easy_csv, medium_csv=medium_csv, hard_csv=hard_csv)
        self.rng = np.random.default_rng(seed)
        self.env_seed = int(seed)
        self.train_pool: Dict[str, List[int]] = self.dataset.get_train_pool(DIFFICULTIES)
        self.samplers: Dict[str, DifficultySampler] = build_samplers(self.train_pool, seed=seed)
        self.current_scenario_id: Optional[int] = None
        self.current_split: str = "train"
        self.current_difficulty: Optional[str] = None
        self.episode_timesteps = 0
        self.metric_accumulator: Optional[EpisodeMetricAccumulator] = None
        self.uses_cached_powerflow = True
        self.epsilon_soc_success = float(epsilon_soc_success)
        self.avgvd_tolerance_ratio = float(avgvd_tolerance_ratio)
        self.vvm_tolerance_ratio = float(vvm_tolerance_ratio)
        self.success_reference_by_difficulty = success_reference_by_difficulty or {
            difficulty: {"avgvd": 1.0, "vvm": 1.0} for difficulty in DIFFICULTIES
        }
        super().__init__()
        if not _skip_success_reference and success_reference_by_difficulty is None:
            self.success_reference_by_difficulty = self._build_no_control_success_references(seed=seed)

    def _build_no_control_success_references(self, seed: int = 0) -> Dict[str, Dict[str, float]]:
        references: Dict[str, Dict[str, float]] = {}
        ref_env = Sparse33BusEnv(
            easy_csv=self.dataset.csv_paths["easy"],
            medium_csv=self.dataset.csv_paths["medium"],
            hard_csv=self.dataset.csv_paths["hard"],
            seed=seed + 7919,
            success_reference_by_difficulty={difficulty: {"avgvd": 1.0, "vvm": 1.0} for difficulty in DIFFICULTIES},
            _skip_success_reference=True,
        )
        for difficulty in DIFFICULTIES:
            avgvd_values: List[float] = []
            vvm_values: List[float] = []
            for scenario_id in ref_env.dataset.get_ids("test", difficulty):
                state, _ = ref_env.reset(scenario_id=scenario_id, split="test", difficulty=difficulty)
                done = False
                while not done:
                    state, _, terminated, truncated, _ = ref_env.step(np.array([0.0], dtype=np.float32))
                    done = bool(terminated or truncated)
                metrics = ref_env.get_episode_metrics()
                avgvd_values.append(float(metrics.get("avg_voltage_deviation", 0.0)))
                vvm_values.append(float(metrics.get("voltage_deviation_max_pu", 0.0)))
            references[difficulty] = {
                "avgvd": float(np.mean(avgvd_values)) if avgvd_values else 1.0,
                "vvm": float(np.mean(vvm_values)) if vvm_values else 1.0,
            }
        return references

    def clear_train_pool(self) -> None:
        self.train_pool = {difficulty: [] for difficulty in DIFFICULTIES}
        self.samplers = {}

    def set_train_pool(self, train_pool: Dict[str, Sequence[int]], seed: Optional[int] = None) -> None:
        normalized_pool: Dict[str, List[int]] = {difficulty: [] for difficulty in DIFFICULTIES}
        for difficulty, ids in train_pool.items():
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"Unsupported difficulty={difficulty!r}")
            for scenario_id in ids:
                meta = self.dataset.get_metadata(int(scenario_id))
                if meta["split"] != "train":
                    raise ValueError(f"Scenario {scenario_id} is split={meta['split']}; train_pool only accepts train scenarios.")
                if meta["difficulty_label"] != difficulty:
                    raise ValueError(
                        f"Scenario {scenario_id} is difficulty={meta['difficulty_label']}, cannot add to {difficulty} pool."
                    )
                normalized_pool[difficulty].append(int(scenario_id))
        self.train_pool = normalized_pool
        self.samplers = build_samplers(self.train_pool, seed=self.env_seed if seed is None else int(seed))

    def set_stage(self, train_difficulties: Sequence[str], seed: Optional[int] = None) -> None:
        self.clear_train_pool()
        self.set_train_pool(self.dataset.get_train_pool(train_difficulties), seed=seed)

    def sample_train_scenario(self, difficulty: Optional[str] = None) -> int:
        if difficulty is not None:
            if difficulty not in self.samplers:
                raise ValueError(f"No train scenarios available for difficulty={difficulty!r}")
            return self.samplers[difficulty].sample()

        available = [item for item in self.samplers.items() if item[1].scenario_ids]
        if not available:
            raise ValueError("Train pool is empty. Call set_train_pool() or set_stage() first.")
        selected_difficulty, sampler = available[int(self.rng.integers(0, len(available)))]
        self.current_difficulty = selected_difficulty
        return sampler.sample()

    def sample_test_scenario(self, difficulty: Optional[str] = None) -> int:
        ids = self.dataset.get_ids("test", difficulty)
        if not ids:
            raise ValueError(f"No test scenarios available for difficulty={difficulty!r}")
        return int(ids[int(self.rng.integers(0, len(ids)))])

    def _activate_scenario(self, scenario_id: int) -> None:
        scenario = self.dataset.get_scenario(int(scenario_id))
        self.L = scenario["L"]
        self.PV = scenario["PV"]
        self.WT = scenario["WT"]
        metadata = self.dataset.get_metadata(int(scenario_id))
        self.current_scenario_id = int(scenario_id)
        self.current_split = metadata["split"]
        self.current_difficulty = metadata["difficulty_label"]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, object]] = None,
        scenario_id: Optional[int] = None,
        split: Optional[str] = None,
        difficulty: Optional[str] = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        scenario_id = int(options.get("scenario_id", scenario_id)) if options.get("scenario_id", scenario_id) is not None else None
        split = str(options.get("split", split or "train"))
        difficulty_option = options.get("difficulty", difficulty)
        difficulty = str(difficulty_option) if difficulty_option is not None else None

        if scenario_id is None:
            if split == "train":
                scenario_id = self.sample_train_scenario(difficulty)
            elif split == "test":
                scenario_id = self.sample_test_scenario(difficulty)
            else:
                raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        else:
            metadata = self.dataset.get_metadata(scenario_id)
            if metadata["split"] != split:
                raise ValueError(f"Scenario {scenario_id} belongs to split={metadata['split']}, requested split={split}")
            if difficulty is not None and metadata["difficulty_label"] != difficulty:
                raise ValueError(
                    f"Scenario {scenario_id} belongs to difficulty={metadata['difficulty_label']}, requested {difficulty}"
                )

        self._activate_scenario(scenario_id)
        state, info = super().reset(seed=seed, options=options)
        self.episode_timesteps = 0
        self.metric_accumulator = EpisodeMetricAccumulator(
            soc_target=self.target_soc,
            num_buses=self.num_buses,
            epsilon_soc_success=self.epsilon_soc_success,
            avgvd_tolerance_ratio=self.avgvd_tolerance_ratio,
            vvm_tolerance_ratio=self.vvm_tolerance_ratio,
            no_control_avgvd=float(self.success_reference_by_difficulty[self.current_difficulty]["avgvd"]),
            no_control_vvm=float(self.success_reference_by_difficulty[self.current_difficulty]["vvm"]),
        )
        self.metric_accumulator.reset(float(state[1]))
        info.update(
            {
                "scenario_id": int(self.current_scenario_id),
                "split": self.current_split,
                "difficulty": self.current_difficulty,
            }
        )
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        action_value = float(info.get("applied_bess_action", np.squeeze(action)))
        self.episode_timesteps += 1
        if self.metric_accumulator is not None:
            self.metric_accumulator.update(
                train_reward=float(reward),
                original_reward=float(reward),
                info=info,
                action=action_value,
            )

        info.update(
            {
                "scenario_id": int(self.current_scenario_id) if self.current_scenario_id is not None else -1,
                "split": self.current_split,
                "difficulty": self.current_difficulty,
                "original_reward": float(reward),
            }
        )
        return state, reward, terminated, truncated, info

    def get_episode_metrics(self) -> Dict[str, float | int | str]:
        if self.metric_accumulator is None:
            return {}
        metrics = self.metric_accumulator.compute()
        metrics["scenario_id"] = int(self.current_scenario_id) if self.current_scenario_id is not None else -1
        metrics["difficulty"] = str(self.current_difficulty)
        metrics["split"] = str(self.current_split)
        metrics["steps"] = int(self.episode_timesteps)
        return metrics


BatteryManagementSparse33BusEnv = Sparse33BusEnv


class SparseHardcodedDayEnv(BatteryManagementEnv):
    """Sparse-module wrapper for the single hard-coded day in ``new_env_ES``.

    This keeps the same physical model and embedded L/PV/WT profiles as
    ``BatteryManagementEnv`` while allowing experiments to explicitly request a
    sparse_33bus environment class.
    """

    pass


__all__ = [
    "BatteryManagementSparse33BusEnv",
    "DEFAULT_EASY_CSV",
    "DEFAULT_HARD_CSV",
    "DEFAULT_MEDIUM_CSV",
    "DEFAULT_THREE_STAGE_CURRICULUM",
    "DIFFICULTIES",
    "DifficultySampler",
    "ScenarioDataset",
    "SparseHardcodedDayEnv",
    "Sparse33BusEnv",
    "StageConfig",
    "SUPPORTED_RARR_RESIDUAL_METHODS",
    "SUPPORTED_THREE_STAGE_POLICIES",
    "TEST_RANKS_BY_DIFFICULTY",
    "build_balanced_three_stage_curriculum",
    "build_samplers",
    "build_stage_schedule",
]

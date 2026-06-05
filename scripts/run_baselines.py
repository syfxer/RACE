import argparse
import os
import sys

if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _configure_runtime_dirs():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runtime_root = os.path.join(repo_root, ".runtime_cache")
    numba_cache_dir = os.path.join(runtime_root, "numba")
    temp_dir = os.path.join(runtime_root, "tmp")

    os.makedirs(numba_cache_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    os.environ["NUMBA_CACHE_DIR"] = numba_cache_dir
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir

    try:
        from numba.core import config as numba_config

        numba_config.CACHE_DIR = numba_cache_dir
    except Exception:
        pass


_configure_runtime_dirs()

from race.baselines.electric_baselines import (
    AVAILABLE_BASELINES,
    DIFFICULTIES,
    build_baseline_controller,
    build_baseline_file_name,
    sanitize_name,
    evaluate_baseline_on_sparse_tests,
)
from race.utils.metric_logger import EVAL_METRIC_COLUMNS, ElectricMetricLogger, build_electric_metric_path, get_last_logged_step


DEFAULT_BASELINES = ["NoControl", "RuleBasedBESS", "StandardMPCPersistence"]


def parse_baselines(raw_value):
    if raw_value.lower() == "all":
        return list(DEFAULT_BASELINES)
    requested = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in AVAILABLE_BASELINES]
    if unknown:
        raise ValueError(f"Unknown baselines: {', '.join(unknown)}")
    return requested


def find_next_available_seed(policy, env_name, rd_method, model_name, result_dir, max_timesteps, start_seed=0):
    seed = start_seed
    while True:
        file_name = build_baseline_file_name(policy, env_name, rd_method, model_name, seed)
        result_path = build_electric_metric_path(result_dir, file_name)
        last_step = get_last_logged_step(result_path)
        if last_step is None or last_step < max_timesteps:
            return seed
        seed += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BatteryManagementEnv")
    parser.add_argument("--baselines", default="all")
    parser.add_argument("--max_timesteps", default=4800, type=int)
    parser.add_argument("--eval_freq", default=96, type=int)
    parser.add_argument("--eval_episodes", default=2, type=int)
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--rd_method", default="Baseline")
    parser.add_argument("--model_name", default="electric-baseline")
    parser.add_argument("--start_seed", default=0, type=int)
    parser.add_argument("--include_initial_eval", action="store_true")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--mpc_horizon", default=12, type=int)
    parser.add_argument("--mpc_action_grid_size", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rule_grid_gain", default=1.25, type=float)
    parser.add_argument("--rule_soc_weight", default=0.35, type=float)
    parser.add_argument("--rule_endgame_soc_weight", default=0.85, type=float)
    parser.add_argument("--rule_endgame_steps", default=12, type=int)
    args = parser.parse_args()

    baseline_names = parse_baselines(args.baselines)
    model_name = sanitize_name(args.model_name)

    for baseline_name in baseline_names:
        if args.overwrite_existing:
            seed = args.start_seed
        else:
            seed = find_next_available_seed(
                baseline_name,
                args.env,
                args.rd_method,
                model_name,
                args.result_dir,
                args.max_timesteps,
                start_seed=args.start_seed,
            )

        file_name = build_baseline_file_name(
            baseline_name,
            args.env,
            args.rd_method,
            model_name,
            seed,
        )
        logger = ElectricMetricLogger(args.result_dir, file_name, fieldnames=EVAL_METRIC_COLUMNS, overwrite=True)

        baseline_kwargs = {}
        if baseline_name in {"MPCPersistence", "StandardMPC", "StandardMPCPersistence"}:
            baseline_kwargs = {
                "horizon": args.mpc_horizon,
            }
        elif baseline_name == "RuleBasedBESS":
            baseline_kwargs = {
                "grid_gain": args.rule_grid_gain,
                "soc_weight": args.rule_soc_weight,
                "endgame_soc_weight": args.rule_endgame_soc_weight,
                "endgame_steps": args.rule_endgame_steps,
            }

        controller = build_baseline_controller(baseline_name, **baseline_kwargs)
        summaries = evaluate_baseline_on_sparse_tests(controller, seed=seed)

        for difficulty in DIFFICULTIES:
            summary = summaries[difficulty]
            logger.append_eval(
                global_episode=args.max_timesteps,
                stage="baseline_test",
                stage_episode=0,
                test_difficulty=difficulty,
                metrics=summary,
            )

        print(
            f"[baseline] {baseline_name} seed={seed} saved to {logger.path} | "
            f"easy_R={summaries['easy']['original_reward']:.3f}, "
            f"medium_R={summaries['medium']['original_reward']:.3f}, "
            f"hard_R={summaries['hard']['original_reward']:.3f}, "
            f"all_R={summaries['all']['original_reward']:.3f}"
        )


if __name__ == "__main__":
    main()

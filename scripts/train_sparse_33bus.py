from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import random
import sys
import traceback
from types import SimpleNamespace
from typing import Dict, List, Mapping

import numpy as np
import torch
import wandb


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import race.algos.ddpg as DDPG
import race.algos.ppo as PPO
import race.algos.sac as SAC
import race.algos.td3 as TD3
from race.training import replay_buffer as utils
from race.utils.metric_logger import (
    EVAL_METRIC_COLUMNS,
    TRAIN_EPISODE_METRIC_COLUMNS,
    ElectricMetricLogger,
)
from race.utils.metrics import aggregate_metrics, build_wandb_eval_metrics, compute_episodes_to_convergence
from race.envs.sparse_33bus_env import (
    DIFFICULTIES,
    SUPPORTED_THREE_STAGE_POLICIES,
    Sparse33BusEnv,
    build_balanced_three_stage_curriculum,
    build_stage_schedule,
)
from race.utils.wandb_utils import finalize_wandb_run


def normalize_rd_method(rd_method: str) -> str:
    aliases = {
        "RARR-SAC": "RARR",
        "RARR_SAC": "RARR",
        "RARR-SAC-VIB": "RARR_VIB",
        "RARR_SAC_VIB": "RARR_VIB",
        "RARR-VIB-SAC": "RARR_VIB",
        "RARR_VIB_SAC": "RARR_VIB",
        "rarr_RD": "RARR_RD",
        "RARR-SAC-RD": "RARR_RD",
        "RARR_SAC_RD": "RARR_RD",
        "rarr_RRD": "RARR_RRD",
        "RARR-SAC-RRD": "RARR_RRD",
        "RARR_SAC_RRD": "RARR_RRD",
        "rarr_Diaster": "RARR_Diaster",
        "RARR-SAC-Diaster": "RARR_Diaster",
        "RARR_SAC_Diaster": "RARR_Diaster",
        "SAC_VIB": "VIB",
        "SAC_Diaster": "Diaster",
        "SAC_RD": "RD",
        "SAC_RRD": "RRD",
    }
    return aliases.get(str(rd_method), str(rd_method))


def uses_rarr_reward(rd_method: str) -> bool:
    return rd_method in {"RARR", "RARR_VIB", "RARR_RD", "RARR_RRD", "RARR_Diaster"}


def uses_rarr_residual(rd_method: str) -> bool:
    return rd_method in {"RARR_VIB", "RARR_RD", "RARR_RRD", "RARR_Diaster"}


def get_result_model_tag(rd_method: str, dense_r: bool = False) -> str:
    if rd_method == "None" and dense_r:
        return "dense-reward"
    return "nonllm"


def build_file_name(policy: str, env_name: str, rd_method: str, seed: int, dense_r: bool = False) -> str:
    return f"{policy}_{env_name}_{rd_method}_{get_result_model_tag(rd_method, dense_r=dense_r)}_seed{seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_algo_args(args: argparse.Namespace, env: Sparse33BusEnv) -> SimpleNamespace:
    return SimpleNamespace(
        policy=args.policy,
        env=args.env,
        seed=args.seed,
        start_timesteps=args.start_timesteps,
        reward_train_start=args.reward_train_start,
        eval_freq=args.eval_freq_episodes,
        max_timesteps=args.episodes * 96,
        eval_episodes=0,
        expl_noise=args.expl_noise,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        adaptive_alpha=True,
        discount=args.discount,
        tau=args.tau,
        target_update_freq=args.target_update_freq,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        dense_r=args.dense_r,
        rrd_unbiased=("unbiased" in args.rd_method),
        rd_method=args.rd_method,
        exp_name=args.exp_name,
        run_name=args.run_name,
        rrd_k=args.rrd_k,
        direct_generate=False,
        IB_latent_dim=args.IB_latent_dim,
        IB_kl_weight=args.IB_kl_weight,
        rarr_lambda_soc=args.rarr_lambda_soc,
        lambda_soc=args.lambda_soc,
        soc_target=env.target_soc,
        soc_deadband=args.soc_deadband,
        soc_deadband_mode=args.soc_deadband_mode,
        lambda_rarr=args.lambda_rarr,
        rarr_mode=args.rarr_mode,
        rarr_voltage_weight=args.rarr_voltage_weight,
        rarr_use_gamma=args.rarr_use_gamma,
        rarr_penalty_power=args.rarr_penalty_power,
        rarr_residual_eta=args.rarr_residual_eta,
        rarr_vib_eta=args.rarr_vib_eta,
        rarr_vib_beta=args.rarr_vib_beta,
        rarr_vib_latent_dim=args.rarr_vib_latent_dim,
        rarr_vib_hidden_dim=args.rarr_vib_hidden_dim,
        rarr_vib_lr=args.rarr_vib_lr,
        rarr_vib_reward_clip=args.rarr_vib_reward_clip,
        obs_dim=env.single_observation_space.shape[0],
        action_dim=env.single_action_space.shape[0],
        max_length=96,
        max_action=float(env.single_action_space.high[0]),
    )


def make_rarr_rewarder(args: argparse.Namespace, env: Sparse33BusEnv):
    if not uses_rarr_reward(args.rd_method):
        return None
    from race.rewards import RARRReward, RARRRewardConfig

    delta_soc_max = env.max_bess_power_mw * env.slice / env.Battery_Capacity
    return RARRReward(
        RARRRewardConfig(
            T=96,
            gamma=args.discount,
            lambda_soc=args.rarr_lambda_soc,
            soc_target=env.target_soc,
            soc_deadband=args.soc_deadband,
            soc_min=env.soc_min,
            soc_max=env.soc_max,
            delta_soc_ch_max=delta_soc_max,
            delta_soc_dis_max=delta_soc_max,
            voltage_weight=args.rarr_voltage_weight,
            use_gamma=args.rarr_use_gamma,
            penalty_power=args.rarr_penalty_power,
            lambda_rarr=args.lambda_rarr,
            rarr_mode=args.rarr_mode,
        )
    )


def configure_reward_deadband(env: Sparse33BusEnv, args: argparse.Namespace) -> None:
    env.soc_deadband = float(args.soc_deadband)
    env.lambda_soc = float(args.lambda_soc)
    env.soc_deadband_mode = str(args.soc_deadband_mode)


def evaluate_policy(policy, env: Sparse33BusEnv) -> Dict[str, Mapping[str, float]]:
    difficulty_metrics = {}
    for difficulty in DIFFICULTIES:
        episode_metrics = []
        for scenario_id in env.dataset.get_ids("test", difficulty):
            state, _ = env.reset(scenario_id=scenario_id, split="test", difficulty=difficulty)
            done = False
            while not done:
                action = policy.select_action(np.asarray(state), deterministic=True)
                state, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
            episode_metrics.append(env.get_episode_metrics())
        difficulty_metrics[difficulty] = aggregate_metrics(episode_metrics)
    return difficulty_metrics


def build_policy(args: argparse.Namespace, algo_args: SimpleNamespace, state_dim: int, action_dim: int, max_action: float):
    kwargs = {
        "args": algo_args,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    if args.policy == "TD3":
        return TD3.TD3(
            **kwargs,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
        )
    if args.policy == "DDPG":
        return DDPG.DDPG(**kwargs)
    if args.policy == "SAC":
        return SAC.SAC(**kwargs)
    if args.policy == "PPO":
        return PPO.PPO(**kwargs)
    raise ValueError(f"Policy not recognized: {args.policy}")


def _mean_metric(difficulty_metrics: Mapping[str, Mapping[str, float]], key: str) -> float:
    values = [
        float(metrics[key])
        for metrics in difficulty_metrics.values()
        if key in metrics
    ]
    return float(np.mean(values)) if values else 0.0


def print_eval_summary(
    global_episode: int,
    stage_name: str,
    stage_episode: int,
    difficulty_metrics: Mapping[str, Mapping[str, float]],
) -> None:
    print("---------------------------------------", flush=True)
    print(
        f"Evaluation episode={global_episode} stage={stage_name} "
        f"stage_episode={stage_episode}",
        flush=True,
    )
    for difficulty in DIFFICULTIES:
        metrics = difficulty_metrics.get(difficulty, {})
        print(
            f"  {difficulty}: reward={float(metrics.get('original_reward', 0.0)):.3f}, "
            f"AvgVD={float(metrics.get('avg_voltage_deviation', 0.0)):.5f} pu, "
            f"MaxVD={float(metrics.get('voltage_deviation_max_pu', 0.0)):.5f} pu, "
            f"VVR={float(metrics.get('voltage_violation_rate', 0.0)):.5f}, "
            f"SOCerr={float(metrics.get('terminal_soc_error', 0.0)):.5f}, "
            f"SR={float(metrics.get('success_rate', 0.0)):.3f}, "
            f"SOCSR={float(metrics.get('soc_success_rate', 0.0)):.3f}, "
            f"VSR={float(metrics.get('voltage_success_rate', 0.0)):.3f}",
            flush=True,
        )
    print(
        f"  all: reward={_mean_metric(difficulty_metrics, 'original_reward'):.3f}, "
        f"AvgVD={_mean_metric(difficulty_metrics, 'avg_voltage_deviation'):.5f} pu, "
        f"MaxVD={_mean_metric(difficulty_metrics, 'voltage_deviation_max_pu'):.5f} pu, "
        f"VVR={_mean_metric(difficulty_metrics, 'voltage_violation_rate'):.5f}, "
        f"SOCerr={_mean_metric(difficulty_metrics, 'terminal_soc_error'):.5f}, "
        f"SR={_mean_metric(difficulty_metrics, 'success_rate'):.3f}, "
        f"SOCSR={_mean_metric(difficulty_metrics, 'soc_success_rate'):.3f}, "
        f"VSR={_mean_metric(difficulty_metrics, 'voltage_success_rate'):.3f}",
        flush=True,
    )
    print("---------------------------------------", flush=True)


def log_wandb(data: Mapping[str, float], global_episode: int) -> None:
    if wandb.run is None:
        return
    payload = dict(data)
    payload["global_episode"] = int(global_episode)
    try:
        wandb.log(payload, step=int(global_episode))
    except Exception as exc:
        print(f"[wandb] log skipped after failure: {exc}", flush=True)


def write_convergence_result(
    result_dir: str,
    file_name: str,
    run_name: str,
    rd_method: str,
    seed: int,
    episodes_to_convergence,
) -> str:
    convergence_path = os.path.join(result_dir, "electric_metric", f"{file_name}_convergence.csv")
    os.makedirs(os.path.dirname(convergence_path), exist_ok=True)
    if isinstance(episodes_to_convergence, Mapping):
        convergence_row = {
            "episodes_to_convergence": episodes_to_convergence.get("all"),
            "easy_episodes_to_convergence": episodes_to_convergence.get("easy"),
            "medium_episodes_to_convergence": episodes_to_convergence.get("medium"),
            "hard_episodes_to_convergence": episodes_to_convergence.get("hard"),
        }
    else:
        convergence_row = {
            "episodes_to_convergence": episodes_to_convergence,
            "easy_episodes_to_convergence": "",
            "medium_episodes_to_convergence": "",
            "hard_episodes_to_convergence": "",
        }
    with open(convergence_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "rd_method",
                "seed",
                "episodes_to_convergence",
                "easy_episodes_to_convergence",
                "medium_episodes_to_convergence",
                "hard_episodes_to_convergence",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_name": run_name,
                "rd_method": rd_method,
                "seed": int(seed),
                **convergence_row,
            }
        )
    return convergence_path


def compute_convergence_by_difficulty(eval_history: List[Mapping[str, float]], max_episode: int) -> Dict[str, object]:
    convergence = {}
    for difficulty in [*DIFFICULTIES, "all"]:
        convergence[difficulty] = compute_episodes_to_convergence(
            eval_history,
            metric_reward=f"eval/{difficulty}/original_reward",
            metric_avgvd=f"eval/{difficulty}/avg_voltage_deviation",
            metric_soc=f"eval/{difficulty}/terminal_soc_error",
            max_episode=max_episode,
            window_size=3,
            consecutive_windows=2,
            reward_rel_threshold=0.05,
            avgvd_rel_threshold=0.01,
            soc_abs_threshold=0.01,
            soc_target_threshold=0.05,
        )
    return convergence


def train_one_episode(
    env: Sparse33BusEnv,
    policy,
    replay_buffer: utils.ReplayBuffer,
    old_replay_buffer,
    algo_args: SimpleNamespace,
    rarr_rewarder,
    global_step: int,
    total_steps: int,
    difficulty: str,
) -> tuple[int, Mapping[str, float], Dict[str, float]]:
    state, _ = env.reset(split="train", difficulty=difficulty)
    done = False
    episode_reward = 0.0
    episode_steps = 0
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    rewards: List[List[float]] = []
    dense_rewards: List[List[float]] = []
    dones: List[List[float]] = []
    a_log_probs: List[np.ndarray] = []
    loss_log: Dict[str, float] = {}
    rarr_episode = {
        "voltage_reward_sum": 0.0,
        "terminal_soc_penalty": 0.0,
        "raw_terminal_soc_error": 0.0,
        "deadband_soc_violation": 0.0,
        "rarr_reward_sum": 0.0,
        "rarr_voltage_sum": 0.0,
        "rarr_soc_credit_sum": 0.0,
        "rarr_d_max": 0.0,
        "rarr_d_final": 0.0,
        "rarr_phi_final": 0.0,
        "recoverability_violation_sum": 0.0,
        "recoverability_violation_rate": 0.0,
        "recoverability_violation_count": 0.0,
        "recoverable_low_min": 0.0,
        "recoverable_low_max": 0.0,
        "recoverable_high_min": 0.0,
        "recoverable_high_max": 0.0,
    }
    recoverable_lows: List[float] = []
    recoverable_highs: List[float] = []

    while not done:
        log_prob = np.zeros(env.single_action_space.shape[0], dtype=np.float32)
        if total_steps < algo_args.start_timesteps and algo_args.policy != "PPO":
            action = env.single_action_space.sample()
        elif algo_args.policy == "TD3":
            action = (
                policy.select_action(np.asarray(state))
                + np.random.normal(0, policy.max_action * algo_args.expl_noise, size=env.single_action_space.shape[0])
            ).clip(-policy.max_action, policy.max_action)
        elif algo_args.policy == "PPO":
            action, log_prob = policy.select_action(np.asarray(state), log_prob=True)
            a_log_probs.append(np.asarray(log_prob, dtype=np.float32))
        else:
            action = policy.select_action(np.asarray(state))

        env_t_before_step = env.t
        next_state, env_reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        episode_steps += 1
        terminal_soc_penalty = float(info.get("terminal_soc_penalty", 0.0))
        rarr_episode["voltage_reward_sum"] += float(info.get("voltage_reward", -float(info.get("quality", 0.0))))
        rarr_episode["terminal_soc_penalty"] += terminal_soc_penalty
        rarr_episode["raw_terminal_soc_error"] = float(info.get("raw_terminal_soc_error", 0.0))
        rarr_episode["deadband_soc_violation"] = float(info.get("deadband_soc_violation", 0.0))

        sparse_episode_reward = 0.0 if not done else episode_reward + float(env_reward)
        train_replay_reward = sparse_episode_reward
        if rarr_rewarder is not None:
            rarr_info = rarr_rewarder.compute(
                t=env_t_before_step,
                soc_t=float(state[1]),
                soc_next=float(next_state[1]),
                voltage_cost_t=float(info.get("quality", 0.0)),
            )
            train_replay_reward = float(rarr_info["reward"])
            if done and not uses_rarr_residual(algo_args.rd_method):
                train_replay_reward += terminal_soc_penalty
            rarr_episode["rarr_reward_sum"] += float(rarr_info["r_rarr"])
            rarr_episode["rarr_voltage_sum"] += float(rarr_info["r_voltage"])
            rarr_episode["rarr_soc_credit_sum"] += float(rarr_info["r_rarr"])
            rarr_episode["rarr_d_max"] = max(rarr_episode["rarr_d_max"], float(rarr_info["d_next"]))
            rarr_episode["rarr_d_final"] = float(rarr_info["d_next"])
            rarr_episode["rarr_phi_final"] = float(rarr_info["phi_next"])
            d_next = float(rarr_info["d_next"])
            rarr_episode["recoverability_violation_sum"] += d_next
            rarr_episode["recoverability_violation_count"] += float(d_next > 0.0)
            recoverable_lows.append(float(rarr_info["lower_next"]))
            recoverable_highs.append(float(rarr_info["upper_next"]))

        done_flag = float(done)
        states.append(np.asarray(state, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        next_states.append(np.asarray(next_state, dtype=np.float32))
        rewards.append([train_replay_reward])
        dense_rewards.append([float(env_reward)])
        dones.append([done_flag])
        if old_replay_buffer is not None:
            old_replay_buffer.add(
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                np.asarray(log_prob, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32),
                train_replay_reward,
                float(env_reward),
                done_flag,
                float(done),
            )

        episode_reward += float(env_reward)
        state = next_state
        total_steps += 1

        if total_steps >= algo_args.start_timesteps:
            if algo_args.policy == "PPO":
                total_loss_log = {}
                if replay_buffer.size >= algo_args.reward_train_start:
                    reward_loss = policy.train_reward(replay_buffer, algo_args.batch_size)
                    if isinstance(reward_loss, dict):
                        total_loss_log.update(reward_loss)
                    else:
                        total_loss_log["reward_model_loss"] = reward_loss
                if old_replay_buffer is not None and old_replay_buffer.size >= algo_args.batch_size:
                    ppo_loss_log = policy.train(
                        old_replay_buffer,
                        replay_buffer,
                        algo_args.batch_size,
                        total_steps=total_steps,
                    )
                    old_replay_buffer.size = 0
                    total_loss_log.update(ppo_loss_log)
                if total_loss_log:
                    loss_log = total_loss_log
            elif replay_buffer.size >= algo_args.batch_size:
                loss_log = policy.train(replay_buffer, algo_args.batch_size)

    replay_buffer.add_traj(
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(dense_rewards, dtype=np.float32),
        np.asarray(dones, dtype=np.float32),
        episode_reward,
        episode_steps,
        np.asarray(a_log_probs, dtype=np.float32) if a_log_probs else None,
    )
    metrics = env.get_episode_metrics()
    metrics["train_reward"] = float(np.sum(np.asarray(rewards, dtype=float)))
    metrics["original_reward"] = float(episode_reward)
    if episode_steps > 0:
        rarr_episode["recoverability_violation_rate"] = (
            float(rarr_episode["recoverability_violation_count"]) / float(episode_steps)
        )
    if recoverable_lows:
        rarr_episode["recoverable_low_min"] = float(np.min(recoverable_lows))
        rarr_episode["recoverable_low_max"] = float(np.max(recoverable_lows))
    if recoverable_highs:
        rarr_episode["recoverable_high_min"] = float(np.min(recoverable_highs))
        rarr_episode["recoverable_high_max"] = float(np.max(recoverable_highs))
    metrics.update(rarr_episode)
    return total_steps, metrics, loss_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse 33-bus three-stage training.")
    parser.add_argument("--policy", default="SAC", choices=SUPPORTED_THREE_STAGE_POLICIES)
    parser.add_argument("--env", default="Sparse33BusEnv")
    parser.add_argument("--rd_method", default="RARR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--eval_freq_episodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--start_timesteps", type=int, default=960)
    parser.add_argument("--reward_train_start", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target_update_freq", type=int, default=2)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--dense_r", action="store_true")
    parser.add_argument("--rrd_k", type=int, default=64)
    parser.add_argument("--IB_latent_dim", type=int, default=8)
    parser.add_argument("--IB_kl_weight", type=float, default=5e-3)
    parser.add_argument("--soc_deadband", type=float, default=0.02)
    parser.add_argument("--lambda_soc", type=float, default=100.0)
    parser.add_argument("--soc_deadband_mode", default="linear", choices=["linear", "squared"])
    parser.add_argument("--lambda_rarr", type=float, default=100.0)
    parser.add_argument("--rarr_mode", default="linear", choices=["linear", "squared"])
    parser.add_argument("--rarr_lambda_soc", type=float, default=100.0)
    parser.add_argument("--rarr_voltage_weight", type=float, default=1.0)
    parser.add_argument("--rarr_use_gamma", action="store_true", default=True)
    parser.add_argument("--rarr_penalty_power", type=float, default=1.0)
    parser.add_argument("--rarr_residual_eta", type=float, default=1.0)
    parser.add_argument("--rarr_vib_eta", type=float, default=1.0)
    parser.add_argument("--rarr_vib_beta", type=float, default=1e-3)
    parser.add_argument("--rarr_vib_latent_dim", type=int, default=8)
    parser.add_argument("--rarr_vib_hidden_dim", type=int, default=128)
    parser.add_argument("--rarr_vib_lr", type=float, default=3e-4)
    parser.add_argument("--rarr_vib_reward_clip", type=float, default=None)
    parser.add_argument("--result_dir", default=os.path.join(PROJECT_ROOT, "result", "sparse_33bus_main"))
    parser.add_argument("--model_dir", default=os.path.join(PROJECT_ROOT, "train_model", "sparse_33bus_main"))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_dir", default=os.path.join(PROJECT_ROOT, "wandb_result"))
    parser.add_argument("--wandb_project", default="sparse_33bus_main")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="three_stage_seed0")
    parser.add_argument("--wandb_tags", default="sparse_33bus,three_stage")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--gpu_gate", action="store_true", default=True)
    parser.add_argument("--no_gpu_gate", dest="gpu_gate", action="store_false")
    parser.add_argument("--gpu_gate_memory_threshold_mb", type=int, default=512)
    parser.add_argument("--gpu_gate_poll_seconds", type=int, default=20)
    parser.add_argument("--gpu_gate_stale_seconds", type=int, default=24 * 3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.rd_method = normalize_rd_method(args.rd_method)
    set_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.wandb_dir, exist_ok=True)

    gate = None
    if args.gpu_gate and torch.cuda.is_available():
        from experiment.gpu_gate import GpuFifoGate

        gate = GpuFifoGate(
            root=os.path.join(PROJECT_ROOT, ".runtime_cache", "gpu_gate"),
            memory_threshold_mb=args.gpu_gate_memory_threshold_mb,
            stale_seconds=args.gpu_gate_stale_seconds,
            poll_seconds=args.gpu_gate_poll_seconds,
        )
        gate.acquire()

    env = Sparse33BusEnv(seed=args.seed)
    eval_env = Sparse33BusEnv(seed=args.seed + 1000)
    configure_reward_deadband(env, args)
    configure_reward_deadband(eval_env, args)
    algo_args = build_algo_args(args, env)
    if not args.run_name:
        args.run_name = f"{args.policy}_{args.rd_method}_sparse_33bus_seed{args.seed}_ep{args.episodes}"
        algo_args.run_name = args.run_name
    file_name = build_file_name(args.policy, args.env, args.rd_method, args.seed, dense_r=args.dense_r)

    if args.wandb_mode == "online":
        try:
            wandb.login(relogin=False)
        except Exception as exc:
            print(f"[wandb] login failed, fallback to offline: {exc}")
            args.wandb_mode = "offline"

    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            config=vars(args),
            name=args.run_name,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] or None,
            save_code=True,
        )
    except Exception as exc:
        print(f"[wandb] init failed, continue with disabled logging: {exc}", flush=True)
        try:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=args.run_name,
                mode="disabled",
                dir=args.wandb_dir,
            )
        except Exception as disabled_exc:
            print(f"[wandb] disabled init also failed, continue without wandb: {disabled_exc}", flush=True)

    if wandb.run is not None:
        try:
            wandb.define_metric("global_episode")
            wandb.define_metric("train/*", step_metric="global_episode")
            wandb.define_metric("eval/*", step_metric="global_episode")
            wandb.define_metric("loss/*", step_metric="global_episode")
        except Exception as exc:
            print(f"[wandb] metric definition skipped after failure: {exc}", flush=True)

    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]
    max_action = float(env.single_action_space.high[0])
    policy = build_policy(args, algo_args, state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    if gate is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gate.release()
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, 96, args.buffer_size, log_prob=(args.policy == "PPO"))
    old_replay_buffer_size = max(args.batch_size + 1, args.start_timesteps + 96 + 1)
    old_replay_buffer = PPO.ReplayBuffer(state_dim, action_dim, old_replay_buffer_size) if args.policy == "PPO" else None
    rarr_rewarder = make_rarr_rewarder(args, env)

    train_logger = ElectricMetricLogger(
        args.result_dir,
        f"{file_name}_train",
        fieldnames=TRAIN_EPISODE_METRIC_COLUMNS + [
            "voltage_reward_sum",
            "terminal_soc_penalty",
            "raw_terminal_soc_error",
            "deadband_soc_violation",
            "rarr_reward_sum",
            "rarr_voltage_sum",
            "rarr_soc_credit_sum",
            "rarr_d_max",
            "rarr_d_final",
            "rarr_phi_final",
            "recoverability_violation_sum",
            "recoverability_violation_rate",
            "recoverability_violation_count",
            "recoverable_low_min",
            "recoverable_low_max",
            "recoverable_high_min",
            "recoverable_high_max",
        ],
        overwrite=True,
    )
    eval_logger = ElectricMetricLogger(
        args.result_dir,
        file_name,
        fieldnames=EVAL_METRIC_COLUMNS,
        overwrite=True,
    )

    total_steps = 0
    global_episode = 0
    eval_history: List[Dict[str, float]] = []

    initial_eval = evaluate_policy(policy, eval_env)
    for difficulty, metrics in initial_eval.items():
        eval_logger.append_eval(0, "initial", 0, difficulty, metrics)
    initial_log = build_wandb_eval_metrics(initial_eval, stage_id=0)
    log_wandb(initial_log, 0)
    eval_history.append({"global_episode": 0, **initial_log})
    print_eval_summary(0, "initial", 0, initial_eval)

    curriculum = build_balanced_three_stage_curriculum(args.episodes)
    print(
        "Curriculum:",
        "; ".join(f"{stage.name}={stage.quota}" for stage in curriculum),
        flush=True,
    )
    for stage_id, stage in enumerate(curriculum, start=1):
        env.set_stage(stage.train_difficulties, seed=args.seed + stage_id)
        schedule = build_stage_schedule(stage.quota, seed=args.seed + stage_id)
        for stage_episode, difficulty in enumerate(schedule, start=1):
            global_episode += 1
            total_steps, train_metrics, loss_log = train_one_episode(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                old_replay_buffer=old_replay_buffer,
                algo_args=algo_args,
                rarr_rewarder=rarr_rewarder,
                global_step=global_episode,
                total_steps=total_steps,
                difficulty=difficulty,
            )
            train_logger.append_train_episode(global_episode, stage.name, stage_episode, train_metrics)
            if global_episode % 20 == 0 or global_episode == 1 or global_episode == args.episodes:
                print(
                    f"Episode {global_episode}/{args.episodes} stage={stage.name} "
                    f"stage_episode={stage_episode} difficulty={difficulty} "
                    f"total_steps={total_steps} reward={float(train_metrics.get('original_reward', 0.0)):.3f} "
                    f"AvgVD={float(train_metrics.get('avg_voltage_deviation', 0.0)):.5f} "
                    f"VVR={float(train_metrics.get('voltage_violation_rate', 0.0)):.5f} "
                    f"SOCerr={float(train_metrics.get('terminal_soc_error', 0.0)):.5f}",
                    flush=True,
                )

            log_data = {
                "global_episode": global_episode,
                "train/stage_id": stage_id,
                "train/original_reward": float(train_metrics.get("original_reward", 0.0)),
                "train/train_reward": float(train_metrics.get("train_reward", 0.0)),
                "train/avg_voltage_deviation": float(train_metrics.get("avg_voltage_deviation", 0.0)),
                "train/terminal_soc_error": float(train_metrics.get("terminal_soc_error", 0.0)),
                "train/voltage_violation_rate": float(train_metrics.get("voltage_violation_rate", 0.0)),
                "train/success": float(train_metrics.get("success", 0.0)),
                "train/total_steps": total_steps,
            }
            for key, value in loss_log.items():
                log_data[f"loss/{key}"] = value
            for key in [
                "voltage_reward_sum",
                "terminal_soc_penalty",
                "raw_terminal_soc_error",
                "deadband_soc_violation",
                "rarr_reward_sum",
                "rarr_voltage_sum",
                "rarr_soc_credit_sum",
                "rarr_d_max",
                "rarr_d_final",
                "rarr_phi_final",
                "recoverability_violation_sum",
                "recoverability_violation_rate",
                "recoverability_violation_count",
            ]:
                log_data[f"train/{key}"] = float(train_metrics.get(key, 0.0))
            log_wandb(log_data, global_episode)

            if global_episode % args.eval_freq_episodes == 0 or global_episode == args.episodes:
                eval_metrics = evaluate_policy(policy, eval_env)
                for difficulty_name, metrics in eval_metrics.items():
                    eval_logger.append_eval(global_episode, stage.name, stage_episode, difficulty_name, metrics)
                eval_log = build_wandb_eval_metrics(eval_metrics, stage_id=stage_id)
                log_wandb(eval_log, global_episode)
                eval_history.append({"global_episode": global_episode, **eval_log})
                print_eval_summary(global_episode, stage.name, stage_episode, eval_metrics)
                if args.save_model:
                    policy.save(os.path.join(args.model_dir, file_name))

            if global_episode >= args.episodes:
                break
        if global_episode >= args.episodes:
            break

    if args.save_model:
        policy.save(os.path.join(args.model_dir, file_name))

    final_eval = evaluate_policy(policy, eval_env)
    final_all = build_wandb_eval_metrics(final_eval)
    convergence_by_difficulty = compute_convergence_by_difficulty(eval_history, max_episode=args.episodes)
    episodes_to_convergence = convergence_by_difficulty.get("all")
    convergence_path = write_convergence_result(
        result_dir=args.result_dir,
        file_name=file_name,
        run_name=args.run_name,
        rd_method=args.rd_method,
        seed=args.seed,
        episodes_to_convergence=convergence_by_difficulty,
    )
    print(
        "Episodes to convergence: "
        f"all={convergence_by_difficulty.get('all')}, "
        f"easy={convergence_by_difficulty.get('easy')}, "
        f"medium={convergence_by_difficulty.get('medium')}, "
        f"hard={convergence_by_difficulty.get('hard')}",
        flush=True,
    )
    finalize_wandb_run(
        summary={
            "total_episodes": int(global_episode),
            "total_steps": int(total_steps),
            "episodes_to_convergence": episodes_to_convergence,
            "easy_episodes_to_convergence": convergence_by_difficulty.get("easy"),
            "medium_episodes_to_convergence": convergence_by_difficulty.get("medium"),
            "hard_episodes_to_convergence": convergence_by_difficulty.get("hard"),
            "final_eval_all_original_reward": final_all.get("eval/all/original_reward"),
            "final_eval_all_avg_voltage_deviation": final_all.get("eval/all/avg_voltage_deviation"),
            "final_eval_all_terminal_soc_error": final_all.get("eval/all/terminal_soc_error"),
            "final_eval_all_voltage_violation_rate": final_all.get("eval/all/voltage_violation_rate"),
            "final_eval_all_success_rate": final_all.get("eval/all/success_rate"),
            "final_eval_all_soc_success_rate": final_all.get("eval/all/soc_success_rate"),
            "final_eval_all_voltage_success_rate": final_all.get("eval/all/voltage_success_rate"),
        },
        file_paths=[train_logger.path, eval_logger.path, convergence_path],
        file_globs=[os.path.join(args.model_dir, f"{file_name}*")] if args.save_model else [],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        parsed = None
        try:
            parsed = parse_args()
            run_name = parsed.run_name or f"{parsed.policy}_{parsed.rd_method}_sparse_33bus_seed{parsed.seed}_ep{parsed.episodes}"
            log_dir = os.path.join(parsed.result_dir, "logs")
        except Exception:
            run_name = "sparse_33bus_train"
            log_dir = os.path.join(PROJECT_ROOT, "result", "sparse_33bus_main", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{run_name}_error.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{_dt.datetime.now().isoformat(timespec='seconds')}] unhandled exception\n")
            f.write(traceback.format_exc())
        print(f"[sparse_33bus_train] unhandled exception, see {log_path}", file=sys.stderr)
        raise

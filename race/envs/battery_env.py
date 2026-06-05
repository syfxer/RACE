import os
import argparse
import tempfile
import csv


def _configure_runtime_dirs():
    project_root = os.path.dirname(os.path.abspath(__file__))
    runtime_root = os.path.join(project_root, ".runtime_cache")
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

import gymnasium as gym
import numpy as np
import torch

import pandapower as pp
import pandapower.networks as pn
from gymnasium import spaces

import wandb
from race.training import replay_buffer as utils
import race.algos.td3 as TD3
import race.algos.ddpg as DDPG
import race.algos.sac as SAC
import race.algos.ppo as PPO
from race.algos.ppo import Normalization, RewardScaling
from race.utils.metric_logger import ElectricMetricLogger
from race.utils.metrics import compute_episodes_to_convergence
from race.rewards.reward_deadband import compute_terminal_soc_penalty
from race.utils.wandb_utils import finalize_wandb_run

torch.set_num_threads(5)


LEGACY_EVAL_METRIC_COLUMNS = [
    "global_episode",
    "reward",
    "steps",
    "quality_sum",
    "quality_mean",
    "voltage_deviation_mean_pu",
    "voltage_deviation_max_pu",
    "voltage_violation_count",
    "voltage_violation_rate",
    "voltage_violation_duration_steps",
    "voltage_violation_duration_hours",
    "voltage_violation_bus_steps",
    "voltage_violation_bus_rate",
    "grid_loss_mwh",
    "soc_penalty_sum",
    "final_soc",
    "terminal_soc_error",
    "bess_throughput_mwh",
    "equivalent_cycles",
    "powerflow_nonconvergence_count",
    "powerflow_nonconvergence_rate",
]


PRE_EX_EVAL_METRIC_COLUMNS = [
    "global_episode",
    "total_reward",
    "terminal_soc_error",
    "average_voltage_deviation",
    "episodes_to_convergence",
]


def build_pre_ex_eval_row(step, eval_summary, episode_length=96, episodes_to_convergence=""):
    return {
        "global_episode": int(step // episode_length),
        "total_reward": float(eval_summary["reward"]),
        "terminal_soc_error": float(eval_summary["terminal_soc_error"]),
        "average_voltage_deviation": float(eval_summary["voltage_deviation_mean_pu"]),
        "episodes_to_convergence": episodes_to_convergence,
    }


def rewrite_pre_ex_eval_rows(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRE_EX_EVAL_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_name(value):
    invalid_chars = '<>:"/\\|?*'
    return "".join("_" if ch in invalid_chars else ch for ch in str(value))


def get_llm_model_name(args):
    return sanitize_name(getattr(args, "llm_model_name", os.environ.get("MODEL", "Qwen3.6-Plus")))


def uses_llm_model_name(rd_method):
    return False


def normalize_rd_method(rd_method):
    aliases = {
        "RARR-SAC": "RARR",
        "RARR_SAC": "RARR",
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
        "SAC_VIB_StepSOC": "VIB_StepSOC",
        "VIB-StepSOC": "VIB_StepSOC",
        "VIB_StepSOC_SAC": "VIB_StepSOC",
    }
    return aliases.get(rd_method, rd_method)


def uses_rarr_reward(rd_method):
    return rd_method in {"RARR", "RARR_VIB", "RARR_RD", "RARR_RRD", "RARR_Diaster"}


def uses_rarr_residual(rd_method):
    return rd_method in {"RARR_VIB", "RARR_RD", "RARR_RRD", "RARR_Diaster"}


def uses_stepsoc_reward(rd_method):
    return rd_method in {"VIB_StepSOC"}


def uses_stepsoc_residual(rd_method):
    return rd_method in {"VIB_StepSOC"}


def get_result_model_tag(args):
    if uses_llm_model_name(args.rd_method):
        return get_llm_model_name(args)
    if args.rd_method == "None" and getattr(args, "dense_r", False):
        return "dense-reward"
    return "nonllm"


def build_file_name(args):
    return f"{args.policy}_{args.env}_{args.rd_method}_{get_result_model_tag(args)}_seed{args.seed}"


def make_training_env(env_name, ctrl_cost_weight=-1):
    if env_name == "BatteryManagementEnv":
        return BatteryManagementEnv()
    if env_name == "SparseHardcodedDayEnv":
        from race.envs.sparse_33bus_env import SparseHardcodedDayEnv

        return SparseHardcodedDayEnv()
    if ctrl_cost_weight >= 0:
        return gym.make(env_name, ctrl_cost_weight=ctrl_cost_weight)
    return gym.make(env_name)

# ====================== 你的环境类（完全未改动） ======================
class BatteryManagementEnv(gym.Env):
    def __init__(self):
        super(BatteryManagementEnv, self).__init__()
        self.single_observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                                                   high=np.array([1.00, 1.00, 1.00, 1.00, 1.00]), dtype=np.float32)
        self.single_action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        self.Battery_Capacity = 7.7
        self.slice = 0.25
        self.state = [[0.00, 0.00, 38.52680745, 0.00, 7.795145639]]
        self.num_envs = 1
        self.cap = 0.7
        # pandapower uses 0-based bus indices; paper/system node numbers are 1-based.
        self.pv_bus = 32
        self.wt_bus = 21
        self.bess_bus = 17
        self.num_buses = 33
        self.target_soc = 0.3
        self.soc_min = 0.0
        self.soc_max = 1.0
        self.soc_deadband = float(os.environ.get("BATTERY_SOC_DEADBAND", "0.02"))
        self.lambda_soc = float(os.environ.get("BATTERY_LAMBDA_SOC", "100.0"))
        self.soc_deadband_mode = os.environ.get("BATTERY_SOC_DEADBAND_MODE", "linear")
        self.max_bess_power_mw = 2.0
        self.voltage_lower_bound = 0.95
        self.voltage_upper_bound = 1.05
        self.voltage_reward_span = 0.07
        self.runpp_use_numba = os.environ.get("BATTERY_RUNPP_USE_NUMBA", "0") == "1"
        self.verbose_step = False
        self.t = 0
        self.WT = np.array(
            [7.795145639, 30, 35, 8.846969231, 16.46602734, 18.13731014, 0, 22.51716352, 0, 17.9231927, 14.15372591, 0,
             11.6233221, 4, 35.35145853, 32, 49, 14.33133905, 19.00167218, 50, 30.48483598, 36.18525123, 21.04862466,
             58.55085761, 7, 13.38712664, 0.9, 3.056863165, 32, 9.720436673, 17, 2.491215512, 15.4306888, 13,
             23.60703705, 50, 50, 66, 29.75268875, 19.94024882, 65.28505115, 62.98944949, 62.75321569, 25.41053571,
             95.88858417, 86.48719301, 75.17048763, 51.40892866, 96.45673321, 98.95653492, 58.69891281, 57.17791032,
             98, 97.33019532, 96.52678304, 72.53758377, 62.91378467, 89.78739585, 69.47327471, 68.24764212,
             72.57372903, 100, 99.7924927, 87.26289182, 73.78566709, 93, 89, 100, 78.12244919, 100, 99, 91.5508517,
             75.62210048, 100, 71.04460315, 83.93701513, 86.05545333, 86.1450672, 88.33546098, 74.33299008,
             89.54168021, 89.98797207, 80.2885353, 100, 96.13676328, 61.66369873, 80.96525581, 86, 40.71576154,
             63.2995258, 32.5655823, 56.56493033, 46.10773804, 11.23771407, 0, 21.58711212])  # 这里放入你原始的PV数据
        self.PV = np.array(
            [0, 0, 0, 1.534893928, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 13.23842331, 18, 17, 12, 6.294037376, 30,
             33.7784774, 40.12795867, 41.45072669, 12.87897941, 20, 48.62089279, 45.56991566, 37.72844958, 55.03924065,
             57.27395133, 77.1460199, 31.53641078, 82.4481725, 54.45656023, 36.45647462, 92, 80.02727971, 66.11332888,
             50.72330911, 74.05187436, 52.62290957, 57.07280675, 17.50162697, 25.76591962, 27.25622161, 45.01228529,
             23.71170186, 44.56052686, 14.11180894, 43.54698389, 8.261743441, 40.91542105, 51.00898113, 18.82405249,
             31.85294629, 31.10056238, 42.19804925, 11.14644011, 41.20804216, 34.76803192, 20.388283, 21.32341377,
             7.464497931, 7.823391371, 3.058413701, 0, 0, 12, 0, 8.458569964, 0, 0, 0, 2.663710742, 0, 0, 0, 2, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 这里放入你原始的WT数据
        self.L = np.array(
            [38.52680745, 48, 52.83431338, 12.70716651, 41.09933268, 32.368543, 51.33887579, 49.72165616, 10.96586938,
             47.17449319, 10.1811827, 28.39494669, 17.52102577, 21.49084683, 50.26449404, 16.33932982, 35.66857677,
             1.122451028, 35.27626195, 25.33153535, 16.57206761, 48.81216394, 1.06183333, 45.06336446, 38.92842238,
             48.35322762, 9.118658638, 34.59910003, 1.290028902, 54.91118903, 1.506844108, 18.66403543, 36.01146046,
             42.94016099, 44.48765596, 17.51547332, 30.24421748, 30, 52.28523656, 43.90483901, 82.03554115, 44.69683367,
             79.31542833, 43.51056931, 83.19949454, 85.16741374, 99.51924713, 55.10400638, 69.72695304, 75.10100365,
             70.58174233, 100, 71.56391619, 73.54736895, 90.84908664, 99.5, 98.40613288, 51.87624311, 40.5054955,
             98.63365864, 26.89196698, 64.37961463, 50.75045943, 65.50168208, 17.1951119, 57.76211918, 22.28244356,
             29.79231207, 89.71000347, 93.12346184, 71.6642721, 42.21021461, 100, 100, 86.32109956, 74.83646558,
             83.61669166, 100, 64.05377977, 98.3, 79.35033602, 84.60830134, 100, 100, 82.97375079, 81.63787632,
             61.00385313, 99.6, 80.94409778, 74.55806944, 54.95072907, 83.05763655, 59.12786812, 62.72374535,
             43.65248929, 10.09138061])  # 这里放入你原始的L数据
        self._normalize_data()
        self._init_powerflow_cache()
        self.state = [[0.00, 0.00, self.L[0], self.PV[0], self.WT[0]]]

    def _normalize_data(self):
        self.PV = self.PV.astype(np.float32) / 100.0
        self.WT = self.WT.astype(np.float32) / 100.0
        self.L = self.L.astype(np.float32) / 100.0
        self.L = np.clip(self.L, 0.0, None)
        peak_mask = self.L >= 0.8
        self.L[peak_mask] = np.minimum(self.L[peak_mask] * 1.12, 1.15)

    def _init_powerflow_cache(self):
        self._pf_net = pn.case33bw()
        self._pf_net["max_vm_pu"] = 2.0
        self._pf_net["min_vm_pu"] = 0.0
        self._base_load_p_mw = self._pf_net.load["p_mw"].to_numpy(dtype=float).copy()
        self._base_load_q_mvar = self._pf_net.load["q_mvar"].to_numpy(dtype=float).copy()
        self._pv_sgen_idx = pp.create.create_sgen(
            self._pf_net,
            self.pv_bus,
            p_mw=0.0,
            q_mvar=0.0,
            name="aggregated_pv",
        )
        self._wt_sgen_idx = pp.create.create_sgen(
            self._pf_net,
            self.wt_bus,
            p_mw=0.0,
            q_mvar=0.0,
            name="aggregated_wt",
        )
        self._bess_storage_idx = pp.create.create_storage(
            self._pf_net,
            self.bess_bus,
            p_mw=0.0,
            max_e_mwh=30,
            name="bess",
        )

    def _update_cached_powerflow_inputs(self, action_power_mw):
        load_scale = max(0.0, float(self.L[self.t]))
        self._pf_net.load.loc[:, "p_mw"] = self._base_load_p_mw * load_scale
        self._pf_net.load.loc[:, "q_mvar"] = self._base_load_q_mvar * load_scale
        self._pf_net.sgen.at[self._pv_sgen_idx, "p_mw"] = float(self.PV[self.t] * self.cap)
        self._pf_net.sgen.at[self._wt_sgen_idx, "p_mw"] = float(self.WT[self.t] * self.cap)
        self._pf_net.storage.at[self._bess_storage_idx, "p_mw"] = float(action_power_mw)

    def _build_uncached_powerflow_net(self, action_power_mw):
        net = pn.case33bw()
        load_scale = max(0.0, float(self.L[self.t]))
        for j in range(len(net.load)):
            net.load.at[j, "p_mw"] *= load_scale
            net.load.at[j, "q_mvar"] *= load_scale

        net["max_vm_pu"] = 2.0
        net["min_vm_pu"] = 0.0

        pv_p_mw = float(self.PV[self.t] * self.cap)
        wt_p_mw = float(self.WT[self.t] * self.cap)
        pp.create.create_sgen(net, self.pv_bus, p_mw=pv_p_mw, q_mvar=0.0, name="aggregated_pv")
        pp.create.create_sgen(net, self.wt_bus, p_mw=wt_p_mw, q_mvar=0.0, name="aggregated_wt")
        pp.create.create_storage(net, self.bess_bus, float(action_power_mw), 30, name="bess")
        return net

    def _build_initial_powerflow_metrics(self, action_projection):
        action_power_mw = float(action_projection["applied_power_mw"])
        throughput_mwh = abs(action_power_mw) * self.slice
        equivalent_cycle_increment = throughput_mwh / (2.0 * self.Battery_Capacity)
        return {
            "quality": 10.0,
            "voltage_deviation_mean_pu": 1.0,
            "voltage_deviation_max_pu": 1.0,
            "voltage_violation_bus_count": float(self.num_buses),
            "voltage_violation_bus_ratio": 1.0,
            "has_voltage_violation": 1.0,
            "grid_loss_mw": 0.0,
            "powerflow_converged": 0.0,
            "bess_power_mw": action_power_mw,
            "bess_throughput_mwh": throughput_mwh,
            "equivalent_cycle_increment": equivalent_cycle_increment,
            "requested_bess_power_mw": float(action_projection["requested_power_mw"]),
            "applied_bess_action": float(action_projection["applied_action"]),
            "requested_bess_action": float(action_projection["requested_action"]),
            "bess_action_clipped": float(action_projection["action_was_clipped"]),
        }

    def _update_metrics_from_powerflow_result(self, metrics, net):
        vm_pu = net.res_bus["vm_pu"].to_numpy(dtype=float)
        abs_dev = np.abs(vm_pu - 1.0)
        voltage_violations = (vm_pu < self.voltage_lower_bound) | (vm_pu > self.voltage_upper_bound)
        line_loss_mw = float(net.res_line["pl_mw"].sum()) if len(net.res_line) > 0 else 0.0
        trafo_loss_mw = 0.0
        if hasattr(net, "res_trafo") and len(net.res_trafo) > 0:
            trafo_loss_mw += float(net.res_trafo["pl_mw"].sum())
        if hasattr(net, "res_trafo3w") and len(net.res_trafo3w) > 0:
            trafo_loss_mw += float(net.res_trafo3w["pl_mw"].sum())

        metrics.update({
            "quality": float(np.minimum(1.0, abs_dev / self.voltage_reward_span).mean()),
            "voltage_deviation_mean_pu": float(abs_dev.mean()),
            "voltage_deviation_max_pu": float(abs_dev.max()),
            "voltage_violation_bus_count": float(voltage_violations.sum()),
            "voltage_violation_bus_ratio": float(voltage_violations.mean()),
            "has_voltage_violation": float(voltage_violations.any()),
            "grid_loss_mw": float(line_loss_mw + trafo_loss_mw),
            "powerflow_converged": 1.0,
        })
        return metrics

    def _clip_soc(self, soc):
        return float(np.clip(soc, self.soc_min, self.soc_max))

    def _project_action_to_soc_limits(self, action):
        current_soc = float(self.state[0][1])
        action_value = float(np.squeeze(action))

        max_charge_power_mw = max(
            0.0,
            (self.soc_max - current_soc) * self.Battery_Capacity / self.slice,
        )
        max_discharge_power_mw = max(
            0.0,
            (current_soc - self.soc_min) * self.Battery_Capacity / self.slice,
        )
        feasible_min_action = -min(self.max_bess_power_mw, max_discharge_power_mw) / self.max_bess_power_mw
        feasible_max_action = min(self.max_bess_power_mw, max_charge_power_mw) / self.max_bess_power_mw
        applied_action = float(np.clip(action_value, feasible_min_action, feasible_max_action))

        requested_power_mw = float(action_value * self.max_bess_power_mw)
        applied_power_mw = float(applied_action * self.max_bess_power_mw)
        next_soc = self._clip_soc(current_soc + applied_power_mw * self.slice / self.Battery_Capacity)
        soc_delta = next_soc - current_soc
        applied_power_mw = float(soc_delta * self.Battery_Capacity / self.slice)
        applied_action = float(applied_power_mw / self.max_bess_power_mw)

        return {
            "requested_action": action_value,
            "applied_action": applied_action,
            "requested_power_mw": requested_power_mw,
            "applied_power_mw": applied_power_mw,
            "next_soc": next_soc,
            "action_was_clipped": float(not np.isclose(action_value, applied_action)),
        }

    def Quality_Evaluate_uncached(self, action):
        action_projection = self._project_action_to_soc_limits(action)
        action_power_mw = float(action_projection["applied_power_mw"])
        current_soc = float(self.state[0][1])

        net = self._build_uncached_powerflow_net(action_power_mw)
        metrics = self._build_initial_powerflow_metrics(action_projection)
        try:
            pp.runpp(net, numba=self.runpp_use_numba)
        except Exception as exc:
            self.state[0][1] = current_soc
            if self.verbose_step:
                print(f"[PowerFlow Warning] t={self.t}, error: {exc}")
        else:
            self._update_metrics_from_powerflow_result(metrics, net)
            self.state[0][1] = float(action_projection["next_soc"])
        return metrics

    def Quality_Evaluate(self, action):
        action_projection = self._project_action_to_soc_limits(action)
        action_power_mw = float(action_projection["applied_power_mw"])
        current_soc = float(self.state[0][1])
        metrics = self._build_initial_powerflow_metrics(action_projection)
        self._update_cached_powerflow_inputs(action_power_mw)
        try:
            pp.runpp(self._pf_net, numba=self.runpp_use_numba)
        except Exception as exc:
            self.state[0][1] = current_soc
            if self.verbose_step:
                print(f"[PowerFlow Warning] t={self.t}, error: {exc}")
        else:
            self._update_metrics_from_powerflow_result(metrics, self._pf_net)
            self.state[0][1] = float(action_projection["next_soc"])
        return metrics

    def reset(self, seed=None, options=None):
        self.t = 0
        self.state = [[0.00, 0.00, self.L[0], self.PV[0], self.WT[0]]]
        return np.array(self.state[0]), {}

    def step(self, action):
        done = self.t == 95

        flow_metrics = self.Quality_Evaluate(action)
        quality = float(flow_metrics["quality"])
        voltage_reward = -quality
        if done:
            soc_terms = compute_terminal_soc_penalty(
                final_soc=float(self.state[0][1]),
                soc_target=self.target_soc,
                soc_deadband=self.soc_deadband,
                lambda_soc=self.lambda_soc,
                mode=self.soc_deadband_mode,
            )
        else:
            soc_terms = {
                "terminal_soc_penalty": 0.0,
                "terminal_soc_penalty_cost": 0.0,
                "raw_terminal_soc_error": float(abs(self.state[0][1] - self.target_soc)),
                "deadband_soc_violation": 0.0,
            }
        reward = voltage_reward + float(soc_terms["terminal_soc_penalty"])
        self.t += 1

        if self.t <= 95:
            next_state = [[self.t / 96.0, self.state[0][1], self.L[self.t], self.PV[self.t], self.WT[self.t]]]
        else:
            next_state = self.state

        self.state = next_state
        info = dict(flow_metrics)
        info.update({
            "quality": float(quality),
            "voltage_reward": float(voltage_reward),
            "soc_penalty": float(soc_terms["terminal_soc_penalty_cost"]),
            "terminal_soc_penalty": float(soc_terms["terminal_soc_penalty"]),
            "raw_terminal_soc_error": float(soc_terms["raw_terminal_soc_error"]),
            "deadband_soc_violation": float(soc_terms["deadband_soc_violation"]),
            "soc_deadband": float(self.soc_deadband),
            "soc": float(self.state[0][1]),
            "soc_error_to_target": float(abs(self.state[0][1] - self.target_soc)),
        })
        return np.array(next_state[0]), reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass


def init_episode_metrics(initial_soc):
    return {
        "quality_sum": 0.0,
        "voltage_deviation_sum_pu": 0.0,
        "voltage_deviation_max_pu": 0.0,
        "voltage_violation_duration_steps": 0.0,
        "voltage_violation_bus_steps": 0.0,
        "grid_loss_mwh": 0.0,
        "soc_penalty_sum": 0.0,
        "final_soc": float(initial_soc),
        "bess_throughput_mwh": 0.0,
        "equivalent_cycles": 0.0,
        "powerflow_nonconvergence_count": 0.0,
    }


def update_episode_metrics(metrics, info, slice_hours):
    metrics["quality_sum"] += float(info.get("quality", 0.0))
    metrics["voltage_deviation_sum_pu"] += float(info.get("voltage_deviation_mean_pu", 0.0))
    metrics["voltage_deviation_max_pu"] = max(
        metrics["voltage_deviation_max_pu"],
        float(info.get("voltage_deviation_max_pu", 0.0)),
    )
    metrics["voltage_violation_duration_steps"] += float(info.get("has_voltage_violation", 0.0))
    metrics["voltage_violation_bus_steps"] += float(info.get("voltage_violation_bus_count", 0.0))
    metrics["grid_loss_mwh"] += float(info.get("grid_loss_mw", 0.0)) * slice_hours
    metrics["soc_penalty_sum"] += float(info.get("soc_penalty", 0.0))
    metrics["final_soc"] = float(info.get("soc", metrics["final_soc"]))
    metrics["bess_throughput_mwh"] += float(info.get("bess_throughput_mwh", 0.0))
    metrics["equivalent_cycles"] += float(info.get("equivalent_cycle_increment", 0.0))
    metrics["powerflow_nonconvergence_count"] += 1.0 - float(info.get("powerflow_converged", 1.0))


def finalize_episode_metrics(metrics, episode_reward, episode_timesteps, env):
    steps = max(int(episode_timesteps), 1)
    violation_duration_steps = float(metrics["voltage_violation_duration_steps"])
    return {
        "reward": float(episode_reward),
        "steps": int(episode_timesteps),
        "quality_sum": float(metrics["quality_sum"]),
        "quality_mean": float(metrics["quality_sum"] / steps),
        "voltage_deviation_mean_pu": float(metrics["voltage_deviation_sum_pu"] / steps),
        "voltage_deviation_max_pu": float(metrics["voltage_deviation_max_pu"]),
        "voltage_violation_count": float(violation_duration_steps),
        "voltage_violation_rate": float(violation_duration_steps / steps),
        "voltage_violation_duration_steps": float(violation_duration_steps),
        "voltage_violation_duration_hours": float(violation_duration_steps * env.slice),
        "voltage_violation_bus_steps": float(metrics["voltage_violation_bus_steps"]),
        "voltage_violation_bus_rate": float(metrics["voltage_violation_bus_steps"] / (steps * env.num_buses)),
        "grid_loss_mwh": float(metrics["grid_loss_mwh"]),
        "soc_penalty_sum": float(metrics["soc_penalty_sum"]),
        "final_soc": float(metrics["final_soc"]),
        "terminal_soc_error": float(abs(metrics["final_soc"] - env.target_soc)),
        "bess_throughput_mwh": float(metrics["bess_throughput_mwh"]),
        "equivalent_cycles": float(metrics["equivalent_cycles"]),
        "powerflow_nonconvergence_count": float(metrics["powerflow_nonconvergence_count"]),
        "powerflow_nonconvergence_rate": float(metrics["powerflow_nonconvergence_count"] / steps),
    }


def average_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    averaged = {}
    for key in keys:
        values = [float(metric_dict[key]) for metric_dict in metric_dicts]
        averaged[key] = float(np.mean(values))
    return averaged


# ====================== 评估函数（适配自定义环境） ======================
def eval_policy(policy, env_name, seed, eval_episodes=10, ctrl_cost_weight=-1):
    eval_env = make_training_env(env_name, ctrl_cost_weight=ctrl_cost_weight)

    episode_summaries = []
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_timesteps = 0
        episode_metrics = init_episode_metrics(float(state[1]))
        while not done:
            action = policy.select_action(np.array(state), deterministic=True)
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_timesteps += 1
            update_episode_metrics(episode_metrics, info, eval_env.slice)
        episode_summaries.append(
            finalize_episode_metrics(episode_metrics, episode_reward, episode_timesteps, eval_env)
        )

    eval_summary = average_metric_dicts(episode_summaries)
    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: reward={eval_summary['reward']:.3f}, "
        f"voltage_dev_mean={eval_summary['voltage_deviation_mean_pu']:.5f} pu, "
        f"grid_loss={eval_summary['grid_loss_mwh']:.5f} MWh"
    )
    print("---------------------------------------")
    return eval_summary


# ====================== 主训练脚本（完全按第二段风格重构） ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--env", default="BatteryManagementEnv")   # 默认改成你的环境
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=10e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)

    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument("--adaptive_alpha", default=True, action="store_false")
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_data", action="store_true")
    parser.add_argument("--load_model", default="")
    parser.add_argument('--target_update_freq', type=int, default=2)

    # ==================== 第二段新增的研究参数（全部保留） ====================
    parser.add_argument("--dense_r", default=False, action="store_true")
    parser.add_argument("--rrd_unbiased", default=False, action="store_true")
    parser.add_argument("--rd_method", default='None')
    parser.add_argument("--exp_name", default='None')
    parser.add_argument("--run_name", default='None')
    parser.add_argument("--rrd_k", default=64, type=int)
    parser.add_argument("--llm_response_dir", type=str, default='factor_reward_decomp_response_data')
    parser.add_argument("--llm_model_name", type=str, default=os.environ.get("MODEL", "Qwen3.6-Plus"))
    parser.add_argument('--llm_n', type=int, default=5)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument("--direct_generate", default=False, action="store_true")
    parser.add_argument("--IB_latent_dim", default=8, type=int)
    parser.add_argument("--IB_kl_weight", default=5e-3, type=float)
    parser.add_argument("--soc_deadband", default=0.02, type=float)
    parser.add_argument("--lambda_soc", default=100.0, type=float)
    parser.add_argument("--soc_deadband_mode", default="linear", choices=["linear", "squared"])
    parser.add_argument("--lambda_rarr", default=100.0, type=float)
    parser.add_argument("--rarr_mode", default="linear", choices=["linear", "squared"])
    parser.add_argument("--lambda_stepsoc", default=100.0, type=float)
    parser.add_argument("--stepsoc_mode", default="linear", choices=["linear", "squared"])
    parser.add_argument("--rarr_lambda_soc", default=100.0, type=float)
    parser.add_argument("--rarr_voltage_weight", default=1.0, type=float)
    parser.add_argument("--rarr_use_gamma", default=True, action="store_false")
    parser.add_argument("--rarr_penalty_power", default=1.0, type=float)
    parser.add_argument("--rarr_residual_eta", default=1.0, type=float)
    parser.add_argument("--rarr_vib_eta", default=1.0, type=float)
    parser.add_argument("--rarr_vib_beta", default=1e-3, type=float)
    parser.add_argument("--rarr_vib_latent_dim", default=8, type=int)
    parser.add_argument("--rarr_vib_hidden_dim", default=128, type=int)
    parser.add_argument("--rarr_vib_lr", default=3e-4, type=float)
    parser.add_argument("--rarr_vib_reward_clip", default=None, type=float)
    parser.add_argument("--ctrl_cost_weight", default=-1, type=float)
    parser.add_argument("--sparse", default=False, action="store_true")
    parser.add_argument("--state_norm", default=True, action="store_false")
    parser.add_argument("--reward_scale", default=False, action="store_true")
    parser.add_argument("--result_dir", default="result", type=str)
    parser.add_argument("--model_dir", default="./models", type=str)
    parser.add_argument("--reward_train_start", default=10000, type=int)
    parser.add_argument("--quick_profile", default="none", choices=["none", "quick_5h"])
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", "./wandb_result"))
    parser.add_argument("--wandb_project", default="RD_BATTERY")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_tags", default="")
    parser.add_argument("--pre_ex_metrics", action="store_true")
    parser.add_argument("--gpu_gate", action="store_true", default=False)
    parser.add_argument("--no_gpu_gate", dest="gpu_gate", action="store_false")
    parser.add_argument("--gpu_gate_memory_threshold_mb", type=int, default=512)
    parser.add_argument("--gpu_gate_poll_seconds", type=int, default=20)
    parser.add_argument("--gpu_gate_stale_seconds", type=int, default=24 * 3600)

    args = parser.parse_args()
    args.rd_method = normalize_rd_method(args.rd_method)

    if args.quick_profile == "quick_5h":
        args.max_timesteps = 1800
        args.eval_freq = 900
        args.eval_episodes = 2
        args.save_model = False
        args.save_data = False
        args.reward_train_start = 512
        if args.policy == "PPO":
            args.batch_size = 512
            args.start_timesteps = 0
        else:
            args.batch_size = 128
            args.start_timesteps = 300

    # 稀疏奖励处理（第二段原逻辑）
    if args.sparse:
        args.ctrl_cost_weight = 0.0
        args.sparse_threshold = 1000

    if args.rd_method == 'IRCR':
        args.buffer_size = 300000
        args.tau = 0.001
        args.batch_size = 512
    if args.policy == "PPO" and args.quick_profile == "none":
        args.batch_size = 2048

    if 'unbiased' in args.rd_method:
        args.rrd_unbiased = True
    else:
        args.rrd_unbiased = False

    if args.exp_name == 'None':
        args.exp_name = args.rd_method + '_' + args.policy
    if args.run_name == 'None':
        args.run_name = args.exp_name + '_' + get_result_model_tag(args) + '_' + args.env + '_' + str(args.seed)

    file_name = build_file_name(args)
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if args.save_model and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    gate = None
    if args.gpu_gate and torch.cuda.is_available():
        from experiment.gpu_gate import GpuFifoGate

        gate = GpuFifoGate(
            root=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime_cache", "gpu_gate"),
            memory_threshold_mb=args.gpu_gate_memory_threshold_mb,
            stale_seconds=args.gpu_gate_stale_seconds,
            poll_seconds=args.gpu_gate_poll_seconds,
        )
        gate.acquire()

    wandb_mode = args.wandb_mode
    if wandb_mode == "online":
        try:
            wandb.login(relogin=False)
        except Exception as exc:
            print(f"[wandb] login failed, fallback to offline mode: {exc}")
            wandb_mode = "offline"
    os.makedirs(args.wandb_dir, exist_ok=True)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        config=vars(args),
        name=args.run_name,
        mode=wandb_mode,
        dir=args.wandb_dir,
        tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] or None,
        save_code=True,
    )
    wandb.define_metric("t")
    for metric_name in [
        "eval_reward",
        "eval_quality_mean",
        "eval_voltage_deviation_mean_pu",
        "eval_voltage_deviation_max_pu",
        "eval_voltage_violation_count",
        "eval_voltage_violation_rate",
        "eval_voltage_violation_duration_steps",
        "eval_voltage_violation_duration_hours",
        "eval_voltage_violation_bus_steps",
        "eval_voltage_violation_bus_rate",
        "eval_grid_loss_mwh",
        "eval_final_soc",
        "eval_terminal_soc_error",
        "eval_bess_throughput_mwh",
        "eval_equivalent_cycles",
        "eval_powerflow_nonconvergence_count",
        "eval_powerflow_nonconvergence_rate",
        "train_reward",
        "episode_steps",
        "episodes",
        "actor_loss",
        "critic_loss",
        "reward_pred_err",
        "reward_model_loss",
        "rarr_reward_sum",
        "rarr_voltage_sum",
        "rarr_soc_credit_sum",
        "rarr_d_max",
        "rarr_d_final",
        "rarr_phi_final",
        "stepsoc_reward_sum",
        "stepsoc_voltage_sum",
        "stepsoc_soc_credit_sum",
        "stepsoc_violation_sum",
        "stepsoc_violation_rate",
        "rarr_vib_loss",
        "rarr_vib_pred_loss",
        "rarr_vib_kl",
        "rarr_vib_residual_return",
        "rarr_vib_pred_return",
        "alpha",
        "alpha_loss",
        "episode_quality_sum",
        "episode_quality_mean",
        "episode_voltage_deviation_mean_pu",
        "episode_voltage_deviation_max_pu",
        "episode_voltage_violation_count",
        "episode_voltage_violation_rate",
        "episode_voltage_violation_duration_steps",
        "episode_voltage_violation_duration_hours",
        "episode_voltage_violation_bus_steps",
        "episode_voltage_violation_bus_rate",
        "episode_grid_loss_mwh",
        "episode_soc_penalty_sum",
        "final_soc",
        "episode_terminal_soc_error",
        "episode_bess_throughput_mwh",
        "episode_equivalent_cycles",
        "episode_powerflow_nonconvergence_count",
        "episode_powerflow_nonconvergence_rate",
    ]:
        wandb.define_metric(metric_name, step_metric="t")

    # ====================== 创建环境（保持不变） ======================
    env = make_training_env(args.env, ctrl_cost_weight=args.ctrl_cost_weight)
    env.soc_deadband = float(args.soc_deadband)
    env.lambda_soc = float(args.lambda_soc)
    env.soc_deadband_mode = str(args.soc_deadband_mode)

    # ====================== 维度 & 最大步数（适配自定义环境） ======================
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]
    max_action = float(env.single_action_space.high[0])
    max_length = 96                                      # 你的环境固定 96 步
    args.obs_dim = state_dim
    args.action_dim = action_dim
    args.max_length = max_length
    args.max_action = max_action

    # ====================== 奖励模型（第二段核心功能） ======================
    rarr_rewarder = None
    if uses_rarr_reward(args.rd_method):
        from race.rewards import RARRReward, RARRRewardConfig

        delta_soc_max = env.max_bess_power_mw * env.slice / env.Battery_Capacity
        rarr_rewarder = RARRReward(
            RARRRewardConfig(
                T=max_length,
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

    stepsoc_rewarder = None
    if uses_stepsoc_reward(args.rd_method):
        from race.rewards import StepSOCReward, StepSOCRewardConfig

        stepsoc_rewarder = StepSOCReward(
            StepSOCRewardConfig(
                lambda_stepsoc=args.lambda_stepsoc,
                soc_target=env.target_soc,
                soc_deadband=args.soc_deadband,
                voltage_weight=args.rarr_voltage_weight,
                stepsoc_mode=args.stepsoc_mode,
            )
        )

    # ====================== 初始化策略 ======================
    kwargs = {
        "args": args,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    elif args.policy == "SAC":
        policy = SAC.SAC(**kwargs)
    elif args.policy == "PPO":
        policy = PPO.PPO(**kwargs)
    else:
        raise ValueError("Policy not recognized.")

    if gate is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gate.release()

    if args.load_model != "":
        policy.load(os.path.join(args.model_dir, file_name if args.load_model == 'default' else args.load_model))

    # ====================== Replay Buffer ======================
    if args.rd_method != 'IRCR':
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_length, args.buffer_size,
                                           log_prob=(args.policy == "PPO"))
    else:
        replay_buffer = utils.IRCRReplayBuffer(state_dim, action_dim, max_length, args.buffer_size)
    old_replay_buffer = PPO.ReplayBuffer(state_dim, action_dim, args.batch_size + 1) if args.policy == "PPO" else None
    state_norm = Normalization(shape=args.obs_dim) if args.policy == "PPO" else None
    reward_scaling = RewardScaling(shape=1, gamma=args.discount) if args.policy == "PPO" else None

    buffer_path = os.path.join(args.result_dir, f"{file_name}_buffer.npz")
    eval_metric_columns = PRE_EX_EVAL_METRIC_COLUMNS if args.pre_ex_metrics else LEGACY_EVAL_METRIC_COLUMNS
    electric_metric_logger = ElectricMetricLogger(
        args.result_dir,
        file_name,
        fieldnames=eval_metric_columns,
        overwrite=True,
    )

    # ====================== 初始评估 ======================
    evaluation_history = []
    pre_ex_eval_rows = []
    initial_eval = eval_policy(policy, args.env, args.seed, eval_episodes=args.eval_episodes, ctrl_cost_weight=args.ctrl_cost_weight)
    evaluation_history.append({"step": 0, **initial_eval})
    if args.pre_ex_metrics:
        initial_pre_ex_row = build_pre_ex_eval_row(0, initial_eval, episode_length=max_length)
        pre_ex_eval_rows.append(initial_pre_ex_row)
        electric_metric_logger.append_row(initial_pre_ex_row)
    else:
        electric_metric_logger.append(0, initial_eval)
    wandb.log({
        "eval_reward": initial_eval["reward"],
        "eval_quality_mean": initial_eval["quality_mean"],
        "eval_voltage_deviation_mean_pu": initial_eval["voltage_deviation_mean_pu"],
        "eval_voltage_deviation_max_pu": initial_eval["voltage_deviation_max_pu"],
        "eval_voltage_violation_count": initial_eval["voltage_violation_count"],
        "eval_voltage_violation_rate": initial_eval["voltage_violation_rate"],
        "eval_voltage_violation_duration_steps": initial_eval["voltage_violation_duration_steps"],
        "eval_voltage_violation_duration_hours": initial_eval["voltage_violation_duration_hours"],
        "eval_voltage_violation_bus_steps": initial_eval["voltage_violation_bus_steps"],
        "eval_voltage_violation_bus_rate": initial_eval["voltage_violation_bus_rate"],
        "eval_grid_loss_mwh": initial_eval["grid_loss_mwh"],
        "eval_final_soc": initial_eval["final_soc"],
        "eval_terminal_soc_error": initial_eval["terminal_soc_error"],
        "eval_bess_throughput_mwh": initial_eval["bess_throughput_mwh"],
        "eval_equivalent_cycles": initial_eval["equivalent_cycles"],
        "eval_powerflow_nonconvergence_count": initial_eval["powerflow_nonconvergence_count"],
        "eval_powerflow_nonconvergence_rate": initial_eval["powerflow_nonconvergence_rate"],
        "t": 0,
    })

    # ====================== 训练循环（完全按第二段结构） ======================
    state, _ = env.reset()
    raw_state = state.copy()
    if state_norm is not None and args.state_norm:
        state = state_norm(state)
    if reward_scaling is not None and args.reward_scale:
        reward_scaling.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    last_train = 0
    episode_metrics = init_episode_metrics(float(raw_state[1]))
    states, actions, next_states, rewards, dense_rewards, dones, a_log_probs = [], [], [], [], [], [], []
    raw_states, raw_next_states = [], []
    rarr_episode = {
        "reward_sum": 0.0,
        "voltage_sum": 0.0,
        "soc_credit_sum": 0.0,
        "d_max": 0.0,
        "d_final": 0.0,
        "phi_final": 0.0,
    }
    stepsoc_episode = {
        "reward_sum": 0.0,
        "voltage_sum": 0.0,
        "soc_credit_sum": 0.0,
        "violation_sum": 0.0,
        "violation_count": 0.0,
    }

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # 动作选择（第二段完全一致写法）
        if t < args.start_timesteps and args.policy != 'PPO':
            action = env.single_action_space.sample()
        else:
            if args.policy == 'TD3':
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
            elif args.policy == 'PPO':
                action, log_prob = policy.select_action(np.array(state), log_prob=True)
                a_log_probs.append(log_prob)
            else:
                action = policy.select_action(np.array(state))

        # 执行动作
        env_t_before_step = env.t
        next_state, env_reward, terminated, truncated, info = env.step(action)

        # 使用 reward_model 替换环境原始奖励（核心功能！）
        raw_next_state = next_state.copy()
        reward = env_reward
        if state_norm is not None and args.state_norm:
            next_state = state_norm(next_state)
        if reward_scaling is not None and args.reward_scale:
            reward = reward_scaling(reward)[0]

        done = terminated or truncated
        done_bool = float(done) if episode_timesteps < args.max_length else 0
        train_replay_reward = 0 if not done else episode_reward + env_reward
        if rarr_rewarder is not None:
            rarr_info = rarr_rewarder.compute(
                t=env_t_before_step,
                soc_t=float(raw_state[1]),
                soc_next=float(raw_next_state[1]),
                voltage_cost_t=float(info.get("quality", 0.0)),
            )
            train_replay_reward = rarr_info["reward"]
            if done and not uses_rarr_residual(args.rd_method):
                train_replay_reward += float(info.get("terminal_soc_penalty", 0.0))
            rarr_episode["reward_sum"] += rarr_info["reward"]
            rarr_episode["voltage_sum"] += rarr_info["r_voltage"]
            rarr_episode["soc_credit_sum"] += rarr_info["r_rarr"]
            rarr_episode["d_max"] = max(rarr_episode["d_max"], rarr_info["d_next"])
            rarr_episode["d_final"] = rarr_info["d_next"]
            rarr_episode["phi_final"] = rarr_info["phi_next"]
        elif stepsoc_rewarder is not None:
            stepsoc_info = stepsoc_rewarder.compute(
                soc_next=float(raw_next_state[1]),
                voltage_cost_t=float(info.get("quality", 0.0)),
            )
            train_replay_reward = stepsoc_info["reward"]
            if done and not uses_stepsoc_residual(args.rd_method):
                train_replay_reward += float(info.get("terminal_soc_penalty", 0.0))
            stepsoc_episode["reward_sum"] += stepsoc_info["reward"]
            stepsoc_episode["voltage_sum"] += stepsoc_info["r_voltage"]
            stepsoc_episode["soc_credit_sum"] += stepsoc_info["r_stepsoc"]
            stepsoc_episode["violation_sum"] += stepsoc_info["step_soc_deadband_violation"]
            if stepsoc_info["step_soc_deadband_violation"] > 0.0:
                stepsoc_episode["violation_count"] += 1.0

        # 收集轨迹（第一段原来缺失的部分）
        states.append(state)
        raw_states.append(raw_state)
        actions.append(action)
        next_states.append(next_state)
        raw_next_states.append(raw_next_state)
        rewards.append([train_replay_reward])
        dense_rewards.append([env_reward])
        dones.append([done_bool])
        if old_replay_buffer is not None:
            old_replay_buffer.add(
                state,
                action,
                log_prob,
                next_state,
                float(train_replay_reward),
                float(env_reward),
                done_bool,
                done,
            )

        episode_reward += env_reward
        update_episode_metrics(episode_metrics, info, env.slice)
        state = next_state
        raw_state = raw_next_state

        # 训练
        if t >= args.start_timesteps:
            total_loss_log = {}
            if args.policy == "PPO":
                if replay_buffer.size >= args.reward_train_start:
                    total_loss_log["reward_model_loss"] = policy.train_reward(replay_buffer, args.batch_size)
                if old_replay_buffer.size >= args.batch_size:
                    loss_log = policy.train(old_replay_buffer, replay_buffer, args.batch_size, total_steps=t)
                    old_replay_buffer.size = 0
                    loss_log["t"] = t
                    for key in loss_log:
                        if key not in total_loss_log:
                            total_loss_log[key] = 0
                        total_loss_log[key] += loss_log[key]
            elif replay_buffer.size >= args.batch_size:
                loss_log = policy.train(replay_buffer, args.batch_size)
                loss_log["t"] = t
                for key in loss_log:
                    if key not in total_loss_log:
                        total_loss_log[key] = 0
                    total_loss_log[key] += loss_log[key]
            if total_loss_log and t - last_train >= 100:
                wandb.log(total_loss_log)
                last_train = t

        # episode 结束
        if done:
            if args.sparse and args.env == 'HalfCheetah-v4':
                episode_reward = float(episode_reward > args.sparse_threshold) * 100

            replay_buffer.add_traj(
                np.array(raw_states if args.policy == "PPO" else states),
                np.array(actions),
                np.array(raw_next_states if args.policy == "PPO" else next_states),
                np.array(rewards),
                np.array(dense_rewards),
                np.array(dones),
                episode_reward,
                episode_timesteps,
                np.array(a_log_probs) if a_log_probs else None
            )

            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            episode_summary = finalize_episode_metrics(episode_metrics, episode_reward, episode_timesteps, env)
            wandb.log({
                "train_reward": episode_summary["reward"],
                "episode_steps": episode_summary["steps"],
                "t": t,
                "episodes": episode_num,
                "episode_quality_sum": episode_summary["quality_sum"],
                "episode_quality_mean": episode_summary["quality_mean"],
                "episode_voltage_deviation_mean_pu": episode_summary["voltage_deviation_mean_pu"],
                "episode_voltage_deviation_max_pu": episode_summary["voltage_deviation_max_pu"],
                "episode_voltage_violation_count": episode_summary["voltage_violation_count"],
                "episode_voltage_violation_rate": episode_summary["voltage_violation_rate"],
                "episode_voltage_violation_duration_steps": episode_summary["voltage_violation_duration_steps"],
                "episode_voltage_violation_duration_hours": episode_summary["voltage_violation_duration_hours"],
                "episode_voltage_violation_bus_steps": episode_summary["voltage_violation_bus_steps"],
                "episode_voltage_violation_bus_rate": episode_summary["voltage_violation_bus_rate"],
                "episode_grid_loss_mwh": episode_summary["grid_loss_mwh"],
                "episode_soc_penalty_sum": episode_summary["soc_penalty_sum"],
                "final_soc": episode_summary["final_soc"],
                "episode_terminal_soc_error": episode_summary["terminal_soc_error"],
                "episode_bess_throughput_mwh": episode_summary["bess_throughput_mwh"],
                "episode_equivalent_cycles": episode_summary["equivalent_cycles"],
                "episode_powerflow_nonconvergence_count": episode_summary["powerflow_nonconvergence_count"],
                "episode_powerflow_nonconvergence_rate": episode_summary["powerflow_nonconvergence_rate"],
                "rarr_reward_sum": rarr_episode["reward_sum"],
                "rarr_voltage_sum": rarr_episode["voltage_sum"],
                "rarr_soc_credit_sum": rarr_episode["soc_credit_sum"],
                "rarr_d_max": rarr_episode["d_max"],
                "rarr_d_final": rarr_episode["d_final"],
                "rarr_phi_final": rarr_episode["phi_final"],
                "stepsoc_reward_sum": stepsoc_episode["reward_sum"],
                "stepsoc_voltage_sum": stepsoc_episode["voltage_sum"],
                "stepsoc_soc_credit_sum": stepsoc_episode["soc_credit_sum"],
                "stepsoc_violation_sum": stepsoc_episode["violation_sum"],
                "stepsoc_violation_rate": stepsoc_episode["violation_count"] / max(episode_timesteps, 1),
            })

            state, _ = env.reset()
            raw_state = state.copy()
            if state_norm is not None and args.state_norm:
                state = state_norm(state)
            if reward_scaling is not None and args.reward_scale:
                reward_scaling.reset()
            done = False
            states, actions, next_states, rewards, dense_rewards, dones, a_log_probs = [], [], [], [], [], [], []
            raw_states, raw_next_states = [], []
            rarr_episode = {
                "reward_sum": 0.0,
                "voltage_sum": 0.0,
                "soc_credit_sum": 0.0,
                "d_max": 0.0,
                "d_final": 0.0,
                "phi_final": 0.0,
            }
            stepsoc_episode = {
                "reward_sum": 0.0,
                "voltage_sum": 0.0,
                "soc_credit_sum": 0.0,
                "violation_sum": 0.0,
                "violation_count": 0.0,
            }
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_metrics = init_episode_metrics(float(raw_state[1]))

        # 评估 & 保存
        if (t + 1) % args.eval_freq == 0:
            eval_summary = eval_policy(policy, args.env, args.seed, eval_episodes=args.eval_episodes, ctrl_cost_weight=args.ctrl_cost_weight)
            evaluation_history.append({"step": t + 1, **eval_summary})
            if args.pre_ex_metrics:
                pre_ex_row = build_pre_ex_eval_row(t + 1, eval_summary, episode_length=max_length)
                pre_ex_eval_rows.append(pre_ex_row)
                electric_metric_logger.append_row(pre_ex_row)
            else:
                electric_metric_logger.append(t + 1, eval_summary)
            wandb.log({
                "eval_reward": eval_summary["reward"],
                "eval_quality_mean": eval_summary["quality_mean"],
                "eval_voltage_deviation_mean_pu": eval_summary["voltage_deviation_mean_pu"],
                "eval_voltage_deviation_max_pu": eval_summary["voltage_deviation_max_pu"],
                "eval_voltage_violation_count": eval_summary["voltage_violation_count"],
                "eval_voltage_violation_rate": eval_summary["voltage_violation_rate"],
                "eval_voltage_violation_duration_steps": eval_summary["voltage_violation_duration_steps"],
                "eval_voltage_violation_duration_hours": eval_summary["voltage_violation_duration_hours"],
                "eval_voltage_violation_bus_steps": eval_summary["voltage_violation_bus_steps"],
                "eval_voltage_violation_bus_rate": eval_summary["voltage_violation_bus_rate"],
                "eval_grid_loss_mwh": eval_summary["grid_loss_mwh"],
                "eval_final_soc": eval_summary["final_soc"],
                "eval_terminal_soc_error": eval_summary["terminal_soc_error"],
                "eval_bess_throughput_mwh": eval_summary["bess_throughput_mwh"],
                "eval_equivalent_cycles": eval_summary["equivalent_cycles"],
                "eval_powerflow_nonconvergence_count": eval_summary["powerflow_nonconvergence_count"],
                "eval_powerflow_nonconvergence_rate": eval_summary["powerflow_nonconvergence_rate"],
                "t": t + 1,
            })
            if args.save_model:
                policy.save(os.path.join(args.model_dir, file_name))

    if args.save_data:
        replay_buffer.save_data(buffer_path)
    episodes_to_convergence = None
    if args.pre_ex_metrics:
        pre_ex_convergence_log = [
            {
                "episode": int(row["global_episode"]),
                "total_reward": row["total_reward"],
                "average_voltage_deviation": row["average_voltage_deviation"],
                "terminal_soc_error": row["terminal_soc_error"],
            }
            for row in pre_ex_eval_rows
        ]
        episodes_to_convergence = compute_episodes_to_convergence(
            pre_ex_convergence_log,
            metric_reward="total_reward",
            metric_avgvd="average_voltage_deviation",
            metric_soc="terminal_soc_error",
            episode_key="episode",
            max_episode=int(args.max_timesteps // max_length),
        )
        for row in pre_ex_eval_rows:
            row["episodes_to_convergence"] = episodes_to_convergence
        rewrite_pre_ex_eval_rows(electric_metric_logger.path, pre_ex_eval_rows)

    finalize_wandb_run(
        summary={
            "best_eval_reward": float(np.max([row["reward"] for row in evaluation_history])) if evaluation_history else None,
            "final_eval_reward": float(evaluation_history[-1]["reward"]) if evaluation_history else None,
            "best_eval_voltage_deviation_mean_pu": float(np.min([row["voltage_deviation_mean_pu"] for row in evaluation_history])) if evaluation_history else None,
            "final_eval_voltage_deviation_mean_pu": float(evaluation_history[-1]["voltage_deviation_mean_pu"]) if evaluation_history else None,
            "final_eval_grid_loss_mwh": float(evaluation_history[-1]["grid_loss_mwh"]) if evaluation_history else None,
            "final_eval_terminal_soc_error": float(evaluation_history[-1]["terminal_soc_error"]) if evaluation_history else None,
            "final_eval_powerflow_nonconvergence_rate": float(evaluation_history[-1]["powerflow_nonconvergence_rate"]) if evaluation_history else None,
            "episodes_to_convergence": episodes_to_convergence,
            "total_episodes": int(episode_num),
            "total_timesteps": int(args.max_timesteps),
        },
        file_paths=[electric_metric_logger.path, buffer_path if args.save_data else None],
        file_globs=[os.path.join(args.model_dir, f"{file_name}*")] if args.save_model else [],
    )

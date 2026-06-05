from dataclasses import dataclass

import numpy as np

from race.utils.metrics import aggregate_metrics
from race.envs.sparse_33bus_env import DIFFICULTIES, Sparse33BusEnv


def sanitize_name(value):
    invalid_chars = '<>:"/\\|?*'
    return "".join("_" if ch in invalid_chars else ch for ch in str(value))


def build_baseline_file_name(policy, env, rd_method, model_name, seed):
    return f"{policy}_{env}_{rd_method}_{sanitize_name(model_name)}_seed{seed}"


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
        averaged[key] = float(np.mean([float(metric_dict[key]) for metric_dict in metric_dicts]))
    return averaged


@dataclass
class BaseElectricBaseline:
    name: str

    def select_action(self, env, state):
        raise NotImplementedError


class NoControlBaseline(BaseElectricBaseline):
    def __init__(self):
        super().__init__(name="NoControl")

    def select_action(self, env, state):
        return np.array([0.0], dtype=np.float32)


class RuleBasedBESSBaseline(BaseElectricBaseline):
    def __init__(
        self,
        grid_gain=1.25,
        soc_weight=0.35,
        endgame_soc_weight=0.85,
        endgame_steps=12,
    ):
        super().__init__(name="RuleBasedBESS")
        self.grid_gain = grid_gain
        self.soc_weight = soc_weight
        self.endgame_soc_weight = endgame_soc_weight
        self.endgame_steps = endgame_steps

    def _soc_tracking_action(self, env, soc, remaining_steps):
        remaining_steps = max(int(remaining_steps), 1)
        required_power_mw = (env.target_soc - soc) * env.Battery_Capacity / (env.slice * remaining_steps)
        return float(np.clip(required_power_mw / env.max_bess_power_mw, -1.0, 1.0))

    def select_action(self, env, state):
        soc = float(state[1])
        load = float(state[2])
        pv = float(state[3])
        wt = float(state[4])
        remaining_steps = max(96 - env.t, 1)

        base_load_p_mw = float(getattr(env, "base_load_p_mw", 3.715))
        renewable_p_mw = float(getattr(env, "cap", 0.4)) * (pv + wt)
        net_load = base_load_p_mw * load - renewable_p_mw
        # In pandapower storage, positive p_mw behaves like charging demand,
        # so a high net load should push the controller toward negative action.
        grid_action = float(np.clip(-self.grid_gain * net_load / env.max_bess_power_mw, -1.0, 1.0))
        soc_action = self._soc_tracking_action(env, soc, remaining_steps)

        soc_weight = self.endgame_soc_weight if remaining_steps <= self.endgame_steps else self.soc_weight
        action = (1.0 - soc_weight) * grid_action + soc_weight * soc_action
        return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)


class _LinDistFlowBatteryModel:
    """Linearized voltage model for the IEEE 33-bus battery control task."""

    _cached = None

    def __init__(self, env):
        import pandapower.networks as pn

        net = pn.case33bw()
        self.num_buses = int(len(net.bus))
        self.root_bus = int(net.ext_grid.iloc[0]["bus"])
        self.base_mva = float(net.sn_mva)
        self.pv_bus = int(env.pv_bus)
        self.wt_bus = int(env.wt_bus)
        self.bess_bus = int(env.bess_bus)

        bus_vn_kv = net.bus["vn_kv"].to_numpy(dtype=float)
        adjacency = {bus_idx: [] for bus_idx in range(self.num_buses)}
        for _, line in net.line.iterrows():
            from_bus = int(line["from_bus"])
            to_bus = int(line["to_bus"])
            length_km = float(line["length_km"])
            r_ohm = float(line["r_ohm_per_km"]) * length_km
            x_ohm = float(line["x_ohm_per_km"]) * length_km
            z_base = float(bus_vn_kv[from_bus] ** 2 / self.base_mva)
            r_pu = r_ohm / z_base
            x_pu = x_ohm / z_base
            adjacency[from_bus].append((to_bus, r_pu, x_pu))
            adjacency[to_bus].append((from_bus, r_pu, x_pu))

        parent = {self.root_bus: None}
        parent_edge = {self.root_bus: (0.0, 0.0)}
        queue = [self.root_bus]
        while queue:
            node = queue.pop(0)
            for nxt, r_pu, x_pu in adjacency[node]:
                if nxt in parent:
                    continue
                parent[nxt] = node
                parent_edge[nxt] = (r_pu, x_pu)
                queue.append(nxt)

        self.bus_paths = {}
        self.edge_paths = {}
        for bus_idx in range(self.num_buses):
            buses = [bus_idx]
            edges = []
            cursor = bus_idx
            while parent[cursor] is not None:
                edges.append(parent_edge[cursor])
                cursor = parent[cursor]
                buses.append(cursor)
            self.bus_paths[bus_idx] = list(reversed(buses))
            self.edge_paths[bus_idx] = list(reversed(edges))

        base_load_p_mw = np.zeros(self.num_buses, dtype=float)
        base_load_q_mvar = np.zeros(self.num_buses, dtype=float)
        for _, load in net.load.iterrows():
            bus_idx = int(load["bus"])
            base_load_p_mw[bus_idx] += float(load["p_mw"])
            base_load_q_mvar[bus_idx] += float(load["q_mvar"])

        base_load_p_pu = base_load_p_mw / self.base_mva
        base_load_q_pu = base_load_q_mvar / self.base_mva

        shared_r = np.zeros((self.num_buses, self.num_buses), dtype=float)
        shared_x = np.zeros((self.num_buses, self.num_buses), dtype=float)
        for bus_i in range(self.num_buses):
            path_i = self.bus_paths[bus_i]
            edge_i = self.edge_paths[bus_i]
            for bus_m in range(self.num_buses):
                path_m = self.bus_paths[bus_m]
                edge_m = self.edge_paths[bus_m]
                shared_steps = min(len(edge_i), len(edge_m))
                common_r = 0.0
                common_x = 0.0
                for step_idx in range(shared_steps):
                    if path_i[step_idx + 1] != path_m[step_idx + 1]:
                        break
                    common_r += edge_i[step_idx][0]
                    common_x += edge_i[step_idx][1]
                shared_r[bus_i, bus_m] = common_r
                shared_x[bus_i, bus_m] = common_x

        self.load_voltage_coeff = -(
            shared_r @ base_load_p_pu + shared_x @ base_load_q_pu
        )
        self.pv_voltage_coeff = shared_r[:, self.pv_bus] / self.base_mva
        self.wt_voltage_coeff = shared_r[:, self.wt_bus] / self.base_mva
        # Positive storage power in the environment means charging demand.
        self.bess_voltage_coeff = -shared_r[:, self.bess_bus] / self.base_mva

    @classmethod
    def get(cls, env):
        if cls._cached is None:
            cls._cached = cls(env)
        return cls._cached


class StandardMPCBaseline(BaseElectricBaseline):
    """
    Standard receding-horizon MPC built on an explicit linearized network model.

    Decision variable: continuous BESS charging power sequence over the horizon.
    Dynamics: linear SOC evolution.
    Network model: LinDistFlow-style linear voltage approximation on the IEEE 33-bus feeder.
    Solver: linear programming via scipy.optimize.linprog (HiGHS).
    """

    def __init__(
        self,
        horizon=12,
        terminal_soc_weight=100.0,
        voltage_violation_penalty=200.0,
        fallback_controller=None,
    ):
        super().__init__(name="StandardMPC")
        self.horizon = int(horizon)
        self.terminal_soc_weight = float(terminal_soc_weight)
        self.voltage_violation_penalty = float(voltage_violation_penalty)
        self.fallback_controller = fallback_controller or RuleBasedBESSBaseline()
        self._network_model = None

    def _get_model(self, env):
        if self._network_model is None:
            self._network_model = _LinDistFlowBatteryModel.get(env)
        return self._network_model

    def _terminal_soc_reference(self, current_soc, target_soc, planning_steps, remaining_steps):
        if planning_steps >= remaining_steps:
            return float(target_soc)
        blend = float(planning_steps) / float(max(remaining_steps, 1))
        return float(current_soc + blend * (target_soc - current_soc))

    def _terminal_soc_interval(self, env, terminal_soc_ref):
        deadband = max(0.0, float(getattr(env, "soc_deadband", 0.0)))
        lower = max(float(env.soc_min), float(terminal_soc_ref) - deadband)
        upper = min(float(env.soc_max), float(terminal_soc_ref) + deadband)
        if lower > upper:
            clipped_ref = float(np.clip(terminal_soc_ref, env.soc_min, env.soc_max))
            return clipped_ref, clipped_ref
        return lower, upper

    def _build_forecasts(self, env, planning_steps):
        load_forecast = np.asarray(env.L[env.t: env.t + planning_steps], dtype=float)
        pv_forecast_mw = np.asarray(env.PV[env.t: env.t + planning_steps], dtype=float) * float(env.cap)
        wt_forecast_mw = np.asarray(env.WT[env.t: env.t + planning_steps], dtype=float) * float(env.cap)
        return load_forecast, pv_forecast_mw, wt_forecast_mw

    def _solve_mpc(self, env):
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix

        model = self._get_model(env)
        remaining_steps = max(96 - env.t, 0)
        planning_steps = min(self.horizon, remaining_steps)
        if planning_steps <= 0:
            return None

        load_forecast, pv_forecast_mw, wt_forecast_mw = self._build_forecasts(env, planning_steps)

        current_soc = float(env.state[0][1])
        terminal_soc_ref = self._terminal_soc_reference(
            current_soc=current_soc,
            target_soc=float(env.target_soc),
            planning_steps=planning_steps,
            remaining_steps=remaining_steps,
        )
        terminal_soc_low, terminal_soc_high = self._terminal_soc_interval(env, terminal_soc_ref)

        num_buses = int(model.num_buses)
        h = int(planning_steps)
        alpha = float(env.slice / env.Battery_Capacity)

        p_offset = 0
        soc_offset = p_offset + h
        xi_offset = soc_offset + h
        nu_offset = xi_offset + h * num_buses
        z_idx = nu_offset + h * num_buses
        num_vars = z_idx + 1

        objective = np.zeros(num_vars, dtype=float)
        objective[xi_offset:nu_offset] = 1.0 / (num_buses * float(env.voltage_reward_span))
        objective[nu_offset:z_idx] = self.voltage_violation_penalty
        objective[z_idx] = self.terminal_soc_weight

        bounds = []
        for _ in range(h):
            bounds.append((-float(env.max_bess_power_mw), float(env.max_bess_power_mw)))
        for _ in range(h):
            bounds.append((float(env.soc_min), float(env.soc_max)))
        for _ in range(h * num_buses):
            bounds.append((0.0, None))
        for _ in range(h * num_buses):
            bounds.append((0.0, None))
        bounds.append((0.0, None))

        a_eq = lil_matrix((h, num_vars), dtype=float)
        b_eq = np.zeros(h, dtype=float)
        for k in range(h):
            a_eq[k, soc_offset + k] = 1.0
            a_eq[k, p_offset + k] = -alpha
            if k == 0:
                b_eq[k] = current_soc
            else:
                a_eq[k, soc_offset + k - 1] = -1.0
                b_eq[k] = 0.0

        num_ineq = 4 * h * num_buses + 2
        a_ub = lil_matrix((num_ineq, num_vars), dtype=float)
        b_ub = np.zeros(num_ineq, dtype=float)
        row = 0

        for k in range(h):
            voltage_base = (
                1.0
                + model.load_voltage_coeff * load_forecast[k]
                + model.pv_voltage_coeff * pv_forecast_mw[k]
                + model.wt_voltage_coeff * wt_forecast_mw[k]
            )
            for bus_idx in range(num_buses):
                xi_idx = xi_offset + k * num_buses + bus_idx
                nu_idx = nu_offset + k * num_buses + bus_idx
                p_coeff = float(model.bess_voltage_coeff[bus_idx])
                base_value = float(voltage_base[bus_idx])

                # xi >= v - 1
                a_ub[row, p_offset + k] = p_coeff
                a_ub[row, xi_idx] = -1.0
                b_ub[row] = 1.0 - base_value
                row += 1

                # xi >= 1 - v
                a_ub[row, p_offset + k] = -p_coeff
                a_ub[row, xi_idx] = -1.0
                b_ub[row] = base_value - 1.0
                row += 1

                # v <= upper + nu
                a_ub[row, p_offset + k] = p_coeff
                a_ub[row, nu_idx] = -1.0
                b_ub[row] = float(env.voltage_upper_bound) - base_value
                row += 1

                # v >= lower - nu
                a_ub[row, p_offset + k] = -p_coeff
                a_ub[row, nu_idx] = -1.0
                b_ub[row] = base_value - float(env.voltage_lower_bound)
                row += 1

        # z >= s_H - s_high: penalize only SOC above the target deadband.
        a_ub[row, soc_offset + h - 1] = 1.0
        a_ub[row, z_idx] = -1.0
        b_ub[row] = terminal_soc_high
        row += 1

        # z >= s_low - s_H: penalize only SOC below the target deadband.
        a_ub[row, soc_offset + h - 1] = -1.0
        a_ub[row, z_idx] = -1.0
        b_ub[row] = -terminal_soc_low

        result = linprog(
            c=objective,
            A_ub=a_ub.tocsr(),
            b_ub=b_ub,
            A_eq=a_eq.tocsr(),
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return None
        return result.x

    def select_action(self, env, state):
        solution = self._solve_mpc(env)
        if solution is None:
            return self.fallback_controller.select_action(env, state)

        requested_power_mw = float(solution[0])
        normalized_action = requested_power_mw / float(env.max_bess_power_mw)
        return np.array([np.clip(normalized_action, -1.0, 1.0)], dtype=np.float32)


class StandardMPCPersistenceBaseline(StandardMPCBaseline):
    """Standard MPC with persistence forecast: current exogenous inputs repeated into the future."""

    def __init__(
        self,
        horizon=12,
        terminal_soc_weight=100.0,
        voltage_violation_penalty=200.0,
        fallback_controller=None,
    ):
        super().__init__(
            horizon=horizon,
            terminal_soc_weight=terminal_soc_weight,
            voltage_violation_penalty=voltage_violation_penalty,
            fallback_controller=fallback_controller,
        )
        self.name = "StandardMPCPersistence"

    def _build_forecasts(self, env, planning_steps):
        current_load = float(env.state[0][2])
        current_pv_mw = float(env.state[0][3]) * float(env.cap)
        current_wt_mw = float(env.state[0][4]) * float(env.cap)
        load_forecast = np.full(int(planning_steps), current_load, dtype=float)
        pv_forecast_mw = np.full(int(planning_steps), current_pv_mw, dtype=float)
        wt_forecast_mw = np.full(int(planning_steps), current_wt_mw, dtype=float)
        return load_forecast, pv_forecast_mw, wt_forecast_mw


class StandardMPCTrueForecastBaseline(StandardMPCBaseline):
    """Standard MPC with the true remaining same-day L/PV/WT trajectory as forecast."""

    def __init__(
        self,
        terminal_soc_weight=100.0,
        voltage_violation_penalty=200.0,
        fallback_controller=None,
    ):
        super().__init__(
            horizon=96,
            terminal_soc_weight=terminal_soc_weight,
            voltage_violation_penalty=voltage_violation_penalty,
            fallback_controller=fallback_controller,
        )
        self.name = "StandardMPCTrueForecast"

    def _solve_mpc(self, env):
        # Use the whole remaining day instead of a fixed short horizon.
        self.horizon = max(96 - int(env.t), 1)
        return super()._solve_mpc(env)


AVAILABLE_BASELINES = {
    "NoControl": NoControlBaseline,
    "RuleBasedBESS": RuleBasedBESSBaseline,
    "StandardMPC": StandardMPCBaseline,
    "StandardMPCPersistence": StandardMPCPersistenceBaseline,
    "StandardMPCTrueForecast": StandardMPCTrueForecastBaseline,
    # Backward-compatible alias: all persistence MPC runs now use the standard
    # continuous LP formulation, not the removed rollout/grid-search baseline.
    "MPCPersistence": StandardMPCPersistenceBaseline,
    "MPCTrueForecast": StandardMPCTrueForecastBaseline,
}


def build_baseline_controller(name, **kwargs):
    if name not in AVAILABLE_BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    return AVAILABLE_BASELINES[name](**kwargs)


def evaluate_baseline_on_sparse_tests(controller, seed=0):
    difficulty_summaries = {}
    all_episode_metrics = []
    for difficulty in DIFFICULTIES:
        env = Sparse33BusEnv(seed=seed)
        episode_summaries = []
        for scenario_id in env.dataset.get_ids("test", difficulty):
            state, _ = env.reset(scenario_id=scenario_id, split="test", difficulty=difficulty)
            done = False

            while not done:
                action = controller.select_action(env, state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            metrics = env.get_episode_metrics()
            episode_summaries.append(metrics)
            all_episode_metrics.append(metrics)

        difficulty_summaries[difficulty] = aggregate_metrics(episode_summaries)
    difficulty_summaries["all"] = aggregate_metrics(all_episode_metrics)
    return difficulty_summaries


def evaluate_baseline(controller, eval_episodes=1, env=None, scenario_ids=None, split="test", difficulty=None):
    episode_summaries = []
    if env is None:
        env = Sparse33BusEnv()

    if scenario_ids is None:
        if difficulty is None:
            scenario_ids = env.dataset.get_ids(split)
        else:
            scenario_ids = env.dataset.get_ids(split, difficulty)

    for scenario_id in list(scenario_ids)[: int(eval_episodes) if eval_episodes else None]:
        metadata = env.dataset.get_metadata(int(scenario_id))
        state, _ = env.reset(
            scenario_id=int(scenario_id),
            split=metadata["split"],
            difficulty=metadata["difficulty_label"],
        )
        done = False

        while not done:
            action = controller.select_action(env, state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        episode_summaries.append(env.get_episode_metrics())

    return aggregate_metrics(episode_summaries)

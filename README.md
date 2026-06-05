# RACE: Recoverability-Aware Credit Evaluator for Sparse BESS Control

This repository contains the code for sparse-reward reinforcement learning in
battery energy storage system (BESS) control on a 33-bus distribution-network
environment.

## Core Idea

RACE adds a recoverability-aware physical credit signal to reward
decomposition. The evaluator checks whether the post-action SOC can still
return to the terminal target deadband within the remaining control horizon.
The current implementation combines this explicit recoverability credit with
VIB-style residual reward decomposition.

## Repository Structure

```text
race/
  envs/        33-bus BESS environments and power-flow wrappers
  algos/       SAC, TD3, DDPG, PPO implementations
  rewards/     RD, RRD, VIB, Diaster, and RACE reward modules
  baselines/   NoControl, RuleBasedBESS, MPC, and oracle-style baselines
  training/    Replay buffer and training utilities
  utils/       Metrics, logging, random seeds, and W&B helpers
scripts/       Command-line entry points
configs/       Reproducible experiment settings
figures/       Plotting scripts for paper figures
data/          Public scenario CSV files
tests/         Lightweight smoke tests
```

Generated `results/` are intentionally not included. They are produced when
users run the training, baseline, aggregation, or plotting scripts.

## Installation

```bash
cd D:/Project/RACE
python -m pip install -r requirements.txt
python -m pip install -e .
```

The original experiments were developed with CUDA-enabled PyTorch. Install the
PyTorch build that matches your CUDA runtime if GPU training is required.

## Running Experiments

### Quick Smoke Test

Run a short RACE-VIB SAC experiment:

```bash
python scripts/train_sparse_33bus.py --policy SAC --rd_method RARR_VIB --episodes 10 --eval_freq 5 --seed 1
```

This command is only intended to check that the environment, power-flow
simulation, RL policy, reward decomposer, and metric logger can run end to end.

### Main Experiments

Run the main SAC + RACE-VIB experiment:

```bash
python experiments/main/run_sac_race_vib.py
```

Run the SAC + VIB baseline used to isolate the evaluator contribution:

```bash
python experiments/main/run_sac_vib.py
```

Run the evaluator-sensitivity study over RD, RRD, Diaster, VIB and their
RACE-enhanced variants:

```bash
python experiments/ablation/run_evaluator_sensitivity.py
```

Run single-day pre-experiments:

```bash
python experiments/pre_ex/run_single_day_sparse.py
python experiments/pre_ex/run_single_day_race.py
```

### Electrical Baselines

Run the default electrical baselines:

```bash
python scripts/run_baselines.py --baselines all --max_timesteps 9600 --eval_freq 96
```

Run a specific subset:

```bash
python scripts/run_baselines.py --baselines NoControl,RuleBasedBESS,StandardMPCPersistence --max_timesteps 9600
```

### Result Aggregation and Plotting

Aggregate 9-day test results after the training and baseline runs finish:

```bash
python scripts/aggregate_selected_9day_results.py
```

Plot pre-experiment learning curves:

```bash
python figures/plot_learning_curves.py
```

Plot final boxplots:

```bash
python figures/plot_final_boxplots.py
```

All generated metrics, tables, and figures are written under `result/` or
figure-specific output folders. These generated outputs are not tracked by the
repository.

## Main Methods

- `None`: sparse episode-return training without reward decomposition.
- `RD`, `RRD`, `VIB`, `Diaster`: reward decomposition baselines.
- `RARR_VIB`: RACE-enhanced VIB residual credit decomposition.
- `RARR_RD`, `RARR_RRD`, `RARR_Diaster`: RACE-enhanced residual variants.

The term `RARR` is retained in code identifiers for backward compatibility.
In paper figures and text, the method is referred to as `RACE`.

## Data

The repository includes the scenario files:

- `data/easy.csv`
- `data/medium.csv`
- `data/hard.csv`

These are sufficient for running the sparse 33-bus experiments.

For broader distributed renewable energy scenarios on the IEEE 33-bus network,
see the public DDRE-33 dataset:

- Dataset: https://figshare.com/articles/dataset/DDRE-33_Dataset_for_Distributed_Renewable_Energy_Scenarios/29374640
- Code repository: https://github.com/YuxuanCEE/DDRE-33-CHIME

## Citation

......

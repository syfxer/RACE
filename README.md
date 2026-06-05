# RACE: Recoverability-Aware Credit Evaluator for Sparse BESS Control

This repository contains the code used for the paper draft in `docs/paper/`.
The project studies sparse-reward reinforcement learning for battery energy
storage system (BESS) control on a 33-bus distribution-network environment.

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
docs/          Method notes and paper draft
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

## Quick Start

Run a short RACE-VIB SAC experiment:

```bash
python scripts/train_sparse_33bus.py --policy SAC --rd_method RARR_VIB --episodes 10 --eval_freq 5 --seed 1
```

Run electrical baselines:

```bash
python scripts/run_baselines.py --baselines NoControl RuleBasedBESS MPC_true_forecast --max_timesteps 9600
```

Plot pre-experiment learning curves after results are generated:

```bash
python figures/plot_learning_curves.py
```

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

## Citation

Please cite the associated paper if you use this code. A BibTeX entry can be
added here after publication.

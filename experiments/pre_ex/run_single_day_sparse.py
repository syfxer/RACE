"""Run a short single-day sparse-reward pre-experiment."""

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "scripts/train_sparse_33bus.py",
        "--env",
        "SparseHardcodedDayEnv",
        "--policy",
        "SAC",
        "--rd_method",
        "None",
        "--episodes",
        "200",
        "--seed",
        "40",
        "--wandb_mode",
        "disabled",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

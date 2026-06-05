"""Run the main SAC + RACE-VIB experiment for one seed."""

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "scripts/train_sparse_33bus.py",
        "--policy",
        "SAC",
        "--rd_method",
        "RARR_VIB",
        "--episodes",
        "1200",
        "--batch_size",
        "128",
        "--start_timesteps",
        "960",
        "--reward_train_start",
        "512",
        "--eval_freq",
        "20",
        "--wandb_mode",
        "disabled",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

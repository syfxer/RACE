"""Run evaluator sensitivity experiments for SAC reward decomposers."""

import subprocess
import sys


METHODS = ["RD", "RRD", "Diaster", "VIB", "RARR_RD", "RARR_RRD", "RARR_Diaster", "RARR_VIB"]


def main() -> None:
    for method in METHODS:
        cmd = [
            sys.executable,
            "scripts/train_sparse_33bus.py",
            "--policy",
            "SAC",
            "--rd_method",
            method,
            "--episodes",
            "1200",
            "--wandb_mode",
            "disabled",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

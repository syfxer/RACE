"""Placeholder evaluation entry point.

Evaluation is currently performed during training and by the baseline scripts.
This file is kept as a stable public entry point for future standalone
checkpoint evaluation.
"""


def main() -> None:
    raise SystemExit(
        "Standalone checkpoint evaluation is not implemented yet. "
        "Use scripts/train_sparse_33bus.py with --eval_freq or scripts/run_baselines.py."
    )


if __name__ == "__main__":
    main()

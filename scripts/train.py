"""Convenience wrapper for sparse 33-bus training.

This wrapper intentionally keeps the command-line interface of
`train_sparse_33bus.py` so users can run either script.
"""

from scripts.train_sparse_33bus import main


if __name__ == "__main__":
    main()

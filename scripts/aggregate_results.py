"""Aggregate paper result tables.

This delegates to the selected 9-day aggregation script retained from the
experiment code.
"""

from scripts.aggregate_selected_9day_results import main


if __name__ == "__main__":
    main()

# Final analysis entry points

- `run_final910_h00_matched_baseline_completion.py` completes matched baselines for the frozen 910-site H00 experiment.
- `run_frozen_quick_wins.py` performs post hoc analyses using frozen predictions only; it does not retrain models.
- `run_supervisor_final_three_checks.py` runs the final persistence, normalised-error, and hot-moment coupling checks.

These scripts originally lived inside versioned output directories. They were moved here and changed to resolve the project root relative to this repository. The scientific checks and frozen assumptions are implemented directly in the scripts and retained configuration files.

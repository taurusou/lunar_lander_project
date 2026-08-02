# Results

Store compact, reproducible outputs here:

- `metrics/` for CSV or JSON experiment summaries
- `plots/` for report-ready figures
- `checkpoints/` for local model files ignored by Git

Record the configuration and seed associated with every result.

The controlled runner writes one row per model to
`metrics/final_experiment_summary.csv`. Smoke-test metrics and plots are local
pipeline checks ignored by Git and must not be used in the final report.

The completed schedule-level outputs are:

- `metrics/final_schedule_summary.csv`
- `plots/final_schedule_comparison.png`

Their calculation and interpretation are documented in
[`docs/final-results.md`](../docs/final-results.md).

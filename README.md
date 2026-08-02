# LunarLander Reinforcement Learning Project

This repository will study reinforcement learning in Gymnasium's
`LunarLander-v3` environment.

## Proposed research question

How does the exploration schedule affect the performance and reliability of a
DQN agent in `LunarLander-v3`?

## Current status

The repository structure, random baseline, DQN implementation, controlled
experiment, unseen-seed evaluation, and final schedule analysis are complete.
The report and recorded demonstration remain to be finished.

## Local setup

See [`docs/setup.md`](docs/setup.md) for beginner-oriented Windows setup and
verification instructions.

After installing the dependencies, run:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\environment_check.py
```

## Random baseline

Run the reproducible 100-episode baseline with:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\random_baseline.py
```

Using seeds 0 through 99 produced:

| Measurement | Result |
| --- | ---: |
| Mean reward | -191.00 |
| Median reward | -167.31 |
| Standard deviation | 107.46 |
| Minimum reward | -428.73 |
| Maximum reward | 51.35 |
| Episodes scoring at least 200 | 0 of 100 |

The raw episode data is in
[`results/metrics/random_baseline.csv`](results/metrics/random_baseline.csv).
The generated figures are in [`results/plots`](results/plots).

## DQN experiment design

The complete plan and explanation are in
[`docs/dqn-experiment-design.md`](docs/dqn-experiment-design.md). Validate the
machine-readable configuration with:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\check_dqn_config.py
```

## DQN component tests

The Q-network and replay memory are explained in
[`docs/dqn-components.md`](docs/dqn-components.md). Run their automated tests
with:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

## DQN development training

The training process is explained in
[`docs/dqn-training.md`](docs/dqn-training.md). Run the 25-episode development
check with:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\train_dqn.py
```

This short run verifies the pipeline and must not be treated as the final
performance experiment.

The completed development run collected 3,654 transitions over 25 episodes,
began optimization in episode 11, and produced a mean reward of -118.88. No
episode reached the 200-point solved threshold.

## DQN development evaluation

The evaluation process is explained in
[`docs/dqn-evaluation.md`](docs/dqn-evaluation.md). Evaluate the local
development checkpoint with greedy actions on unseen seeds:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\evaluate_dqn.py
```

This development evaluation verifies the pipeline and is not a final
performance comparison.

The 25 unseen-seed episodes produced a mean reward of -92.34 and no episode
reached the 200-point solved threshold. All episodes reached the 1,000-step time
limit, so this early policy has not learned to complete the landing task.

## Planned comparison

- Random-action baseline
- DQN with a quickly decreasing exploration rate
- DQN with a gradually decreasing exploration rate

Each trained agent should be evaluated with the same episode budget and on
seeds that were not used during training.

## Controlled experiment runner

See [`docs/controlled-experiments.md`](docs/controlled-experiments.md) for the
safe, step-by-step commands. Preview the six final runs without starting them:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\run_controlled_experiments.py
```

The full six-model experiment is complete. The runner safely resumed after its
first execution window ended and did not repeat the four completed models.

## Final result

See [`docs/final-results.md`](docs/final-results.md) for the complete results
and cautious interpretation. Across 300 evaluation episodes per schedule:

| Measurement | Fast decay | Gradual decay |
| --- | ---: | ---: |
| Mean reward | 238.77 | 192.94 |
| Solved percentage | 86.00% | 60.67% |
| Standard deviation between seed means | 14.55 | 78.39 |

Under this experiment's fixed settings, fast decay was stronger and more
consistent across training seeds.

Recreate the schedule-level table and comparison plot with:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\analyze_final_results.py
```

## Repository layout

```text
configs/                  Experiment settings
docs/                     Project plan and working notes
notebooks/                Exploration and analysis notebooks
references/               Documentation and source links
report/                   Final report material
src/lunar_lander_rl/      Reusable project source code
tests/                    Small correctness checks
results/                  Metrics, plots, and ignored checkpoints
artifacts/videos/         Local demo recordings (ignored by Git)
```

## Main deliverables

- A working, commented repository or notebook
- A demo video or GIF
- A report explaining the approach, experiments, results, conclusions, and
  references

## Course-material note

The supplied CartPole notebook is a learning reference only. CartPole will not
be used as the final project environment.

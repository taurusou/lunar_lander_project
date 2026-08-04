# LunarLander Reinforcement Learning Project

This project studies reinforcement learning in Gymnasium's
`LunarLander-v3` environment.

## Research Question

How the exploration schedule affects the performance and reliability of a
Deep Q-Network (DQN) agent in `LunarLander-v3`?

## Project Status

The project is complete. It includes:

- A random-action baseline
- A DQN agent
- Fast and gradual epsilon-decay experiments
- Three training seeds for each schedule
- Evaluation on 100 unseen seeds per trained model
- Final metrics and plots
- A final report
- A recorded demonstration

## Start Here

The main files are:

- [`src/lunar_lander_rl/dqn_components.py`](src/lunar_lander_rl/dqn_components.py) — neural network and replay memory
- [`src/lunar_lander_rl/train_dqn.py`](src/lunar_lander_rl/train_dqn.py) — DQN training
- [`src/lunar_lander_rl/evaluate_dqn.py`](src/lunar_lander_rl/evaluate_dqn.py) — trained-model evaluation
- [`src/lunar_lander_rl/analyze_final_results.py`](src/lunar_lander_rl/analyze_final_results.py) — final analysis and comparison plot
- [`src/lunar_lander_rl/record_demo.py`](src/lunar_lander_rl/record_demo.py) — demonstration recording
- [`report/final_report.md`](report/final_report.md) — complete project report

## Setup

See [`docs/setup.md`](docs/setup.md) for beginner-friendly Windows setup steps.

After installing the dependencies, check the environment with:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\environment_check.py
```

## Main Commands

Run the random baseline:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\random_baseline.py
```

Run the automated tests:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Run the short development training check:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\train_dqn.py
```

Preview or run the controlled six-model experiment:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\run_controlled_experiments.py
```

Recreate the final comparison:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\analyze_final_results.py
```

Record a demonstration using the included checkpoint:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\record_demo.py
```

## Experiment Design

The experiment compared:

- DQN with fast epsilon decay (`0.98`)
- DQN with gradual epsilon decay (`0.995`)

Both schedules used the same neural network, optimizer, replay memory, training
budget, training seeds, and evaluation seeds. The epsilon-decay factor was the
main changed variable.

Each schedule was trained with seeds 0, 1, and 2 for 800 episodes per seed.
Each trained model was evaluated for 100 episodes using unseen seeds 1000–1099.

## Final Results

| Measurement | Fast decay | Gradual decay |
|---|---:|---:|
| Mean reward | 238.77 | 192.94 |
| Solved percentage | 86.00% | 60.67% |
| Standard deviation between seed means | 14.55 | 78.39 |

Under the fixed settings used in this project, fast epsilon decay produced
stronger and more consistent results.

The random-action baseline had a mean reward of `-191.00` and solved 0 of 100
episodes.

For the complete discussion and limitations, see the
[final report](report/final_report.md).

## Final Deliverables

- [Final report](report/final_report.md)
- [Final comparison plot](results/plots/final_schedule_comparison.png)
- [Demonstration video](https://youtu.be/UTzQmFBCSVw)
- [Project source code](src/lunar_lander_rl)

## Repository Layout

```text
configs/                  Experiment settings
docs/                     Setup and supporting explanations
references/               Source and documentation links
report/                   Final project report
src/lunar_lander_rl/      Main Python source code
tests/                    Automated correctness checks
results/metrics/          Experiment data and summaries
results/plots/            Training, evaluation, and comparison plots
results/checkpoints/      Selected checkpoint used by the demo
```

## Course-Material Note

The supplied CartPole notebook was used only as a learning reference.
`LunarLander-v3`, not CartPole, was used as the final project environment.

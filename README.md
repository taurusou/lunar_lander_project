# LunarLander Reinforcement Learning Project

This repository studies reinforcement learning in Gymnasium's
`LunarLander-v3` environment.

## Proposed research question

How does the exploration schedule affect the performance and reliability of a
DQN agent in `LunarLander-v3`?

## Current status

The project is complete.

The repository includes:

- A random-action baseline
- A Deep Q-Network implementation
- Fast and gradual epsilon-decay experiments
- Six trained models using three training seeds per schedule
- Evaluation on 100 unseen seeds per model
- Final result tables and plots
- A written final report
- A recorded agent demonstration

## Start Here

The main files for understanding this project are:

- [`src/lunar_lander_rl/dqn_components.py`](src/lunar_lander_rl/dqn_components.py)  
  Contains the neural network and replay memory.

- [`src/lunar_lander_rl/train_dqn.py`](src/lunar_lander_rl/train_dqn.py)  
  Contains the main DQN training process.

- [`src/lunar_lander_rl/evaluate_dqn.py`](src/lunar_lander_rl/evaluate_dqn.py)  
  Evaluates trained models on unseen seeds.

- [`src/lunar_lander_rl/analyze_final_results.py`](src/lunar_lander_rl/analyze_final_results.py)  
  Creates the final comparison table and plot.

- [`report/final_report.md`](report/final_report.md)  
  Contains the complete project explanation, results, discussion, and conclusion.

The other configuration, testing, and experiment-management files support
reproducibility but are not required to understand the main DQN algorithm.


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

## Automated Tests

The Q-network and replay memory are explained in
[`docs/dqn-components.md`](docs/dqn-components.md).

The automated tests check the main DQN components, training process,
evaluation process, experiment runner, and final-result analysis.

Run all tests with:

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

## Experiment Comparison

The completed experiment compared:

- A random-action baseline
- DQN with a quickly decreasing exploration rate
- DQN with a gradually decreasing exploration rate

Both DQN schedules used the same neural network, optimizer, replay-memory
settings, training budget, and evaluation process. The epsilon-decay factor
was the main variable being tested.

Each trained model was evaluated using the same 100 unseen seeds.

## Controlled Experiment Runner

The complete experiment trained six models:

- Three models using fast epsilon decay
- Three models using gradual epsilon decay

See [`docs/controlled-experiments.md`](docs/controlled-experiments.md) for the
step-by-step commands.

Preview the planned runs without starting training:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\run_controlled_experiments.py
```

The six-model experiment is complete. When the first execution window ended, the runner recognized the four completed models and continued with the remaining two instead of repeating finished runs.

## Final Results

Each epsilon schedule was trained with three random seeds. Every trained model
was then evaluated for 100 episodes using unseen seeds from 1000 to 1099.

| Measurement | Fast decay | Gradual decay |
|---|---:|---:|
| Mean reward | 238.77 | 192.94 |
| Solved percentage | 86.00% | 60.67% |
| Standard deviation between seed means | 14.55 | 78.39 |

Under the fixed settings used in this project, fast epsilon decay produced
stronger and more consistent results.

For the complete analysis and limitations, see the
[final report](report/final_report.md).

## Final Deliverables

- [Final report](report/final_report.md)
- [Final comparison plot](results/plots/final_schedule_comparison.png)
- [Demonstration video](https://youtu.be/UTzQmFBCSVw)
- [Project source code](src/lunar_lander_rl)

## Repository layout

```text
configs/                  Experiment settings
docs/                     Project plan and supporting documentation
notebooks/                Exploration and analysis notebooks
references/               Documentation and source links
report/                   Final project report
src/lunar_lander_rl/      Main project source code
tests/                    Automated correctness checks
results/                  Experiment metrics, plots, and generated outputs
artifacts/videos/         Local demo recordings ignored by Git
```

## Course-material note

The CartPole notebook was used only as a learning reference.
CartPole was not used as the final project environment, following the project requirement.

# LunarLander Reinforcement Learning Project

This repository will study reinforcement learning in Gymnasium's
`LunarLander-v3` environment.

## Proposed research question

How does the exploration schedule affect the performance and reliability of a
DQN agent in `LunarLander-v3`?

## Current status

The repository structure and experiment plan have been created. The local
environment check uses random actions to verify Gymnasium and LunarLander. No
learning agent or training implementation has been added yet.

## Local setup

See [`docs/setup.md`](docs/setup.md) for beginner-oriented Windows setup and
verification instructions.

After installing the dependencies, run:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\environment_check.py
```

## Planned comparison

- Random-action baseline
- DQN with a quickly decreasing exploration rate
- DQN with a gradually decreasing exploration rate

Each trained agent should be evaluated with the same episode budget and on
seeds that were not used during training.

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

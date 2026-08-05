# Controlled experiment runner

The controlled experiment runner applies the same training and evaluation steps to every epsilon schedule and training seed. This reduces manual work and
helps prevent accidentally giving one model different settings.

## Step 1: preview the final plan

Run the command without a flag:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\run_controlled_experiments.py
```

It prints the six planned models and exits without training anything:

```text
2 epsilon schedules x 3 training seeds = 6 models
```

Each final model uses 800 training episodes and 100 greedy evaluation episodes.
All models use evaluation seeds 1000 through 1099.

## Step 2: run the small smoke test

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\run_controlled_experiments.py --smoke-test
```

The smoke test runs both schedules, but uses only one training seed, two training episodes, and two evaluation episodes. It checks that the loop,
checkpoints, evaluation, and comparison table work together.

The smoke-test rewards are not research results. Two episodes are too few for DQN learning, and both schedules still have epsilon very close to 1.0.
Smoke-test files are local checks and Git will ignore them.

## Step 3: start the final experiment only when ready

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\run_controlled_experiments.py --run-final
```

This command trains all six models, so it may take a long time on a CPU. Keep the computer awake and connected to power. Please don't edit the configuration after
starting because every model must use the same shared settings.

## Safe restart behavior

For each model, the runner expects five files:

- Training metrics CSV
- Training reward plot
- Local checkpoint
- Evaluation metrics CSV
- Evaluation reward plot

If all five exist, the runner loads that completed result and moves on. If all three training files exist but evaluation files do not, it continues from the saved checkpoint without retraining. If only part of either group exists, it stops and asks for inspecting the files instead of silently overwriting them.

After each completed model, the runner updates:

```text
results/metrics/final_experiment_summary.csv
```

That table will contain the mean, median, standard deviation, minimum, maximum,
solved percentage, and training time for each model.

## Important checkpoint note

PyTorch checkpoint files can be large, so I let Git ignores them. Keep them locally until the report, plots, and demo have been completed. 

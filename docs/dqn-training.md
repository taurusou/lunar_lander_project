# DQN training loop

The first training run is a 25-episode development check. It is intentionally too short to judge whether DQN solves LunarLander.

## Training sequence

For each episode:

1. Calculate epsilon from the selected exploration schedule.
2. Reset LunarLander with a reproducible environment seed.
3. Choose each action with the epsilon-greedy policy.
4. Store the resulting transition in replay memory.
5. Once memory has 1,000 transitions, sample batches of 128.
6. Calculate predicted and target Q-values.
7. Minimize Smooth L1 loss with AdamW.
8. Clip unusually large gradients.
9. Soft-update the target network.
10. Save episode metrics.

## Epsilon-greedy actions

The policy generates a random number between 0 and 1:

- If the number is below epsilon, choose a random action.
- Otherwise, choose the action with the largest policy-network Q-value.

This balances exploration and exploitation.

## DQN target

The training target has two parts:

```text
target = immediate reward + discounted best future Q-value
```

For a naturally terminated transition, the future part becomes zero. A
time-limit truncation remains separate and is not automatically treated as a
natural terminal state.

## Two networks

- The policy network selects actions and is changed by the optimizer.
- The target network produces more stable future-value estimates.

After each environment step, the target parameters move 0.5% of the distance
toward the policy parameters because `tau` is `0.005`.

## Development command

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\train_dqn.py
```

The run uses:

- Experiment: `fast_epsilon_decay`
- Training seed: 0
- Episodes: 25
- Device: CPU

It creates:

```text
results/metrics/development_fast_epsilon_decay_seed_0.csv
results/plots/development_fast_epsilon_decay_seed_0_rewards.png
results/checkpoints/development_fast_epsilon_decay_seed_0.pt
```

Git will ignore model checkpoints.

## Important interpretation

The development run answers, “Does the training pipeline work?” It does not answer, “How well does the agent learn?” Final conclusions require 800 episodes,
both epsilon schedules, three training seeds, and separate evaluation.

## Development-run result

The configured run completed successfully with:

| Measurement | Result |
| --- | ---: |
| Episodes | 25 |
| Mean reward | -118.88 |
| Minimum reward | -352.26 |
| Maximum reward | 45.87 |
| Episodes scoring at least 200 | 0 |
| Final replay-memory size | 3,654 |
| First episode containing optimization loss | 11 |
| Final epsilon | 0.6158 |

The moving average increased during this short run, but no episode reached the solved threshold. This result confirms that experience collection, optimization,
target updates, metrics, plotting, and checkpoint saving work. It is not evidence that the final agent is solved or that one epsilon schedule is better.

## Reference

- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

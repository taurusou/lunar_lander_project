# DQN evaluation

Evaluation measures a saved policy without allowing it to explore or learn.

## Separation from training

During evaluation:

- Epsilon is 0, so every action uses the largest predicted Q-value.
- No replay memory is created.
- No loss is calculated.
- No optimizer step occurs.
- No target-network update occurs.
- Model parameters are copied before evaluation and checked afterward.

This isolates what the policy learned during training.

## Unseen seeds

Training seed 0 used environment seeds beginning at 0. Development evaluation
uses seeds 1000 through 1024, which were not used during training.

Every final model will later use the full configured evaluation set of seeds
1000 through 1099. Using the same unseen seeds makes comparisons fair.

## Development command

First run development training so the ignored checkpoint exists:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\train_dqn.py
```

Then evaluate it:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\evaluate_dqn.py
```

The evaluation produces:

```text
results/metrics/development_evaluation_fast_epsilon_decay_seed_0.csv
results/plots/development_evaluation_fast_epsilon_decay_seed_0_rewards.png
```

## Development result

| Measurement | Result |
| --- | ---: |
| Evaluation episodes | 25 |
| Unseen seeds | 1000 through 1024 |
| Mean reward | -92.34 |
| Median reward | -93.92 |
| Standard deviation | 19.75 |
| Minimum reward | -133.57 |
| Maximum reward | -61.80 |
| Episodes scoring at least 200 | 0 of 25 |
| Episodes reaching the 1,000-step limit | 25 of 25 |
| Model parameters unchanged | Yes |

The parameter check shows that the evaluation code measured the saved model
without training it. Every episode reached the time limit instead of ending in
a landing or crash. This suggests that the briefly trained agent learned to
avoid an immediate failure, but it did not learn to finish the landing task.

Its mean reward is higher than the random baseline mean, but this is not yet a
fair final comparison. The development model trained for only 25 episodes and
was evaluated on only 25 different seeds, while the random baseline used 100
episodes. The final experiment will evaluate every trained model on the same
100 seeds.

## Interpretation rule

The development checkpoint trained for only 25 episodes. Its evaluation is a
pipeline check, not a final comparison against the random baseline and not
evidence that the environment is solved.

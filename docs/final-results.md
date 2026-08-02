# Final experiment results

## Experiment completed

The final experiment trained six DQN models:

```text
2 epsilon schedules x 3 training seeds x 800 episodes = 4,800 episodes
```

Each saved model was evaluated with greedy actions for 100 episodes using the
same unseen seeds, 1000 through 1099. This produced 300 evaluation episodes per
schedule and 600 evaluation episodes in total.

## Results for each trained model

| Epsilon schedule | Training seed | Mean reward | Reward SD | Solved episodes |
| --- | ---: | ---: | ---: | ---: |
| Fast decay | 0 | 252.76 | 52.52 | 85 of 100 |
| Fast decay | 1 | 239.84 | 25.61 | 92 of 100 |
| Fast decay | 2 | 223.72 | 87.12 | 81 of 100 |
| Gradual decay | 0 | 222.42 | 63.84 | 75 of 100 |
| Gradual decay | 1 | 252.32 | 39.28 | 90 of 100 |
| Gradual decay | 2 | 104.08 | 85.84 | 17 of 100 |

## Results combined across training seeds

| Measurement | Fast decay | Gradual decay |
| --- | ---: | ---: |
| Trained models | 3 | 3 |
| Evaluation episodes | 300 | 300 |
| Mean reward | 238.77 | 192.94 |
| Median reward | 255.41 | 236.15 |
| Reward standard deviation | 61.52 | 91.71 |
| Solved episodes | 258 of 300 | 182 of 300 |
| Solved percentage | 86.00% | 60.67% |
| Standard deviation between seed means | 14.55 | 78.39 |
| Total training time | 42.04 minutes | 42.27 minutes |

The standard deviation between seed means measures consistency across the
three independently trained models. A lower value means that changing the
training seed had a smaller effect on the final average reward.

## Interpretation

Under this project's fixed settings, fast epsilon decay performed better and
was more reliable. Its combined mean reward was 45.83 points higher, and its
solved percentage was 25.33 percentage points higher. Every fast-decay seed had
a mean reward above 200.

The comparison was not one-sided for every seed. Gradual decay beat fast decay
by 12.48 points for seed 1. However, gradual decay seed 2 had a mean reward of
only 104.08 and solved 17% of its evaluation episodes. That weak run caused the
gradual schedule to have much larger variation between seeds.

This seed-2 result should not be hidden or deleted. Reinforcement learning is
sensitive to the experiences collected during training, and reporting repeated
runs is one reason the experiment used three seeds. The result suggests that
gradual decay was less dependable with this training budget and configuration.

## Limits on the conclusion

The evidence supports the following limited claim:

> Fast epsilon decay produced stronger and more consistent DQN policies than
> gradual epsilon decay in this LunarLander-v3 experiment.

It does not prove that fast decay is always better. The experiment used one
environment, one neural-network design, one 800-episode budget, and only three
training seeds. More seeds or different hyperparameters could change the
result.

The random baseline had a mean reward of -191.00 and solved 0 of 100 episodes.
Both trained schedules improved substantially over that reference. However,
the baseline and final evaluations used different seed sets, so this comparison
is useful context rather than a perfectly paired statistical test.

## Reproducing the analysis

Run:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\analyze_final_results.py
```

The command validates the final files and creates:

```text
results/metrics/final_schedule_summary.csv
results/plots/final_schedule_comparison.png
```

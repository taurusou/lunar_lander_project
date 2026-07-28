# Project plan

## Working title

The Effect of Exploration Decay on DQN Performance in LunarLander-v3

## Research question

How does the rate at which exploration decreases affect average reward,
landing success, and consistency in `LunarLander-v3`?

## Why this environment

LunarLander is more substantial than a tabular toy problem and is not part of
the classic-control group. Its observation is a small continuous vector, its
standard action space has four choices, and its behavior can be recorded for a
clear visual demonstration.

## Planned experiment

1. Measure a random-action baseline.
2. Train a DQN agent with a quickly decreasing exploration rate.
3. Train the same DQN with a gradually decreasing exploration rate.
4. Keep the model structure, training budget, and other settings fixed.
5. Evaluate each result on the same set of unseen seeds.

## Planned measurements

- Episode return during training
- Moving-average return
- Mean and standard deviation of evaluation return
- Safe-landing or success rate
- Training time

## Fairness rules

- Do not select only the best-looking episode.
- Report variability across multiple evaluation episodes and preferably
  multiple training seeds.
- Separate training episodes from evaluation episodes.
- Change one main experimental variable at a time.
- Record failed experiments and troubleshooting decisions.

## Milestones

- [x] Select a tentative environment and research question
- [x] Initialize the repository structure
- [x] Confirm the local Python environment
- [x] Run the environment with a random policy
- [x] Define the experiment configuration
- [x] Implement and verify the baseline
- [ ] Implement and verify DQN training
- [ ] Run controlled experiments
- [ ] Produce plots and a results table
- [ ] Record the demo
- [ ] Complete the report

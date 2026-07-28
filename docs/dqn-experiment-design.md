# DQN experiment design

## Research question

How does the speed of epsilon decay affect the performance and reliability of
a DQN agent in `LunarLander-v3`?

The experiment compares a fast exploration decrease with a gradual exploration
decrease. The epsilon-decay factor is the only planned difference between the
two agents.

## Why epsilon matters

DQN uses an epsilon-greedy policy during training:

- With probability epsilon, the agent chooses a random action and explores.
- Otherwise, the agent chooses the action with the largest predicted Q-value
  and uses what it has learned.

At the beginning, epsilon is `1.0`, so the agent explores heavily. After every
episode, epsilon is updated with:

```text
next epsilon = current epsilon × decay factor
```

Epsilon is never allowed to go below `0.05`.

## Compared schedules

| Experiment | Decay factor | Approximately reaches 0.05 |
| --- | ---: | ---: |
| Fast decay | 0.98 | Episode 149 |
| Gradual decay | 0.995 | Episode 598 |

Fast decay makes the agent trust its early estimates sooner. This may speed up
learning, but it may also cause the agent to settle on a poor strategy.

Gradual decay explores for longer. It may discover more useful experiences, but
continued random actions may slow improvement.

## Shared settings

Both experiments use the same settings below. Keeping these fixed makes the
comparison fair.

### Environment

- Environment: `LunarLander-v3`
- Observations: 8 values
- Actions: 4 choices
- Solved threshold: reward of at least 200
- Maximum episode length: 1,000 steps

### Neural network

- Input layer: 8 observation values
- First hidden layer: 128 neurons with ReLU
- Second hidden layer: 128 neurons with ReLU
- Output layer: 4 Q-values, one for each action

The two 128-neuron hidden layers follow the approachable network shape used in
PyTorch's official DQN tutorial. LunarLander uses state numbers rather than
screen images, so the network input will be the eight environment observations.

### Optimization

- Optimizer: AdamW
- Learning rate: 0.0003
- Batch size: 128 experiences
- Discount factor: 0.99
- Loss: Smooth L1 loss, also called Huber loss
- Gradient clipping value: 100

The discount factor means that the agent values immediate rewards fully and
future rewards slightly less. Smooth L1 loss is less sensitive than squared
error to unusually large and noisy Q-value errors.

### Replay memory

- Capacity: 50,000 experiences
- Begin learning after collecting 1,000 experiences

Replay memory stores past transitions:

```text
(observation, action, reward, next observation, ending information)
```

Training on random samples from this memory reduces the strong ordering between
consecutive experiences. Waiting for 1,000 experiences also gives the first
training batches more variety.

### Target network

- Use a separate target network
- Soft-update rate (`tau`): 0.005
- Update after every environment step

The target network changes slowly, which gives the policy network a more stable
learning target.

## Training and evaluation plan

### Development run

First run only 25 episodes with seed 0. The purpose is to verify that training,
model saving, metrics, and evaluation all work. These results will not be used
as the final experiment.

### Final training

- Train each exploration schedule for 800 episodes.
- Repeat each schedule with training seeds 0, 1, and 2.
- Use CPU training so runs use the same device.
- Record reward, episode length, epsilon, average loss, and elapsed time.

This produces six trained models:

```text
2 exploration schedules × 3 training seeds = 6 models
```

### Evaluation

- Evaluate every trained model for 100 episodes.
- Use evaluation seeds 1000 through 1099.
- Set epsilon to 0 during evaluation.
- Do not update the network during evaluation.

Every model therefore faces the same unseen starting conditions without random
exploration. This isolates the policy it learned.

## Planned comparisons

- Mean and standard deviation of evaluation reward
- Median evaluation reward
- Percentage of evaluation episodes scoring at least 200
- Training reward curves
- Moving-average training reward over 100 episodes
- Variation between training seeds
- Training time

## Interpretation rule

We will not decide that one schedule is better from its single best episode.
The conclusion must use averages, variability, and success rates across all
evaluation episodes and training seeds.

## References

- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Original DQN paper](https://doi.org/10.1038/nature14236)
- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)


# DQN components

The first DQN implementation step contains two independent pieces. Neither piece runs an environment or trains an agent yet.

## Q-network

`DQNNetwork` receives LunarLander's eight observation values and produces four Q-values:

```text
8 observations → 128 neurons → 128 neurons → 4 Q-values
```

Each output corresponds to one LunarLander action. The agent will normally choose the action with the largest Q-value when it is not exploring.

The output does not use softmax. Q-values are predicted future rewards, not probabilities, so they may be negative or positive and do not need to add up to
one.

## Transition

A `Transition` stores one interaction:

```text
observation
action
reward
next observation
terminated
truncated
```

Termination and truncation remain separate because they have different meanings. A natural terminal state has no future reward. A time-limit
truncation ended the recorded episode, but it is not automatically the same as a natural terminal state when calculating a DQN target.

## Replay memory

`ReplayMemory` stores transitions in a fixed-size circular buffer:

1. New transitions fill empty positions.
2. When the memory is full, a new transition replaces the oldest transition.
3. Training will randomly sample batches from the stored transitions.
4. Sampling does not remove anything.

The final configuration allows 50,000 transitions, although the automated tests use tiny capacities so replacement behavior is easy to verify.

## Automated tests

Run:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

The tests check:

- One observation produces four Q-values.
- A real LunarLander observation fits the network.
- A batch preserves its number of rows.
- Every network parameter receives a gradient.
- Replay-memory length grows correctly.
- A full memory replaces its oldest transition.
- Sampling returns the requested number without deleting data.
- Invalid sampling produces a clear error.
- Termination and truncation remain separate.

## Reference

- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

# Experiment configurations

Store named experiment settings here so comparisons remain fair and
reproducible. Planned settings include the random seed, episode budget,
learning rate, discount factor, replay-buffer size, batch size, and exploration
schedule.

Do not change several variables at once when testing a hypothesis.

## Current configuration

[`dqn_experiments.json`](dqn_experiments.json) defines the shared DQN settings
and the two epsilon-decay schedules. The JSON file contains values only; the
reason for every choice is explained in
[`docs/dqn-experiment-design.md`](../docs/dqn-experiment-design.md).

Validate and preview the configuration with:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\lunar_lander_rl\check_dqn_config.py
```

The file also contains tiny smoke-test counts used only to check the controlled
experiment runner. They are separate from the final 800-episode settings.

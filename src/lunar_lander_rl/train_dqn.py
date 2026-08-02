"""Train a DQN agent in LunarLander-v3.

The default run is intentionally small: 25 episodes using the fast epsilon
schedule and training seed 0. Its purpose is to verify the complete training
pipeline before running the final six-model experiment.

The DQN update follows the main structure of PyTorch's official tutorial:
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import csv
import json
import random
import statistics
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy
import torch
from torch import nn

from dqn_components import DQNNetwork, ReplayMemory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "dqn_experiments.json"

# The development run uses only one schedule. Later, the final experiment will
# call the same training function for both schedules and all three seeds.
DEVELOPMENT_EXPERIMENT_NAME = "fast_epsilon_decay"


def load_configuration():
    """Load the shared experiment settings from JSON."""

    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        configuration = json.load(config_file)

    return configuration


def find_experiment(configuration, experiment_name):
    """Find one named epsilon schedule in the configuration."""

    for experiment in configuration["experiments"]:
        if experiment["name"] == experiment_name:
            return experiment

    raise ValueError("Unknown experiment name: " + experiment_name)


def calculate_epsilon(experiment, episode_number):
    """Calculate the exploration rate for one episode."""

    epsilon_start = experiment["epsilon_start"]
    epsilon_end = experiment["epsilon_end"]
    decay_factor = experiment["epsilon_decay_factor_per_episode"]

    decayed_epsilon = epsilon_start * (decay_factor**episode_number)
    epsilon = max(epsilon_end, decayed_epsilon)
    return epsilon


def set_random_seeds(seed):
    """Seed Python, NumPy, and PyTorch for reproducible training."""

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def select_action(
    observation,
    epsilon,
    policy_network,
    environment,
    device,
):
    """Choose a random or network-selected action using epsilon-greedy policy."""

    random_number = random.random()

    if random_number < epsilon:
        # Exploration: try a valid action without consulting the network.
        action = environment.action_space.sample()
        return action

    # Exploitation: ask the network for one Q-value per action and choose the
    # largest. unsqueeze adds the batch dimension expected by PyTorch.
    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # Action selection is not a training update, so gradients are unnecessary.
    with torch.no_grad():
        q_values = policy_network(observation_tensor)
        action = int(q_values.argmax(dim=1).item())

    return action


def optimize_model(
    policy_network,
    target_network,
    replay_memory,
    optimizer,
    configuration,
    device,
):
    """Perform one DQN learning update from a random replay-memory batch."""

    optimization_settings = configuration["optimization"]
    replay_settings = configuration["replay_memory"]
    batch_size = optimization_settings["batch_size"]
    minimum_memory_size = replay_settings["minimum_size_before_training"]

    # Early experiences are only collected. Waiting gives the replay memory
    # enough variety before the first gradient update.
    if len(replay_memory) < minimum_memory_size:
        return None

    if len(replay_memory) < batch_size:
        return None

    transitions = replay_memory.sample(batch_size)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminated_values = []

    for transition in transitions:
        observations.append(transition.observation)
        actions.append(transition.action)
        rewards.append(transition.reward)
        next_observations.append(transition.next_observation)
        terminated_values.append(transition.terminated)

    # Converting the lists together creates one tensor row per transition.
    observation_batch = torch.tensor(
        numpy.array(observations),
        dtype=torch.float32,
        device=device,
    )
    action_batch = torch.tensor(
        actions,
        dtype=torch.int64,
        device=device,
    )
    reward_batch = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    )
    next_observation_batch = torch.tensor(
        numpy.array(next_observations),
        dtype=torch.float32,
        device=device,
    )
    terminated_batch = torch.tensor(
        terminated_values,
        dtype=torch.float32,
        device=device,
    )

    # The policy network predicts four Q-values per observation. gather keeps
    # only the Q-value for the action that was actually taken.
    all_predicted_q_values = policy_network(observation_batch)
    predicted_q_values = all_predicted_q_values.gather(
        1,
        action_batch.unsqueeze(1),
    ).squeeze(1)

    discount_factor = optimization_settings["discount_factor"]

    with torch.no_grad():
        # Standard DQN uses the slowly changing target network to estimate the
        # best future value at the next observation.
        next_q_values = target_network(next_observation_batch)
        best_next_q_values = next_q_values.max(dim=1).values

        # A natural termination has no future reward, so its multiplier is 0.
        # Truncation remains separate: a time limit ends the recorded episode
        # but does not automatically mean the underlying state is terminal.
        future_value_multiplier = 1.0 - terminated_batch
        target_q_values = reward_batch + (
            discount_factor
            * best_next_q_values
            * future_value_multiplier
        )

    # Smooth L1 is also called Huber loss. It is less sensitive than squared
    # error to unusually large, noisy Q-value mistakes.
    loss_function = nn.SmoothL1Loss()
    loss = loss_function(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(
        policy_network.parameters(),
        optimization_settings["gradient_clip_value"],
    )
    optimizer.step()

    return float(loss.item())


def soft_update_target_network(policy_network, target_network, tau):
    """Move each target-network parameter slightly toward the policy network."""

    with torch.no_grad():
        for target_parameter, policy_parameter in zip(
            target_network.parameters(),
            policy_network.parameters(),
        ):
            updated_parameter = (
                tau * policy_parameter
                + (1.0 - tau) * target_parameter
            )
            target_parameter.copy_(updated_parameter)


def make_run_name(experiment_name, training_seed, run_type):
    """Create one consistent name for metrics, plots, and checkpoints."""

    return (
        run_type
        + "_"
        + experiment_name
        + "_seed_"
        + str(training_seed)
    )


def save_metrics(metrics, run_name):
    """Save one CSV row per training episode."""

    metrics_file = (
        PROJECT_ROOT / "results" / "metrics" / (run_name + ".csv")
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    column_names = [
        "episode",
        "training_seed",
        "environment_seed",
        "total_reward",
        "steps",
        "epsilon",
        "average_loss",
        "replay_memory_size",
        "total_environment_steps",
        "terminated",
        "truncated",
        "solved",
        "elapsed_seconds",
    ]

    with metrics_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(metrics)

    return metrics_file


def calculate_moving_average(rewards, window_size):
    """Calculate a trailing average for each point in a reward list."""

    moving_averages = []

    for reward_index in range(len(rewards)):
        first_index = max(0, reward_index - window_size + 1)
        current_window = rewards[first_index : reward_index + 1]
        moving_average = statistics.mean(current_window)
        moving_averages.append(moving_average)

    return moving_averages


def create_training_plot(metrics, run_name, moving_average_window):
    """Save a plot of episode reward and its moving average."""

    episodes = []
    rewards = []

    for episode_metrics in metrics:
        episodes.append(episode_metrics["episode"])
        rewards.append(episode_metrics["total_reward"])

    moving_averages = calculate_moving_average(
        rewards,
        moving_average_window,
    )

    plot_file = (
        PROJECT_ROOT / "results" / "plots" / (run_name + "_rewards.png")
    )
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Episode reward", alpha=0.6)
    plt.plot(
        episodes,
        moving_averages,
        label="Moving average",
        linewidth=2,
    )
    plt.axhline(
        y=200.0,
        color="darkgreen",
        linestyle="--",
        label="Solved threshold",
    )
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    return plot_file


def save_checkpoint(
    policy_network,
    optimizer,
    configuration,
    experiment,
    training_seed,
    run_name,
):
    """Save model and optimizer state for later evaluation and inspection."""

    checkpoint_file = (
        PROJECT_ROOT / "results" / "checkpoints" / (run_name + ".pt")
    )
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "policy_network_state": policy_network.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "configuration": configuration,
        "experiment": experiment,
        "training_seed": training_seed,
    }

    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def train_one_model(
    configuration,
    experiment_name,
    training_seed,
    number_of_episodes,
    run_type,
):
    """Train one DQN model and save its metrics, plot, and checkpoint."""

    experiment = find_experiment(configuration, experiment_name)
    environment_settings = configuration["environment"]
    network_settings = configuration["network"]
    optimization_settings = configuration["optimization"]
    replay_settings = configuration["replay_memory"]
    target_settings = configuration["target_network"]
    training_settings = configuration["training"]

    set_random_seeds(training_seed)

    # This project intentionally uses CPU for consistent comparisons.
    device = torch.device(training_settings["device"])

    environment = gym.make(environment_settings["name"])
    environment.action_space.seed(training_seed)

    policy_network = DQNNetwork(
        environment_settings["observation_count"],
        environment_settings["action_count"],
        network_settings["hidden_layer_sizes"],
    ).to(device)

    target_network = DQNNetwork(
        environment_settings["observation_count"],
        environment_settings["action_count"],
        network_settings["hidden_layer_sizes"],
    ).to(device)

    # Both networks begin with identical parameters. Only the policy network is
    # changed directly by the optimizer.
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.AdamW(
        policy_network.parameters(),
        lr=optimization_settings["learning_rate"],
        amsgrad=True,
    )

    replay_memory = ReplayMemory(replay_settings["capacity"])
    metrics = []
    total_environment_steps = 0
    training_start_time = time.perf_counter()

    try:
        for episode_index in range(number_of_episodes):
            episode_number = episode_index + 1
            epsilon = calculate_epsilon(experiment, episode_index)

            # The same seed formula will be used for both schedules, giving
            # them the same sequence of starting conditions.
            environment_seed = training_seed * 10_000 + episode_index
            observation, information = environment.reset(
                seed=environment_seed
            )

            total_reward = 0.0
            episode_steps = 0
            episode_losses = []
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = select_action(
                    observation,
                    epsilon,
                    policy_network,
                    environment,
                    device,
                )

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    information,
                ) = environment.step(action)

                replay_memory.push(
                    observation,
                    action,
                    float(reward),
                    next_observation,
                    terminated,
                    truncated,
                )

                loss = optimize_model(
                    policy_network,
                    target_network,
                    replay_memory,
                    optimizer,
                    configuration,
                    device,
                )

                if loss is not None:
                    episode_losses.append(loss)

                total_environment_steps = total_environment_steps + 1
                episode_steps = episode_steps + 1
                total_reward = total_reward + float(reward)
                observation = next_observation

                update_interval = target_settings[
                    "update_every_environment_steps"
                ]

                if total_environment_steps % update_interval == 0:
                    soft_update_target_network(
                        policy_network,
                        target_network,
                        target_settings["tau"],
                    )

            if len(episode_losses) > 0:
                average_loss = statistics.mean(episode_losses)
            else:
                average_loss = None

            elapsed_seconds = time.perf_counter() - training_start_time

            episode_metrics = {
                "episode": episode_number,
                "training_seed": training_seed,
                "environment_seed": environment_seed,
                "total_reward": total_reward,
                "steps": episode_steps,
                "epsilon": epsilon,
                "average_loss": average_loss,
                "replay_memory_size": len(replay_memory),
                "total_environment_steps": total_environment_steps,
                "terminated": terminated,
                "truncated": truncated,
                "solved": (
                    total_reward >= environment_settings["solved_reward"]
                ),
                "elapsed_seconds": elapsed_seconds,
            }
            metrics.append(episode_metrics)

            if episode_number % 5 == 0 or episode_number == 1:
                print(
                    "Episode",
                    episode_number,
                    "of",
                    number_of_episodes,
                    "- reward:",
                    round(total_reward, 2),
                    "- epsilon:",
                    round(epsilon, 3),
                    "- memory:",
                    len(replay_memory),
                )
    finally:
        environment.close()

    run_name = make_run_name(
        experiment_name,
        training_seed,
        run_type,
    )

    metrics_file = save_metrics(metrics, run_name)
    plot_file = create_training_plot(
        metrics,
        run_name,
        training_settings["moving_average_window"],
    )
    checkpoint_file = save_checkpoint(
        policy_network,
        optimizer,
        configuration,
        experiment,
        training_seed,
        run_name,
    )

    return metrics, metrics_file, plot_file, checkpoint_file


def print_development_summary(metrics):
    """Print a short summary without treating 25 episodes as final results."""

    rewards = []
    solved_count = 0

    for episode_metrics in metrics:
        rewards.append(episode_metrics["total_reward"])

        if episode_metrics["solved"]:
            solved_count = solved_count + 1

    print("\nDevelopment-run summary")
    print("-----------------------")
    print("Episodes:", len(metrics))
    print("Mean reward:", round(statistics.mean(rewards), 2))
    print("Minimum reward:", round(min(rewards), 2))
    print("Maximum reward:", round(max(rewards), 2))
    print("Episodes scoring at least 200:", solved_count)
    print(
        "Important: this short run verifies the pipeline; it is not a final",
        "performance result.",
    )


def main():
    """Run the configured 25-episode development training."""

    configuration = load_configuration()
    development_settings = configuration["development_run"]

    print("Starting DQN development training...")
    print("Experiment:", DEVELOPMENT_EXPERIMENT_NAME)
    print("Training seed:", development_settings["training_seed"])
    print("Episodes:", development_settings["number_of_episodes"])

    metrics, metrics_file, plot_file, checkpoint_file = train_one_model(
        configuration,
        DEVELOPMENT_EXPERIMENT_NAME,
        development_settings["training_seed"],
        development_settings["number_of_episodes"],
        "development",
    )

    print_development_summary(metrics)
    print("\nSaved metrics to:")
    print(metrics_file)
    print("Saved plot to:")
    print(plot_file)
    print("Saved local checkpoint to:")
    print(checkpoint_file)


if __name__ == "__main__":
    main()

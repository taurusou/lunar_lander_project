"""Evaluate a saved DQN policy without exploration or learning.

The default command evaluates the 25-episode development checkpoint. Actions
are always chosen from the largest predicted Q-value, replay memory is not
used, and no optimizer updates occur.
"""

import csv
import statistics
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from dqn_components import DQNNetwork
from train_dqn import find_experiment, load_configuration, make_run_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_EXPERIMENT_NAME = "fast_epsilon_decay"
DEVELOPMENT_TRAINING_SEED = 0


def get_checkpoint_path(experiment_name, training_seed, run_type):
    """Return the checkpoint path for one named training run."""

    training_run_name = make_run_name(
        experiment_name,
        training_seed,
        run_type,
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "checkpoints"
        / (training_run_name + ".pt")
    )
    return checkpoint_path


def get_development_checkpoint_path():
    """Return the checkpoint path produced by the development training run."""

    return get_checkpoint_path(
        DEVELOPMENT_EXPERIMENT_NAME,
        DEVELOPMENT_TRAINING_SEED,
        "development",
    )


def load_checkpoint(checkpoint_path, device):
    """Load a local PyTorch checkpoint and verify its required information."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Run train_dqn.py first: "
            + str(checkpoint_path)
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = [
        "policy_network_state",
        "configuration",
        "experiment",
        "training_seed",
    ]

    for required_key in required_keys:
        if required_key not in checkpoint:
            raise ValueError(
                "Checkpoint is missing required key: " + required_key
            )

    return checkpoint


def build_policy_network(configuration, checkpoint, device):
    """Recreate the network, load trained parameters, and enter evaluation mode."""

    environment_settings = configuration["environment"]
    network_settings = configuration["network"]

    policy_network = DQNNetwork(
        environment_settings["observation_count"],
        environment_settings["action_count"],
        network_settings["hidden_layer_sizes"],
    ).to(device)

    policy_network.load_state_dict(checkpoint["policy_network_state"])

    # This also keeps evaluation correct if dropout or batch normalization is added to the network later.
    policy_network.eval()
    return policy_network


def select_greedy_action(observation, policy_network, device):
    """Choose the action with the largest Q-value and use no randomness."""

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # Evaluation must not build gradients or change network parameters.
    with torch.no_grad():
        q_values = policy_network(observation_tensor)
        action = int(q_values.argmax(dim=1).item())

    return action


def copy_model_parameters(policy_network):
    """Copy parameters so we can check that evaluation leaves the model unchanged."""

    parameter_copies = []

    for parameter in policy_network.parameters():
        parameter_copies.append(parameter.detach().clone())

    return parameter_copies


def model_parameters_match(parameter_copies, policy_network):
    """Return True only if every parameter still equals its saved copy."""

    for old_parameter, current_parameter in zip(
        parameter_copies,
        policy_network.parameters(),
    ):
        if not torch.equal(old_parameter, current_parameter):
            return False

    return True


def evaluate_policy(
    configuration,
    policy_network,
    number_of_episodes,
    first_evaluation_seed,
    device,
):
    """Run greedy evaluation episodes and return one metrics row per episode."""

    environment_settings = configuration["environment"]
    environment = gym.make(environment_settings["name"])
    evaluation_metrics = []

    try:
        for episode_index in range(number_of_episodes):
            episode_number = episode_index + 1
            evaluation_seed = first_evaluation_seed + episode_index
            observation, information = environment.reset(seed=evaluation_seed)

            total_reward = 0.0
            step_count = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = select_greedy_action(
                    observation,
                    policy_network,
                    device,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    information,
                ) = environment.step(action)

                total_reward = total_reward + float(reward)
                step_count = step_count + 1

                if (
                    step_count
                    > environment_settings["maximum_steps_per_episode"]
                ):
                    raise RuntimeError(
                        "Evaluation exceeded the configured step limit."
                    )

            episode_metrics = {
                "episode": episode_number,
                "evaluation_seed": evaluation_seed,
                "total_reward": total_reward,
                "steps": step_count,
                "terminated": terminated,
                "truncated": truncated,
                "solved": (
                    total_reward >= environment_settings["solved_reward"]
                ),
            }
            evaluation_metrics.append(episode_metrics)

            if episode_number % 5 == 0 or episode_number == 1:
                print(
                    "Evaluated episode",
                    episode_number,
                    "of",
                    number_of_episodes,
                    "- reward:",
                    round(total_reward, 2),
                )
    finally:
        environment.close()

    return evaluation_metrics


def calculate_summary(evaluation_metrics):
    """Calculate statistics across all evaluation episodes."""

    rewards = []
    solved_count = 0

    for episode_metrics in evaluation_metrics:
        rewards.append(episode_metrics["total_reward"])

        if episode_metrics["solved"]:
            solved_count = solved_count + 1

    if len(rewards) < 2:
        reward_standard_deviation = 0.0
    else:
        reward_standard_deviation = statistics.stdev(rewards)

    summary = {
        "mean": statistics.mean(rewards),
        "median": statistics.median(rewards),
        "standard_deviation": reward_standard_deviation,
        "minimum": min(rewards),
        "maximum": max(rewards),
        "solved_count": solved_count,
        "solved_percentage": 100.0 * solved_count / len(rewards),
    }
    return summary


def make_evaluation_run_name(experiment_name, training_seed, run_type):
    """Create a consistent name for evaluation CSV and plot files."""

    return (
        run_type
        + "_evaluation_"
        + experiment_name
        + "_seed_"
        + str(training_seed)
    )


def save_evaluation_metrics(evaluation_metrics, evaluation_run_name):
    """Save evaluation episode results to CSV."""

    metrics_file = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / (evaluation_run_name + ".csv")
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    column_names = [
        "episode",
        "evaluation_seed",
        "total_reward",
        "steps",
        "terminated",
        "truncated",
        "solved",
    ]

    with metrics_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(evaluation_metrics)

    return metrics_file


def create_evaluation_plot(
    evaluation_metrics,
    summary,
    solved_reward,
    evaluation_run_name,
):
    """Save a plot showing each evaluation reward and the mean reward."""

    episode_numbers = []
    rewards = []

    for episode_metrics in evaluation_metrics:
        episode_numbers.append(episode_metrics["episode"])
        rewards.append(episode_metrics["total_reward"])

    plot_file = (
        PROJECT_ROOT
        / "results"
        / "plots"
        / (evaluation_run_name + "_rewards.png")
    )
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, rewards, marker="o", label="Evaluation reward")
    plt.axhline(
        y=summary["mean"],
        color="darkorange",
        linestyle="-",
        label="Mean reward",
    )
    plt.axhline(
        y=solved_reward,
        color="darkgreen",
        linestyle="--",
        label="Solved threshold",
    )
    plt.title("DQN Evaluation Rewards")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    return plot_file


def print_summary(summary, number_of_episodes):
    """Print evaluation statistics in a readable format."""

    print("\nDevelopment-evaluation summary")
    print("------------------------------")
    print("Episodes:", number_of_episodes)
    print("Mean reward:", round(summary["mean"], 2))
    print("Median reward:", round(summary["median"], 2))
    print(
        "Reward standard deviation:",
        round(summary["standard_deviation"], 2),
    )
    print("Minimum reward:", round(summary["minimum"], 2))
    print("Maximum reward:", round(summary["maximum"], 2))
    print("Solved episodes:", summary["solved_count"])
    print("Solved percentage:", round(summary["solved_percentage"], 2), "%")
    print(
        "Important: this evaluates a 25-episode development model, not a",
        "final trained model.",
    )


def evaluate_saved_checkpoint(
    configuration,
    experiment_name,
    training_seed,
    number_of_episodes,
    first_evaluation_seed,
    run_type,
):
    """Load, evaluate, and save results for one trained DQN checkpoint."""

    device = torch.device(configuration["training"]["device"])
    experiment = find_experiment(configuration, experiment_name)
    checkpoint_path = get_checkpoint_path(
        experiment_name,
        training_seed,
        run_type,
    )
    checkpoint = load_checkpoint(checkpoint_path, device)

    # These checks help prevent labeling one model with another run's name.
    if checkpoint["experiment"]["name"] != experiment["name"]:
        raise ValueError("Checkpoint experiment does not match evaluation.")

    if checkpoint["training_seed"] != training_seed:
        raise ValueError("Checkpoint training seed does not match evaluation.")

    policy_network = build_policy_network(
        configuration,
        checkpoint,
        device,
    )
    parameters_before_evaluation = copy_model_parameters(policy_network)

    evaluation_metrics = evaluate_policy(
        configuration,
        policy_network,
        number_of_episodes,
        first_evaluation_seed,
        device,
    )

    if not model_parameters_match(
        parameters_before_evaluation,
        policy_network,
    ):
        raise RuntimeError("Evaluation unexpectedly changed model parameters.")

    summary = calculate_summary(evaluation_metrics)
    evaluation_run_name = make_evaluation_run_name(
        experiment_name,
        training_seed,
        run_type,
    )
    metrics_file = save_evaluation_metrics(
        evaluation_metrics,
        evaluation_run_name,
    )
    plot_file = create_evaluation_plot(
        evaluation_metrics,
        summary,
        configuration["environment"]["solved_reward"],
        evaluation_run_name,
    )

    return (
        evaluation_metrics,
        summary,
        metrics_file,
        plot_file,
        checkpoint_path,
    )


def main():
    """Load the development checkpoint and run greedy unseen-seed evaluation."""

    configuration = load_configuration()
    development_settings = configuration["development_run"]
    evaluation_settings = configuration["evaluation"]
    checkpoint_path = get_development_checkpoint_path()
    number_of_episodes = development_settings["evaluation_episodes"]
    first_evaluation_seed = evaluation_settings["first_seed"]

    print("Starting greedy DQN development evaluation...")
    print("Checkpoint:", checkpoint_path)
    print("Evaluation episodes:", number_of_episodes)
    print("Evaluation seeds:", first_evaluation_seed, "through", end=" ")
    print(first_evaluation_seed + number_of_episodes - 1)
    print("Exploration rate:", evaluation_settings["exploration_rate"])

    (
        evaluation_metrics,
        summary,
        metrics_file,
        plot_file,
        checkpoint_path,
    ) = evaluate_saved_checkpoint(
        configuration,
        DEVELOPMENT_EXPERIMENT_NAME,
        DEVELOPMENT_TRAINING_SEED,
        number_of_episodes,
        first_evaluation_seed,
        "development",
    )

    print_summary(summary, number_of_episodes)
    print("Model parameters unchanged: True")
    print("\nSaved evaluation metrics to:")
    print(metrics_file)
    print("Saved evaluation plot to:")
    print(plot_file)


if __name__ == "__main__":
    main()

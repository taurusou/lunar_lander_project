"""Read and validate the DQN experiment configuration.

This script does not train a neural network. It checks the JSON configuration
before an experiment starts, which helps catch unfair or invalid settings early.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "dqn_experiments.json"


def load_configuration():
    """Load the JSON file and return it as a Python dictionary."""

    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        configuration = json.load(config_file)

    return configuration


def calculate_epsilon(experiment, episode_number):
    """Calculate epsilon after a given number of completed episodes."""

    epsilon_start = experiment["epsilon_start"]
    epsilon_end = experiment["epsilon_end"]
    decay_factor = experiment["epsilon_decay_factor_per_episode"]

    decayed_epsilon = epsilon_start * (decay_factor**episode_number)

    # max prevents epsilon from falling below the required minimum.
    epsilon = max(epsilon_end, decayed_epsilon)
    return epsilon


def find_first_minimum_epsilon_episode(experiment, training_episodes):
    """Find the first episode where epsilon reaches its minimum value."""

    epsilon_end = experiment["epsilon_end"]

    for episode_number in range(training_episodes + 1):
        epsilon = calculate_epsilon(experiment, episode_number)

        if epsilon <= epsilon_end:
            return episode_number

    # None means the schedule does not reach its minimum during training.
    return None


def validate_positive_number(value, setting_name):
    """Raise a readable error if a setting is zero or negative."""

    if value <= 0:
        raise ValueError(setting_name + " must be greater than zero.")


def validate_configuration(configuration):
    """Check important correctness and fairness rules."""

    environment = configuration["environment"]
    network = configuration["network"]
    optimization = configuration["optimization"]
    replay_memory = configuration["replay_memory"]
    target_network = configuration["target_network"]
    training = configuration["training"]
    evaluation = configuration["evaluation"]
    experiments = configuration["experiments"]

    if environment["observation_count"] != 8:
        raise ValueError("LunarLander-v3 should have 8 observation values.")

    if environment["action_count"] != 4:
        raise ValueError("Discrete LunarLander-v3 should have 4 actions.")

    if network["activation_function"] != "ReLU":
        raise ValueError("This implementation uses ReLU activation.")

    if optimization["optimizer"] != "AdamW":
        raise ValueError("This implementation uses the AdamW optimizer.")

    if not isinstance(optimization["amsgrad"], bool):
        raise ValueError("amsgrad must be either true or false.")

    if optimization["loss_function"] != "SmoothL1Loss":
        raise ValueError("This implementation uses SmoothL1Loss.")

    if target_network["update_method"] != "soft":
        raise ValueError("This implementation uses soft target-network updates.")

    if evaluation["exploration_rate"] != 0.0:
        raise ValueError("Evaluation must use an exploration rate of 0.")

    validate_positive_number(
        training["number_of_episodes"],
        "Training episode count",
    )
    validate_positive_number(
        evaluation["episodes_per_trained_model"],
        "Evaluation episode count",
    )
    validate_positive_number(
        configuration["development_run"]["evaluation_episodes"],
        "Development evaluation episode count",
    )
    validate_positive_number(
        configuration["smoke_test"]["number_of_episodes"],
        "Smoke-test training episode count",
    )
    validate_positive_number(
        configuration["smoke_test"]["evaluation_episodes"],
        "Smoke-test evaluation episode count",
    )
    validate_positive_number(
        optimization["batch_size"],
        "Batch size",
    )
    validate_positive_number(
        replay_memory["capacity"],
        "Replay-memory capacity",
    )

    if replay_memory["minimum_size_before_training"] < optimization["batch_size"]:
        raise ValueError(
            "Replay memory must contain at least one full batch before training."
        )

    if len(set(training["training_seeds"])) != len(training["training_seeds"]):
        raise ValueError("Training seeds must not contain duplicates.")

    evaluation_seeds = range(
        evaluation["first_seed"],
        evaluation["first_seed"] + evaluation["episodes_per_trained_model"],
    )

    if not set(training["training_seeds"]).isdisjoint(evaluation_seeds):
        raise ValueError("Training and evaluation seeds must not overlap.")

    if len(experiments) != 2:
        raise ValueError("This project should compare exactly two schedules.")

    first_experiment = experiments[0]
    second_experiment = experiments[1]

    # Both schedules must begin and end at the same epsilon values. The decay
    # factor is the experiment's only intended independent variable.
    if first_experiment["epsilon_start"] != second_experiment["epsilon_start"]:
        raise ValueError("Both schedules must use the same starting epsilon.")

    if first_experiment["epsilon_end"] != second_experiment["epsilon_end"]:
        raise ValueError("Both schedules must use the same minimum epsilon.")

    for experiment in experiments:
        epsilon_start = experiment["epsilon_start"]
        epsilon_end = experiment["epsilon_end"]
        decay_factor = experiment["epsilon_decay_factor_per_episode"]

        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Epsilon values must be between 0 and 1.")

        if not 0 < decay_factor < 1:
            raise ValueError("Each epsilon decay factor must be between 0 and 1.")


def print_configuration_summary(configuration):
    """Print the most important settings and epsilon checkpoints."""

    training = configuration["training"]
    evaluation = configuration["evaluation"]
    experiments = configuration["experiments"]
    training_episodes = training["number_of_episodes"]

    print("DQN experiment configuration")
    print("----------------------------")
    print("Training episodes per model:", training_episodes)
    print("Training seeds:", training["training_seeds"])
    print(
        "Evaluation episodes per model:",
        evaluation["episodes_per_trained_model"],
    )
    print("Evaluation begins with seed:", evaluation["first_seed"])

    checkpoint_episodes = [0, 100, 200, 400, 600, 800]

    for experiment in experiments:
        print("\nExperiment:", experiment["name"])
        print(
            "Decay factor:",
            experiment["epsilon_decay_factor_per_episode"],
        )

        for episode_number in checkpoint_episodes:
            epsilon = calculate_epsilon(experiment, episode_number)
            print(
                "  Epsilon after episode",
                episode_number,
                "=",
                round(epsilon, 4),
            )

        minimum_episode = find_first_minimum_epsilon_episode(
            experiment,
            training_episodes,
        )
        print("  First episode at minimum epsilon:", minimum_episode)


def main():
    """Load, validate, and summarize the DQN configuration."""

    configuration = load_configuration()
    validate_configuration(configuration)
    print_configuration_summary(configuration)
    print("\nConfiguration check passed.")


if __name__ == "__main__":
    main()

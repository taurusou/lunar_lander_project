"""Run and measure a random-action baseline in LunarLander-v3.

A baseline is a simple method that gives us something to compare against.
This agent does not learn from experience. At every step, it chooses one of
the four actions randomly.

The reset/step loop follows Gymnasium's official basic-usage pattern:
https://gymnasium.farama.org/main/introduction/basic_usage/
"""

import csv
import statistics
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt


# Keeping important settings near the top makes the experiment easy to read
# and change. These values also tell another student exactly what we ran.
ENVIRONMENT_NAME = "LunarLander-v3"
NUMBER_OF_EPISODES = 100
FIRST_SEED = 0
SOLVED_REWARD = 200.0
SAFETY_STEP_LIMIT = 2_000


# This file is inside src/lunar_lander_rl. Going up three folder levels gives
# us the repository's root folder, regardless of where the command is run.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "results" / "metrics" / "random_baseline.csv"
EPISODE_PLOT_FILE = (
    PROJECT_ROOT / "results" / "plots" / "random_rewards_by_episode.png"
)
DISTRIBUTION_PLOT_FILE = (
    PROJECT_ROOT / "results" / "plots" / "random_reward_distribution.png"
)


def run_one_episode(environment, seed):
    """Run one episode with random actions and return its measurements."""

    # reset begins a new episode. Giving each episode its own seed makes the
    # experiment repeatable.
    observation, information = environment.reset(seed=seed)

    # The action space has its own random-number generator. Seeding it makes
    # the random action sequence repeatable too.
    environment.action_space.seed(seed)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    # An episode is finished when either Gymnasium ending condition is true.
    while not terminated and not truncated:
        # sample chooses a valid random action from Discrete(4).
        action = environment.action_space.sample()

        # step applies the action and returns the new state of the environment.
        observation, reward, terminated, truncated, information = (
            environment.step(action)
        )

        total_reward = total_reward + float(reward)
        step_count = step_count + 1

        # This should never happen because LunarLander has its own time limit.
        # It protects the program if the environment is accidentally misused.
        if step_count >= SAFETY_STEP_LIMIT:
            raise RuntimeError("An episode exceeded the safety step limit.")

    # A dictionary gives a clear name to every value that will become a
    # column in the CSV file.
    episode_result = {
        "seed": seed,
        "total_reward": total_reward,
        "steps": step_count,
        "terminated": terminated,
        "truncated": truncated,
        "solved": total_reward >= SOLVED_REWARD,
    }

    return episode_result


def run_baseline_experiment():
    """Run all random episodes and return a list of result dictionaries."""

    # Rendering is omitted because drawing every frame would make the
    # 100-episode experiment much slower.
    environment = gym.make(ENVIRONMENT_NAME)
    all_results = []

    try:
        for episode_number in range(1, NUMBER_OF_EPISODES + 1):
            seed = FIRST_SEED + episode_number - 1
            episode_result = run_one_episode(environment, seed)

            # Add the human-friendly episode number before saving the result.
            episode_result["episode"] = episode_number
            all_results.append(episode_result)

            # A progress message every 10 episodes shows that the program is
            # still working without printing 100 nearly identical lines.
            if episode_number % 10 == 0:
                rounded_reward = round(episode_result["total_reward"], 2)
                print(
                    "Finished episode",
                    episode_number,
                    "of",
                    NUMBER_OF_EPISODES,
                    "- reward:",
                    rounded_reward,
                )
    finally:
        # Always release the environment's physics resources.
        environment.close()

    return all_results


def save_results_to_csv(all_results):
    """Save every episode's measurements in a CSV file."""

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    column_names = [
        "episode",
        "seed",
        "total_reward",
        "steps",
        "terminated",
        "truncated",
        "solved",
    ]

    # newline="" avoids blank lines between rows on Windows.
    with METRICS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()

        for episode_result in all_results:
            writer.writerow(episode_result)


def calculate_summary(all_results):
    """Calculate statistics that describe the random agent's performance."""

    rewards = []

    for episode_result in all_results:
        rewards.append(episode_result["total_reward"])

    solved_count = 0

    for episode_result in all_results:
        if episode_result["solved"]:
            solved_count = solved_count + 1

    summary = {
        "mean": statistics.mean(rewards),
        "median": statistics.median(rewards),
        "standard_deviation": statistics.stdev(rewards),
        "minimum": min(rewards),
        "maximum": max(rewards),
        "solved_count": solved_count,
    }

    return summary


def print_summary(summary):
    """Print the baseline statistics in a readable form."""

    print("\nRandom baseline summary")
    print("-----------------------")
    print("Episodes:", NUMBER_OF_EPISODES)
    print("Mean reward:", round(summary["mean"], 2))
    print("Median reward:", round(summary["median"], 2))
    print(
        "Reward standard deviation:",
        round(summary["standard_deviation"], 2),
    )
    print("Minimum reward:", round(summary["minimum"], 2))
    print("Maximum reward:", round(summary["maximum"], 2))
    print(
        "Episodes scoring at least",
        SOLVED_REWARD,
        ":",
        summary["solved_count"],
    )


def create_episode_plot(all_results):
    """Create a line plot showing the reward from every episode."""

    episode_numbers = []
    rewards = []

    for episode_result in all_results:
        episode_numbers.append(episode_result["episode"])
        rewards.append(episode_result["total_reward"])

    EPISODE_PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, rewards, color="steelblue", linewidth=1)
    plt.axhline(
        y=SOLVED_REWARD,
        color="darkgreen",
        linestyle="--",
        label="Solved threshold",
    )
    plt.title("Random Agent Reward by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EPISODE_PLOT_FILE, dpi=150)
    plt.close()


def create_distribution_plot(all_results):
    """Create a histogram showing how the episode rewards are distributed."""

    rewards = []

    for episode_result in all_results:
        rewards.append(episode_result["total_reward"])

    DISTRIBUTION_PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(rewards, bins=15, color="cornflowerblue", edgecolor="black")
    plt.axvline(
        x=SOLVED_REWARD,
        color="darkgreen",
        linestyle="--",
        label="Solved threshold",
    )
    plt.title("Distribution of Random Agent Rewards")
    plt.xlabel("Total reward")
    plt.ylabel("Number of episodes")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT_FILE, dpi=150)
    plt.close()


def main():
    """Run the complete baseline experiment from start to finish."""

    print("Running", NUMBER_OF_EPISODES, "random LunarLander episodes...")

    all_results = run_baseline_experiment()
    save_results_to_csv(all_results)

    summary = calculate_summary(all_results)
    print_summary(summary)

    create_episode_plot(all_results)
    create_distribution_plot(all_results)

    print("\nSaved episode data to:")
    print(METRICS_FILE)
    print("Saved plots to:")
    print(EPISODE_PLOT_FILE)
    print(DISTRIBUTION_PLOT_FILE)


if __name__ == "__main__":
    main()

"""Create a schedule-level table and plot from the final DQN results.

The experiment runner saves one row for each trained model. This file performs
the next level of analysis by combining the three training seeds belonging to
each epsilon schedule. It does not train or change any model.
"""

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from train_dqn import load_configuration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SUMMARY_FILE = (
    PROJECT_ROOT / "results" / "metrics" / "final_experiment_summary.csv"
)


def load_model_summary(summary_file):
    """Load the six model rows and convert numbers from CSV text."""

    if not summary_file.exists():
        raise FileNotFoundError(
            "Final model summary was not found: " + str(summary_file)
        )

    model_rows = []

    with summary_file.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            model_rows.append(
                {
                    "experiment_name": row["experiment_name"],
                    "training_seed": int(row["training_seed"]),
                    "training_episodes": int(row["training_episodes"]),
                    "evaluation_episodes": int(row["evaluation_episodes"]),
                    "mean_reward": float(row["mean_reward"]),
                    "median_reward": float(row["median_reward"]),
                    "reward_standard_deviation": float(
                        row["reward_standard_deviation"]
                    ),
                    "minimum_reward": float(row["minimum_reward"]),
                    "maximum_reward": float(row["maximum_reward"]),
                    "solved_count": int(row["solved_count"]),
                    "solved_percentage": float(row["solved_percentage"]),
                    "training_seconds": float(row["training_seconds"]),
                }
            )

    return model_rows


def load_schedule_rewards(experiment_name, model_rows):
    """Load every evaluation reward produced for one epsilon schedule."""

    rewards = []

    for model_row in model_rows:
        training_seed = model_row["training_seed"]
        metrics_file = (
            PROJECT_ROOT
            / "results"
            / "metrics"
            / (
                "final_evaluation_"
                + experiment_name
                + "_seed_"
                + str(training_seed)
                + ".csv"
            )
        )

        if not metrics_file.exists():
            raise FileNotFoundError(
                "Evaluation metrics were not found: " + str(metrics_file)
            )

        model_reward_count = 0

        with metrics_file.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                rewards.append(float(row["total_reward"]))
                model_reward_count = model_reward_count + 1

        expected_count = model_row["evaluation_episodes"]

        if model_reward_count != expected_count:
            raise ValueError(
                metrics_file.name
                + " contains "
                + str(model_reward_count)
                + " rewards, but "
                + str(expected_count)
                + " were expected."
            )

    return rewards


def calculate_schedule_summary(experiment_name, model_rows, rewards):
    """Calculate final measurements across all seeds for one schedule."""

    if len(model_rows) == 0:
        raise ValueError("A schedule must contain at least one trained model.")

    if len(rewards) == 0:
        raise ValueError("A schedule must contain evaluation rewards.")

    seed_mean_rewards = []
    solved_count = 0
    total_training_seconds = 0.0

    for model_row in model_rows:
        seed_mean_rewards.append(model_row["mean_reward"])
        solved_count = solved_count + model_row["solved_count"]
        total_training_seconds = (
            total_training_seconds + model_row["training_seconds"]
        )

    # Variation between seed means tells us whether the training method is
    # dependable across repeated runs, not merely whether one model was good.
    if len(seed_mean_rewards) < 2:
        seed_mean_standard_deviation = 0.0
    else:
        seed_mean_standard_deviation = statistics.stdev(seed_mean_rewards)

    return {
        "experiment_name": experiment_name,
        "trained_models": len(model_rows),
        "evaluation_episodes": len(rewards),
        "pooled_mean_reward": statistics.mean(rewards),
        "pooled_median_reward": statistics.median(rewards),
        "pooled_reward_standard_deviation": statistics.stdev(rewards),
        "solved_count": solved_count,
        "solved_percentage": 100.0 * solved_count / len(rewards),
        "mean_of_seed_mean_rewards": statistics.mean(seed_mean_rewards),
        "seed_mean_standard_deviation": seed_mean_standard_deviation,
        "total_training_minutes": total_training_seconds / 60.0,
    }


def save_schedule_summaries(schedule_summaries):
    """Save one compact CSV row per epsilon schedule."""

    output_file = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / "final_schedule_summary.csv"
    )

    column_names = [
        "experiment_name",
        "trained_models",
        "evaluation_episodes",
        "pooled_mean_reward",
        "pooled_median_reward",
        "pooled_reward_standard_deviation",
        "solved_count",
        "solved_percentage",
        "mean_of_seed_mean_rewards",
        "seed_mean_standard_deviation",
        "total_training_minutes",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(schedule_summaries)

    return output_file


def make_display_name(experiment_name):
    """Change a configuration name into a short plot label."""

    if experiment_name == "fast_epsilon_decay":
        return "Fast decay"

    if experiment_name == "gradual_epsilon_decay":
        return "Gradual decay"

    return experiment_name.replace("_", " ").title()


def create_schedule_comparison_plot(
    model_rows,
    experiment_names,
    solved_reward,
):
    """Compare reward and success for every schedule and training seed."""

    training_seeds = sorted(
        set(model_row["training_seed"] for model_row in model_rows)
    )
    base_positions = list(range(len(training_seeds)))
    bar_width = 0.35

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for experiment_index, experiment_name in enumerate(experiment_names):
        schedule_rows = [
            model_row
            for model_row in model_rows
            if model_row["experiment_name"] == experiment_name
        ]
        schedule_rows.sort(key=lambda row: row["training_seed"])

        # Shift the fast bars left and gradual bars right so both values for
        # one seed remain easy to compare.
        position_shift = (
            experiment_index - (len(experiment_names) - 1) / 2
        ) * bar_width
        bar_positions = [
            position + position_shift for position in base_positions
        ]

        mean_rewards = [row["mean_reward"] for row in schedule_rows]
        solved_percentages = [
            row["solved_percentage"] for row in schedule_rows
        ]
        display_name = make_display_name(experiment_name)

        axes[0].bar(
            bar_positions,
            mean_rewards,
            width=bar_width,
            label=display_name,
        )
        axes[1].bar(
            bar_positions,
            solved_percentages,
            width=bar_width,
            label=display_name,
        )

    axes[0].axhline(
        y=solved_reward,
        color="darkgreen",
        linestyle="--",
        label="Solved reward threshold",
    )
    axes[0].set_title("Mean Evaluation Reward by Training Seed")
    axes[0].set_ylabel("Mean reward")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_title("Solved Evaluations by Training Seed")
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Episodes solved (%)")
    axes[1].set_xticks(base_positions, [str(seed) for seed in training_seeds])
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plot_file = (
        PROJECT_ROOT
        / "results"
        / "plots"
        / "final_schedule_comparison.png"
    )
    figure.tight_layout()
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)
    return plot_file


def print_schedule_summaries(schedule_summaries):
    """Display the most important final measurements."""

    print("Final schedule comparison")
    print("-------------------------")

    for summary in schedule_summaries:
        print("\nSchedule:", make_display_name(summary["experiment_name"]))
        print("Trained models:", summary["trained_models"])
        print("Evaluation episodes:", summary["evaluation_episodes"])
        print("Mean reward:", round(summary["pooled_mean_reward"], 2))
        print("Median reward:", round(summary["pooled_median_reward"], 2))
        print(
            "Reward standard deviation:",
            round(summary["pooled_reward_standard_deviation"], 2),
        )
        print(
            "Solved episodes:",
            str(summary["solved_count"])
            + " of "
            + str(summary["evaluation_episodes"]),
        )
        print(
            "Solved percentage:",
            round(summary["solved_percentage"], 2),
        )
        print(
            "Variation between seed means:",
            round(summary["seed_mean_standard_deviation"], 2),
        )


def main():
    """Validate the six models and produce final comparison artifacts."""

    configuration = load_configuration()
    model_rows = load_model_summary(MODEL_SUMMARY_FILE)
    expected_model_count = (
        len(configuration["experiments"])
        * len(configuration["training"]["training_seeds"])
    )

    if len(model_rows) != expected_model_count:
        raise ValueError(
            "Expected "
            + str(expected_model_count)
            + " final models, but found "
            + str(len(model_rows))
            + "."
        )

    experiment_names = [
        experiment["name"] for experiment in configuration["experiments"]
    ]
    schedule_summaries = []

    for experiment_name in experiment_names:
        schedule_model_rows = [
            model_row
            for model_row in model_rows
            if model_row["experiment_name"] == experiment_name
        ]

        if len(schedule_model_rows) != len(
            configuration["training"]["training_seeds"]
        ):
            raise ValueError(
                experiment_name + " does not have all configured seeds."
            )

        rewards = load_schedule_rewards(
            experiment_name,
            schedule_model_rows,
        )
        schedule_summary = calculate_schedule_summary(
            experiment_name,
            schedule_model_rows,
            rewards,
        )
        schedule_summaries.append(schedule_summary)

    summary_file = save_schedule_summaries(schedule_summaries)
    plot_file = create_schedule_comparison_plot(
        model_rows,
        experiment_names,
        configuration["environment"]["solved_reward"],
    )

    print_schedule_summaries(schedule_summaries)
    print("\nSaved schedule summary to:")
    print(summary_file)
    print("Saved comparison plot to:")
    print(plot_file)


if __name__ == "__main__":
    main()

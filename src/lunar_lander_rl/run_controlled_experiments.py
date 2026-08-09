"""Run the controlled DQN comparison defined in the shared configuration.

Running this file without a flag only displays the experiment plan, which
avoids accidentally starting six long training runs. Use ``--smoke-test`` for
a short pipeline check and ``--run-final`` for the complete experiment.
"""

import argparse
import csv
from pathlib import Path

from evaluate_dqn import (
    calculate_summary,
    evaluate_saved_checkpoint,
    make_evaluation_run_name,
)
from train_dqn import load_configuration, make_run_name, train_one_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_run_plan(configuration, run_type):
    """Create one plan row for each experiment and training-seed pair."""

    if run_type == "smoke":
        settings = configuration["smoke_test"]
        number_of_episodes = settings["number_of_episodes"]
        evaluation_episodes = settings["evaluation_episodes"]
        training_seeds = settings["training_seeds"]
    elif run_type == "final":
        settings = configuration["training"]
        number_of_episodes = settings["number_of_episodes"]
        evaluation_episodes = configuration["evaluation"][
            "episodes_per_trained_model"
        ]
        training_seeds = settings["training_seeds"]
    else:
        raise ValueError("Run type must be 'smoke' or 'final'.")

    run_plan = []

    for experiment in configuration["experiments"]:
        for training_seed in training_seeds:
            planned_run = {
                "run_type": run_type,
                "experiment_name": experiment["name"],
                "training_seed": training_seed,
                "number_of_episodes": number_of_episodes,
                "evaluation_episodes": evaluation_episodes,
                "first_evaluation_seed": configuration["evaluation"][
                    "first_seed"
                ],
            }
            run_plan.append(planned_run)

    return run_plan


def get_expected_output_paths(planned_run):
    """Return the five files produced for one trained and evaluated model."""

    training_run_name = make_run_name(
        planned_run["experiment_name"],
        planned_run["training_seed"],
        planned_run["run_type"],
    )
    evaluation_run_name = make_evaluation_run_name(
        planned_run["experiment_name"],
        planned_run["training_seed"],
        planned_run["run_type"],
    )

    return {
        "training_metrics": (
            PROJECT_ROOT
            / "results"
            / "metrics"
            / (training_run_name + ".csv")
        ),
        "training_plot": (
            PROJECT_ROOT
            / "results"
            / "plots"
            / (training_run_name + "_rewards.png")
        ),
        "checkpoint": (
            PROJECT_ROOT
            / "results"
            / "checkpoints"
            / (training_run_name + ".pt")
        ),
        "evaluation_metrics": (
            PROJECT_ROOT
            / "results"
            / "metrics"
            / (evaluation_run_name + ".csv")
        ),
        "evaluation_plot": (
            PROJECT_ROOT
            / "results"
            / "plots"
            / (evaluation_run_name + "_rewards.png")
        ),
    }


def classify_run_state(output_paths):
    """Describe whether a model is pending, trained, complete, or incomplete."""

    training_keys = ["training_metrics", "training_plot", "checkpoint"]
    evaluation_keys = ["evaluation_metrics", "evaluation_plot"]

    training_exists = []
    evaluation_exists = []

    for key in training_keys:
        training_exists.append(output_paths[key].exists())

    for key in evaluation_keys:
        evaluation_exists.append(output_paths[key].exists())

    if not any(training_exists) and not any(evaluation_exists):
        return "pending"

    if all(training_exists) and not any(evaluation_exists):
        return "trained"

    if all(training_exists) and all(evaluation_exists):
        return "complete"

    return "incomplete"


def load_training_metrics(metrics_file):
    """Read previously saved training metrics for a resumed run."""

    with metrics_file.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_evaluation_metrics(metrics_file):
    """Read evaluation CSV values and restore their useful Python types."""

    evaluation_metrics = []

    with metrics_file.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            evaluation_metrics.append(
                {
                    "episode": int(row["episode"]),
                    "evaluation_seed": int(row["evaluation_seed"]),
                    "total_reward": float(row["total_reward"]),
                    "steps": int(row["steps"]),
                    "terminated": row["terminated"] == "True",
                    "truncated": row["truncated"] == "True",
                    "solved": row["solved"] == "True",
                }
            )

    return evaluation_metrics


def make_summary_row(planned_run, training_metrics, evaluation_metrics):
    """Combine one model's important training and evaluation measurements."""

    expected_training_episodes = planned_run["number_of_episodes"]
    expected_evaluation_episodes = planned_run["evaluation_episodes"]

    if len(training_metrics) != expected_training_episodes:
        raise ValueError(
            "Training CSV has "
            + str(len(training_metrics))
            + " rows, but "
            + str(expected_training_episodes)
            + " were expected."
        )

    if len(evaluation_metrics) != expected_evaluation_episodes:
        raise ValueError(
            "Evaluation CSV has "
            + str(len(evaluation_metrics))
            + " rows, but "
            + str(expected_evaluation_episodes)
            + " were expected."
        )

    summary = calculate_summary(evaluation_metrics)

    # elapsed_seconds is cumulative, so the final training row contains the
    # complete training time. Loaded CSV values are strings; fresh values are
    # numbers, and float safely handles either form.
    training_seconds = float(training_metrics[-1]["elapsed_seconds"])

    return {
        "experiment_name": planned_run["experiment_name"],
        "training_seed": planned_run["training_seed"],
        "training_episodes": len(training_metrics),
        "evaluation_episodes": len(evaluation_metrics),
        "mean_reward": summary["mean"],
        "median_reward": summary["median"],
        "reward_standard_deviation": summary["standard_deviation"],
        "minimum_reward": summary["minimum"],
        "maximum_reward": summary["maximum"],
        "solved_count": summary["solved_count"],
        "solved_percentage": summary["solved_percentage"],
        "training_seconds": training_seconds,
    }


def save_summary_rows(summary_rows, run_type):
    """Save a compact table that compares every completed model."""

    summary_file = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / (run_type + "_experiment_summary.csv")
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    column_names = [
        "experiment_name",
        "training_seed",
        "training_episodes",
        "evaluation_episodes",
        "mean_reward",
        "median_reward",
        "reward_standard_deviation",
        "minimum_reward",
        "maximum_reward",
        "solved_count",
        "solved_percentage",
        "training_seconds",
    ]

    with summary_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_file


def run_one_model(configuration, planned_run):
    """Train and evaluate one model, or safely continue an existing run."""

    output_paths = get_expected_output_paths(planned_run)
    run_state = classify_run_state(output_paths)
    experiment_name = planned_run["experiment_name"]
    training_seed = planned_run["training_seed"]

    if run_state == "incomplete":
        raise RuntimeError(
            "Some, but not all, expected files exist for "
            + experiment_name
            + " seed "
            + str(training_seed)
            + ". Inspect those files before continuing."
        )

    if run_state == "complete":
        print("This model is already complete; loading its saved results.")
        training_metrics = load_training_metrics(
            output_paths["training_metrics"]
        )
        evaluation_metrics = load_evaluation_metrics(
            output_paths["evaluation_metrics"]
        )
        return make_summary_row(
            planned_run,
            training_metrics,
            evaluation_metrics,
        )

    if run_state == "pending":
        print("Training model...")
        (
            training_metrics,
            metrics_file,
            plot_file,
            checkpoint_file,
        ) = train_one_model(
            configuration,
            experiment_name,
            training_seed,
            planned_run["number_of_episodes"],
            planned_run["run_type"],
        )
        print("Training metrics:", metrics_file)
        print("Training plot:", plot_file)
        print("Local checkpoint:", checkpoint_file)
    else:
        # The training files are complete, so an interrupted run can continue
        # directly with evaluation instead of repeating a long training job.
        print("Training already finished; continuing with evaluation.")
        training_metrics = load_training_metrics(
            output_paths["training_metrics"]
        )

    print("Evaluating saved model with greedy actions...")
    (
        evaluation_metrics,
        evaluation_summary,
        evaluation_metrics_file,
        evaluation_plot_file,
        checkpoint_file,
    ) = evaluate_saved_checkpoint(
        configuration,
        experiment_name,
        training_seed,
        planned_run["evaluation_episodes"],
        planned_run["first_evaluation_seed"],
        planned_run["run_type"],
    )
    print("Evaluation mean reward:", round(evaluation_summary["mean"], 2))
    print("Evaluation metrics:", evaluation_metrics_file)
    print("Evaluation plot:", evaluation_plot_file)

    return make_summary_row(
        planned_run,
        training_metrics,
        evaluation_metrics,
    )


def print_run_plan(run_plan):
    """Display the exact work that a command would perform."""

    print("Controlled DQN experiment plan")
    print("------------------------------")

    for run_number, planned_run in enumerate(run_plan, start=1):
        print(
            str(run_number) + ".",
            planned_run["experiment_name"],
            "- training seed",
            planned_run["training_seed"],
            "-",
            planned_run["number_of_episodes"],
            "training episodes -",
            planned_run["evaluation_episodes"],
            "evaluation episodes",
        )


def run_all_models(configuration, run_plan):
    """Run every planned model and update the summary after each one."""

    summary_rows = []
    summary_file = None

    for run_number, planned_run in enumerate(run_plan, start=1):
        print("\nModel", run_number, "of", len(run_plan))
        print("Experiment:", planned_run["experiment_name"])
        print("Training seed:", planned_run["training_seed"])

        summary_row = run_one_model(configuration, planned_run)
        summary_rows.append(summary_row)

        # Saving after every model preserves completed results if a later run
        # is interrupted. On restart, completed model files are loaded again.
        summary_file = save_summary_rows(
            summary_rows,
            planned_run["run_type"],
        )
        print("Updated comparison table:", summary_file)

    return summary_rows, summary_file


def parse_arguments():
    """Read the selected command-line run mode."""

    parser = argparse.ArgumentParser(
        description="Plan, smoke-test, or run the controlled DQN experiment."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 training and 2 evaluation episodes per schedule.",
    )
    mode_group.add_argument(
        "--run-final",
        action="store_true",
        help="Run all six final 800-episode models.",
    )
    return parser.parse_args()


def main():
    """Show the plan by default or run the explicitly selected mode."""

    arguments = parse_arguments()
    configuration = load_configuration()

    if arguments.smoke_test:
        run_type = "smoke"
    else:
        run_type = "final"

    run_plan = build_run_plan(configuration, run_type)
    print_run_plan(run_plan)

    if not arguments.smoke_test and not arguments.run_final:
        print("\nPlan only: no training was started.")
        print("Use --smoke-test for the quick pipeline check.")
        print("Use --run-final only when ready for the long experiment.")
        return

    if arguments.smoke_test:
        print("\nStarting the small pipeline smoke test.")
    else:
        print("\nStarting all six final training and evaluation runs.")

    summary_rows, summary_file = run_all_models(configuration, run_plan)
    print("\nCompleted models:", len(summary_rows))
    print("Final comparison table:", summary_file)


if __name__ == "__main__":
    main()

"""Automated checks for the controlled DQN experiment runner."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src" / "lunar_lander_rl"
CONFIG_FILE = PROJECT_ROOT / "configs" / "dqn_experiments.json"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from run_controlled_experiments import (  # noqa: E402
    build_run_plan,
    classify_run_state,
    make_summary_row,
)


def load_test_configuration():
    """Load a fresh configuration dictionary for each test."""

    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def make_temporary_output_paths(directory):
    """Create path names matching the five files from one model."""

    return {
        "training_metrics": directory / "training.csv",
        "training_plot": directory / "training.png",
        "checkpoint": directory / "checkpoint.pt",
        "evaluation_metrics": directory / "evaluation.csv",
        "evaluation_plot": directory / "evaluation.png",
    }


class RunPlanTests(unittest.TestCase):
    """Check that the final comparison follows the experiment design."""

    def test_final_plan_contains_six_models(self):
        """Two schedules and three seeds should produce six model runs."""

        configuration = load_test_configuration()
        run_plan = build_run_plan(configuration, "final")

        self.assertEqual(len(run_plan), 6)

        experiment_and_seed_pairs = set()

        for planned_run in run_plan:
            experiment_and_seed_pairs.add(
                (
                    planned_run["experiment_name"],
                    planned_run["training_seed"],
                )
            )
            self.assertEqual(planned_run["number_of_episodes"], 800)
            self.assertEqual(planned_run["evaluation_episodes"], 100)
            self.assertEqual(planned_run["first_evaluation_seed"], 1000)

        self.assertEqual(len(experiment_and_seed_pairs), 6)

    def test_smoke_plan_checks_both_schedules(self):
        """The quick check should visit both schedules with one seed."""

        configuration = load_test_configuration()
        run_plan = build_run_plan(configuration, "smoke")

        self.assertEqual(len(run_plan), 2)
        self.assertEqual(run_plan[0]["number_of_episodes"], 2)
        self.assertEqual(run_plan[1]["number_of_episodes"], 2)
        self.assertNotEqual(
            run_plan[0]["experiment_name"],
            run_plan[1]["experiment_name"],
        )


class RunStateTests(unittest.TestCase):
    """Check safe restart decisions using tiny temporary files."""

    def test_no_files_means_training_is_pending(self):
        """A new model should begin with no output files."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output_paths = make_temporary_output_paths(directory)

            self.assertEqual(classify_run_state(output_paths), "pending")

    def test_training_files_can_continue_to_evaluation(self):
        """Three complete training files should not require retraining."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output_paths = make_temporary_output_paths(directory)

            output_paths["training_metrics"].touch()
            output_paths["training_plot"].touch()
            output_paths["checkpoint"].touch()

            self.assertEqual(classify_run_state(output_paths), "trained")

    def test_all_five_files_means_run_is_complete(self):
        """Training plus evaluation files should be safely reusable."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output_paths = make_temporary_output_paths(directory)

            for output_path in output_paths.values():
                output_path.touch()

            self.assertEqual(classify_run_state(output_paths), "complete")

    def test_one_file_is_treated_as_incomplete(self):
        """A partial group should stop instead of overwriting silently."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output_paths = make_temporary_output_paths(directory)
            output_paths["training_metrics"].touch()

            self.assertEqual(classify_run_state(output_paths), "incomplete")


class SummaryRowTests(unittest.TestCase):
    """Check the compact comparison row with simple rewards."""

    def test_summary_row_combines_training_and_evaluation(self):
        """The result should include episode counts, reward, and time."""

        planned_run = {
            "experiment_name": "example_schedule",
            "training_seed": 7,
            "number_of_episodes": 2,
            "evaluation_episodes": 2,
        }
        training_metrics = [
            {"elapsed_seconds": 5.0},
            {"elapsed_seconds": 12.5},
        ]
        evaluation_metrics = [
            {"total_reward": -10.0, "solved": False},
            {"total_reward": 30.0, "solved": True},
        ]

        summary_row = make_summary_row(
            planned_run,
            training_metrics,
            evaluation_metrics,
        )

        self.assertEqual(summary_row["training_episodes"], 2)
        self.assertEqual(summary_row["evaluation_episodes"], 2)
        self.assertEqual(summary_row["mean_reward"], 10.0)
        self.assertEqual(summary_row["solved_count"], 1)
        self.assertEqual(summary_row["training_seconds"], 12.5)

    def test_summary_rejects_wrong_episode_count(self):
        """A short or damaged saved CSV must not be labeled as final."""

        planned_run = {
            "experiment_name": "example_schedule",
            "training_seed": 7,
            "number_of_episodes": 800,
            "evaluation_episodes": 100,
        }
        training_metrics = [{"elapsed_seconds": 5.0}]
        evaluation_metrics = [
            {"total_reward": 1.0, "solved": False}
        ]

        with self.assertRaisesRegex(ValueError, "800 were expected"):
            make_summary_row(
                planned_run,
                training_metrics,
                evaluation_metrics,
            )


if __name__ == "__main__":
    unittest.main()

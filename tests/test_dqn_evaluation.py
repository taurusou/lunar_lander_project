"""Automated checks for DQN evaluation helpers."""

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src" / "lunar_lander_rl"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from dqn_components import DQNNetwork  # noqa: E402
from evaluate_dqn import (  # noqa: E402
    calculate_summary,
    copy_model_parameters,
    model_parameters_match,
    print_summary,
    select_greedy_action,
)


class GreedyActionTests(unittest.TestCase):
    """Check that evaluation chooses only the largest Q-value."""

    def test_greedy_action_uses_largest_q_value(self):
        """Output action 2 when its output bias is the largest."""

        policy_network = DQNNetwork(8, 4, [128, 128])

        with torch.no_grad():
            for parameter in policy_network.parameters():
                parameter.fill_(0.0)

            policy_network.output_layer.bias.copy_(
                torch.tensor([-1.0, 0.0, 3.0, 1.0])
            )

        observation = [0.0] * 8
        action = select_greedy_action(
            observation,
            policy_network,
            torch.device("cpu"),
        )

        self.assertEqual(action, 2)


class EvaluationSummaryTests(unittest.TestCase):
    """Check evaluation statistics with easy-to-verify values."""

    def test_summary_calculates_expected_values(self):
        """Rewards -1, 2, and 5 should have mean and median 2."""

        evaluation_metrics = [
            {"total_reward": -1.0, "solved": False},
            {"total_reward": 2.0, "solved": False},
            {"total_reward": 5.0, "solved": True},
        ]

        summary = calculate_summary(evaluation_metrics)

        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["minimum"], -1.0)
        self.assertEqual(summary["maximum"], 5.0)
        self.assertEqual(summary["solved_count"], 1)
        self.assertAlmostEqual(summary["solved_percentage"], 100.0 / 3.0)

    def test_print_summary_includes_all_main_statistics(self):
        """The student-facing console summary should not omit its last lines."""

        summary = {
            "mean": 1.0,
            "median": 2.0,
            "standard_deviation": 3.0,
            "minimum": -4.0,
            "maximum": 5.0,
            "solved_count": 1,
            "solved_percentage": 25.0,
        }
        printed_output = StringIO()

        with redirect_stdout(printed_output):
            print_summary(summary, number_of_episodes=4)

        output_text = printed_output.getvalue()
        self.assertIn("Minimum reward: -4.0", output_text)
        self.assertIn("Maximum reward: 5.0", output_text)
        self.assertIn("Solved percentage: 25.0 %", output_text)


class ParameterSafetyTests(unittest.TestCase):
    """Check the guard that proves evaluation does not train the model."""

    def test_parameter_copies_match_unchanged_model(self):
        """An untouched network should still match its copied parameters."""

        policy_network = DQNNetwork(8, 4, [128, 128])
        parameter_copies = copy_model_parameters(policy_network)

        self.assertTrue(
            model_parameters_match(parameter_copies, policy_network)
        )

    def test_parameter_guard_detects_a_change(self):
        """Changing one parameter should make the safety check return False."""

        policy_network = DQNNetwork(8, 4, [128, 128])
        parameter_copies = copy_model_parameters(policy_network)

        with torch.no_grad():
            first_parameter = next(policy_network.parameters())
            first_parameter[0, 0] = first_parameter[0, 0] + 1.0

        self.assertFalse(
            model_parameters_match(parameter_copies, policy_network)
        )


if __name__ == "__main__":
    unittest.main()

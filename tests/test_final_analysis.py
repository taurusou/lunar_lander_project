"""Automated checks for the final schedule-level analysis."""

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src" / "lunar_lander_rl"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from analyze_final_results import (  # noqa: E402
    calculate_schedule_summary,
    make_display_name,
)


class ScheduleSummaryTests(unittest.TestCase):
    """Check aggregate calculations with small hand-verifiable values."""

    def test_schedule_summary_combines_models_and_rewards(self):
        """Two seeds and four rewards should produce the expected summary."""

        model_rows = [
            {
                "mean_reward": 1.0,
                "solved_count": 1,
                "training_seconds": 60.0,
            },
            {
                "mean_reward": 3.0,
                "solved_count": 2,
                "training_seconds": 120.0,
            },
        ]
        rewards = [-1.0, 1.0, 3.0, 5.0]

        summary = calculate_schedule_summary(
            "example_schedule",
            model_rows,
            rewards,
        )

        self.assertEqual(summary["trained_models"], 2)
        self.assertEqual(summary["evaluation_episodes"], 4)
        self.assertEqual(summary["pooled_mean_reward"], 2.0)
        self.assertEqual(summary["pooled_median_reward"], 2.0)
        self.assertAlmostEqual(
            summary["pooled_reward_standard_deviation"],
            math.sqrt(20.0 / 3.0),
        )
        self.assertEqual(summary["solved_count"], 3)
        self.assertEqual(summary["solved_percentage"], 75.0)
        self.assertAlmostEqual(
            summary["seed_mean_standard_deviation"],
            math.sqrt(2.0),
        )
        self.assertEqual(summary["total_training_minutes"], 3.0)

    def test_empty_models_are_rejected(self):
        """A schedule cannot be summarized without trained models."""

        with self.assertRaisesRegex(ValueError, "at least one trained model"):
            calculate_schedule_summary("empty", [], [1.0])

    def test_fast_schedule_has_readable_display_name(self):
        """Configuration-style names should become report-friendly labels."""

        self.assertEqual(
            make_display_name("fast_epsilon_decay"),
            "Fast decay",
        )


if __name__ == "__main__":
    unittest.main()

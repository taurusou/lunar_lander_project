"""Automated checks for small DQN training-loop helpers."""

import json
import random
import sys
import unittest
from pathlib import Path

import numpy
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src" / "lunar_lander_rl"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from dqn_components import DQNNetwork, ReplayMemory  # noqa: E402
from train_dqn import (  # noqa: E402
    calculate_epsilon,
    optimize_model,
    soft_update_target_network,
)


CONFIG_FILE = PROJECT_ROOT / "configs" / "dqn_experiments.json"


def load_test_configuration():
    """Load a fresh configuration dictionary for each test."""

    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        configuration = json.load(config_file)

    return configuration


class EpsilonTests(unittest.TestCase):
    """Check the independent variable used by the experiment."""

    def test_epsilon_starts_at_one(self):
        """Both schedules should begin with complete exploration."""

        configuration = load_test_configuration()

        for experiment in configuration["experiments"]:
            epsilon = calculate_epsilon(experiment, episode_number=0)
            self.assertEqual(epsilon, 1.0)

    def test_fast_schedule_decays_faster(self):
        """Fast epsilon should be smaller than gradual epsilon at episode 100."""

        configuration = load_test_configuration()
        fast_experiment = configuration["experiments"][0]
        gradual_experiment = configuration["experiments"][1]

        fast_epsilon = calculate_epsilon(
            fast_experiment,
            episode_number=100,
        )
        gradual_epsilon = calculate_epsilon(
            gradual_experiment,
            episode_number=100,
        )

        self.assertLess(fast_epsilon, gradual_epsilon)

    def test_epsilon_does_not_fall_below_minimum(self):
        """A very large episode number should still return epsilon 0.05."""

        configuration = load_test_configuration()
        fast_experiment = configuration["experiments"][0]
        epsilon = calculate_epsilon(
            fast_experiment,
            episode_number=10_000,
        )

        self.assertEqual(epsilon, fast_experiment["epsilon_end"])


class TargetNetworkTests(unittest.TestCase):
    """Check that the target network moves slowly toward the policy network."""

    def test_soft_update_uses_tau(self):
        """Policy 1.0 and target 0.0 should produce target 0.25 at tau 0.25."""

        policy_network = DQNNetwork(8, 4, [128, 128])
        target_network = DQNNetwork(8, 4, [128, 128])

        with torch.no_grad():
            for parameter in policy_network.parameters():
                parameter.fill_(1.0)

            for parameter in target_network.parameters():
                parameter.fill_(0.0)

        soft_update_target_network(
            policy_network,
            target_network,
            tau=0.25,
        )

        for parameter in target_network.parameters():
            expected_values = torch.full_like(parameter, 0.25)
            self.assertTrue(torch.allclose(parameter, expected_values))


class OptimizationTests(unittest.TestCase):
    """Check that a replay batch can perform one real gradient update."""

    def test_optimization_returns_loss_and_changes_policy(self):
        """A full test batch should update at least one network parameter."""

        random.seed(0)
        numpy.random.seed(0)
        torch.manual_seed(0)

        configuration = load_test_configuration()

        # Tiny test values make this check fast. The actual JSON values remain
        # unchanged for real training.
        configuration["optimization"]["batch_size"] = 4
        configuration["replay_memory"]["minimum_size_before_training"] = 4

        policy_network = DQNNetwork(8, 4, [128, 128])
        target_network = DQNNetwork(8, 4, [128, 128])
        target_network.load_state_dict(policy_network.state_dict())

        optimizer = torch.optim.AdamW(
            policy_network.parameters(),
            lr=configuration["optimization"]["learning_rate"],
        )

        replay_memory = ReplayMemory(capacity=10)

        for transition_number in range(4):
            observation = numpy.full(
                8,
                float(transition_number),
                dtype=numpy.float32,
            )
            next_observation = observation + 0.1

            replay_memory.push(
                observation,
                transition_number % 4,
                float(transition_number),
                next_observation,
                False,
                False,
            )

        parameters_before_update = []

        for parameter in policy_network.parameters():
            parameters_before_update.append(parameter.detach().clone())

        loss = optimize_model(
            policy_network,
            target_network,
            replay_memory,
            optimizer,
            configuration,
            torch.device("cpu"),
        )

        at_least_one_parameter_changed = False

        for old_parameter, new_parameter in zip(
            parameters_before_update,
            policy_network.parameters(),
        ):
            if not torch.equal(old_parameter, new_parameter):
                at_least_one_parameter_changed = True

        self.assertIsInstance(loss, float)
        self.assertTrue(at_least_one_parameter_changed)


if __name__ == "__main__":
    unittest.main()


"""Automated checks for the DQN network and replay memory."""

import json
import random
import sys
import unittest
from pathlib import Path

import gymnasium as gym
import torch


# Add the source folder so the test can import the project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src" / "lunar_lander_rl"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from dqn_components import DQNNetwork, ReplayMemory  # noqa: E402


CONFIG_FILE = PROJECT_ROOT / "configs" / "dqn_experiments.json"


def load_test_configuration():
    """Load the same settings used by the training program."""
    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        configuration = json.load(config_file)

    return configuration


def make_test_transition(number):
    """Create a small, recognizable transition for replay-memory tests."""

    observation = torch.tensor([float(number)])
    action = number % 4
    reward = float(number)
    next_observation = torch.tensor([float(number + 1)])
    terminated = number % 2 == 0
    truncated = False

    return (
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
    )


class DQNNetworkTests(unittest.TestCase):
    """Check the network's input, output, and learning-related behavior."""

    @classmethod
    def setUpClass(cls):
        """Read shared network settings once before running these tests."""

        configuration = load_test_configuration()
        cls.observation_count = configuration["environment"]["observation_count"]
        cls.action_count = configuration["environment"]["action_count"]
        cls.hidden_layer_sizes = configuration["network"]["hidden_layer_sizes"]

    def setUp(self):
        """Create the same predictable model before each network test."""

        torch.manual_seed(0)
        self.network = DQNNetwork(
            self.observation_count,
            self.action_count,
            self.hidden_layer_sizes,
        )

    def test_one_observation_produces_four_q_values(self):
        """One LunarLander observation should produce one row of 4 Q-values."""

        observation = torch.zeros((1, self.observation_count))
        q_values = self.network(observation)

        self.assertEqual(q_values.shape, (1, self.action_count))

    def test_batch_preserves_the_number_of_rows(self):
        """A batch of 5 observations should produce 5 rows of Q-values."""

        observations = torch.zeros((5, self.observation_count))
        q_values = self.network(observations)

        self.assertEqual(q_values.shape, (5, self.action_count))

    def test_network_parameters_receive_gradients(self):
        """Backpropagation should calculate a gradient for every parameter."""

        observations = torch.ones((2, self.observation_count))
        q_values = self.network(observations)

        # This artificial loss is only for checking the gradient path. The real
        # training program will use Smooth L1 loss from the configuration.
        test_loss = q_values.sum()
        test_loss.backward()

        for parameter in self.network.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_real_lunarlander_observation_can_enter_network(self):
        """The environment's actual observation should fit the network input."""

        environment = gym.make("LunarLander-v3")

        try:
            observation, information = environment.reset(seed=0)

            # PyTorch networks work with batches. unsqueeze adds a batch
            # dimension, changing the shape from (8,) to (1, 8).
            observation_tensor = torch.tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            q_values = self.network(observation_tensor)
            self.assertEqual(q_values.shape, (1, self.action_count))
        finally:
            environment.close()


class ReplayMemoryTests(unittest.TestCase):
    """Check storage, replacement, and random sampling."""

    def test_push_increases_memory_length(self):
        """Each push should add a transition until capacity is reached."""

        memory = ReplayMemory(capacity=3)
        memory.push(*make_test_transition(0))
        memory.push(*make_test_transition(1))

        self.assertEqual(len(memory), 2)

    def test_oldest_transition_is_replaced_when_full(self):
        """A fourth item in capacity 3 should replace the first item."""

        memory = ReplayMemory(capacity=3)

        for number in range(4):
            memory.push(*make_test_transition(number))

        stored_rewards = []

        for transition in memory.transitions:
            stored_rewards.append(transition.reward)

        self.assertEqual(len(memory), 3)
        self.assertNotIn(0.0, stored_rewards)
        self.assertCountEqual(stored_rewards, [1.0, 2.0, 3.0])

    def test_sample_returns_requested_number_without_removing_items(self):
        """Sampling should return a batch while leaving memory unchanged."""

        random.seed(0)
        memory = ReplayMemory(capacity=5)

        for number in range(5):
            memory.push(*make_test_transition(number))

        batch = memory.sample(batch_size=3)

        self.assertEqual(len(batch), 3)
        self.assertEqual(len(memory), 5)

    def test_sample_rejects_a_batch_larger_than_memory(self):
        """Sampling too early should produce a clear error."""

        memory = ReplayMemory(capacity=5)
        memory.push(*make_test_transition(0))

        with self.assertRaises(ValueError):
            memory.sample(batch_size=2)

    def test_transition_keeps_both_ending_flags(self):
        """Termination and truncation must remain separate for DQN targets."""

        memory = ReplayMemory(capacity=1)
        transition_values = make_test_transition(0)
        memory.push(*transition_values)
        stored_transition = memory.transitions[0]

        self.assertTrue(stored_transition.terminated)
        self.assertFalse(stored_transition.truncated)


if __name__ == "__main__":
    unittest.main()

"""Small building blocks used by the future DQN training program.

This module contains:

1. DQNNetwork, which predicts one Q-value for each possible action.
2. ReplayMemory, which stores past experiences for random sampling.

These components are based on the main ideas in PyTorch's official DQN
tutorial, but they are written here with beginner-oriented names and comments:
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random

import torch
from torch import nn


class DQNNetwork(nn.Module):
    """A neural network that predicts Q-values from an observation.

    For LunarLander-v3:

    - The input contains 8 observation values.
    - The output contains 4 Q-values, one for each possible action.
    - Larger Q-values represent actions the network currently expects to
      produce larger future rewards.
    """

    def __init__(
        self,
        observation_count,
        action_count,
        hidden_layer_sizes,
    ):
        """Create the network layers using values from the configuration."""

        # nn.Module performs important PyTorch setup. A subclass should always
        # call its parent constructor before creating layers.
        super().__init__()

        if observation_count <= 0:
            raise ValueError("observation_count must be greater than zero.")

        if action_count <= 0:
            raise ValueError("action_count must be greater than zero.")

        if len(hidden_layer_sizes) != 2:
            raise ValueError("This beginner network requires two hidden layers.")

        first_hidden_size = hidden_layer_sizes[0]
        second_hidden_size = hidden_layer_sizes[1]

        if first_hidden_size <= 0 or second_hidden_size <= 0:
            raise ValueError("Hidden-layer sizes must be greater than zero.")

        # A Linear layer connects every value from the previous layer to every
        # neuron in the next layer.
        self.input_layer = nn.Linear(observation_count, first_hidden_size)
        self.hidden_layer = nn.Linear(first_hidden_size, second_hidden_size)
        self.output_layer = nn.Linear(second_hidden_size, action_count)

    def forward(self, observations):
        """Pass one observation or a batch of observations through the model."""

        # ReLU replaces negative hidden values with zero. This gives the
        # network the nonlinearity needed to learn more than a straight line.
        hidden_values = torch.relu(self.input_layer(observations))
        hidden_values = torch.relu(self.hidden_layer(hidden_values))

        # Do not apply ReLU or softmax to the output. Q-values are estimates of
        # future reward, so they may be negative or positive and do not need to
        # add up to 1.
        q_values = self.output_layer(hidden_values)
        return q_values


class Transition:
    """One experience collected while the agent interacts with the environment."""

    def __init__(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
    ):
        """Store all information needed for one future DQN update."""

        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.terminated = terminated
        self.truncated = truncated


class ReplayMemory:
    """A fixed-size collection of past transitions.

    When the memory is full, each new transition replaces the oldest one.
    Random sampling breaks up the strong ordering between consecutive game
    steps and lets the agent reuse earlier experiences.
    """

    def __init__(self, capacity):
        """Create an empty memory with a maximum number of transitions."""

        if capacity <= 0:
            raise ValueError("Replay-memory capacity must be greater than zero.")

        self.capacity = capacity
        self.transitions = []
        self.next_position = 0

    def push(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
    ):
        """Add one transition, replacing the oldest transition when full."""

        transition = Transition(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )

        if len(self.transitions) < self.capacity:
            # During initial collection, the memory still has empty space.
            self.transitions.append(transition)
        else:
            # Once full, next_position identifies the oldest item.
            self.transitions[self.next_position] = transition

        # The remainder operator wraps the position back to zero after it
        # reaches capacity. This makes the list act like a circular buffer.
        self.next_position = (self.next_position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a random batch without removing transitions from memory."""

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if batch_size > len(self.transitions):
            raise ValueError(
                "Cannot sample more transitions than the memory currently holds."
            )

        sampled_transitions = random.sample(self.transitions, batch_size)
        return sampled_transitions

    def __len__(self):
        """Allow len(memory) to report the number of stored transitions."""

        return len(self.transitions)


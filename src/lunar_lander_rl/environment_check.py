"""Check that the project can run the LunarLander environment.

This file is only a setup test. The actions are random, so this is not yet a
learning agent.

The basic reset/step loop follows Gymnasium's official usage pattern:
https://gymnasium.farama.org/main/introduction/basic_usage/
"""

import sys

import gymnasium as gym
import matplotlib
import numpy
import torch


# A seed makes the random test repeatable. Reproducibility will be important
# when we compare agents later in the project.
RANDOM_SEED = 42

# LunarLander normally ends an episode on its own. This extra limit protects us
# from an accidental endless loop if the environment is used incorrectly.
SAFETY_STEP_LIMIT = 2_000


def print_package_versions():
    """Print useful version information for troubleshooting and the report."""

    print("Python:", sys.version.split()[0])
    print("Gymnasium:", gym.__version__)
    print("NumPy:", numpy.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("PyTorch:", torch.__version__)


def run_random_episode():
    """Run one complete episode using random actions.

    A real agent will use the observation to choose an intelligent action. For
    this setup test, sampling random actions is enough to prove that the
    environment, Box2D physics, and rendering dependencies work.
    """

    # rgb_array rendering returns each picture frame as an array. It lets us
    # test rendering without opening a separate game window.
    environment = gym.make("LunarLander-v3", render_mode="rgb_array")

    try:
        # reset starts a new episode. Gymnasium returns both the first
        # observation and an information dictionary.
        observation, information = environment.reset(seed=RANDOM_SEED)

        # The action sampler has its own random-number generator, so we seed it
        # separately to make the random action sequence repeatable.
        environment.action_space.seed(RANDOM_SEED)

        print("\nEnvironment: LunarLander-v3")
        print("Observation space:", environment.observation_space)
        print("Action space:", environment.action_space)
        print("First observation:", observation)

        # Confirm that the initial observation has the type and range promised
        # by the environment's observation space.
        if not environment.observation_space.contains(observation):
            raise ValueError("The first observation is outside the expected space.")

        # Rendering one frame checks that Pygame and the graphical part of the
        # environment are installed. The frame should be a height x width x
        # color array.
        first_frame = environment.render()
        print("Rendered frame shape:", first_frame.shape)

        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False

        # terminated means the lander finished naturally, for example by
        # landing or crashing. truncated means an outside limit, such as the
        # maximum episode length, stopped the episode.
        while not terminated and not truncated:
            action = environment.action_space.sample()

            # step applies one action and returns the result of that action.
            observation, reward, terminated, truncated, information = (
                environment.step(action)
            )

            total_reward = total_reward + float(reward)
            step_count = step_count + 1

            if step_count >= SAFETY_STEP_LIMIT:
                raise RuntimeError("The episode exceeded the safety step limit.")

        print("Episode steps:", step_count)
        print("Episode reward:", round(total_reward, 2))
        print("Ended by termination:", terminated)
        print("Ended by truncation:", truncated)

    finally:
        # close releases the environment's rendering and physics resources,
        # even if an error occurs during the test.
        environment.close()


def main():
    """Run all parts of the environment check."""

    print_package_versions()
    run_random_episode()
    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()


"""Record one demonstration episode using a trained DQN checkpoint."""

from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

from dqn_components import DQNNetwork


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# This model had a strong and consistent final evaluation result.
CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "final_fast_epsilon_decay_seed_1.pt"
)

VIDEO_FOLDER = PROJECT_ROOT / "artifacts" / "videos"

# Seed 1004 previously produced a successful landing with a reward around 269.
DEMO_SEED = 1004


def choose_greedy_action(observation, policy_network, device):
    """Choose the action with the largest predicted Q-value."""

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # Recording is evaluation only, so gradients are not needed.
    with torch.no_grad():
        q_values = policy_network(observation_tensor)

    action = int(q_values.argmax(dim=1).item())
    return action


def record_first_episode(episode_number):
    """Tell Gymnasium to record only the first episode."""

    return episode_number == 0


def main():
    """Load the trained model and record one episode."""

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            "The trained checkpoint was not found:\n"
            + str(CHECKPOINT_FILE)
        )

    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
        weights_only=False,
    )

    configuration = checkpoint["configuration"]
    environment_settings = configuration["environment"]
    network_settings = configuration["network"]

    policy_network = DQNNetwork(
        environment_settings["observation_count"],
        environment_settings["action_count"],
        network_settings["hidden_layer_sizes"],
    ).to(device)

    policy_network.load_state_dict(
        checkpoint["policy_network_state"]
    )
    policy_network.eval()

    VIDEO_FOLDER.mkdir(parents=True, exist_ok=True)

    environment = gym.make(
        environment_settings["name"],
        render_mode="rgb_array",
    )

    environment = RecordVideo(
        environment,
        video_folder=str(VIDEO_FOLDER),
        name_prefix="fast-decay-dqn-demo",
        episode_trigger=record_first_episode,
        disable_logger=True,
    )

    try:
        observation, information = environment.reset(seed=DEMO_SEED)

        terminated = False
        truncated = False
        total_reward = 0.0
        number_of_steps = 0

        while not terminated and not truncated:
            action = choose_greedy_action(
                observation,
                policy_network,
                device,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                information,
            ) = environment.step(action)

            total_reward = total_reward + float(reward)
            number_of_steps = number_of_steps + 1
    finally:
        # Closing the wrapper finishes and saves the video file.
        environment.close()

    print("Demo episode finished.")
    print("Seed:", DEMO_SEED)
    print("Total reward:", round(total_reward, 2))
    print("Number of steps:", number_of_steps)
    print("Video folder:", VIDEO_FOLDER)


if __name__ == "__main__":
    main()

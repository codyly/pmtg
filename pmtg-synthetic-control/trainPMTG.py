""" Training Script

Author: Ren
"""
import numpy as np

import torch

from modulated_trajectory_generator import TrajectoryGenerator
from net import VanillaPMTG
from utils.loader import load_trajectory
from utils.visualize import view_reward_curve

torch.manual_seed(0)

TRAJECTORY_FILE = "trajectory.npy"
NUM_STEPS_ROLLOUT = 1000
NUM_TRAINING_ITERS = 500
NUM_BATCH_SIZE = 1

WEIGHTS_DIR = "weights"


def loss_fn(coord_, target_):
    return 1.0 - torch.exp(-torch.mean(torch.sqrt(torch.sum(torch.square(coord_ - target_), axis=0))))


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    target = torch.from_numpy(load_trajectory(TRAJECTORY_FILE, NUM_STEPS_ROLLOUT)).to(device=device, dtype=float)
    model = VanillaPMTG().to(device=device, dtype=float)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    num_rollouts_train = NUM_TRAINING_ITERS
    steps = NUM_STEPS_ROLLOUT

    batch_size = NUM_BATCH_SIZE

    tg = TrajectoryGenerator(device=device, num_envs=batch_size)
    dt = 1.0 / steps

    rewards_log = np.zeros(num_rollouts_train)

    for i in range(num_rollouts_train):

        # initialize feedback states (x, y) cache
        feedback = torch.zeros([batch_size, 2]).to(device=device, dtype=float)
        episode_avg_reward = 0.0

        tg.reset()

        for step in range(steps):
            tg_phase = tg.get_phase().to(device=device, dtype=float)
            policy_net_input = torch.cat([feedback, tg_phase], dim=1)
            target_coords = target[step]

            # Compute prediction error
            policy_net_output = model(policy_net_input)

            u_tg = tg.get_action(dt, ax=policy_net_output[:, 0], ay=policy_net_output[:, 1])
            u_fb = policy_net_output[:, 2:]

            # predicted_next_coords
            u = u_tg + u_fb

            loss = loss_fn(u, target_coords)

            # simply use (1-loss) for reward
            episode_avg_reward += 1 - loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            feedback = u.detach()

        episode_avg_reward /= steps
        rewards_log[i] = episode_avg_reward
        if i % 10 == 0:
            print(f"episode_avg_reward: {episode_avg_reward:>4f}  [{i:>5d}/{num_rollouts_train:>5d}]")

    view_reward_curve(rewards_log)
    torch.save(model.state_dict(), f"{WEIGHTS_DIR}/model-pmtg.pth")
    print(f"Saved PyTorch Model State to {WEIGHTS_DIR}/model-pmtg.pth")


if __name__ == "__main__":
    train()

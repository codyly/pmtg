from typing import List

import numpy as np
from matplotlib import pyplot as plt


def view_trajectory(trajectory: np.ndarray, title: str = "Trajecotry", root: str = "figs"):
    plt.clf()
    plt.title(title)
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r+", label="scatter")
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b", label="curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"{root}/{title}.png")


def view_trajectory_eval(
    trajectory: np.ndarray, target: np.ndarray, title: str = "Trajecotry Eval", root: str = "figs"
):
    plt.clf()
    plt.title(title)
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b", label="predicted curve")
    plt.plot(target[:, 0], target[:, 1], "r--", label="ground truth")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"{root}/{title}.png")


def view_reward_curve(reward: List[int], title: str = "Reward", root: str = "figs"):
    x = np.arange(len(reward))
    plt.clf()
    plt.plot(x, reward, "g", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.savefig(f"{root}/{title}.png")

""" Eval/Test Script

Author: Ren
"""
import torch

from modulated_trajectory_generator import TrajectoryGenerator
from net import VanillaPMTG
from utils.loader import load_trajectory
from utils.visualize import view_trajectory_eval

torch.manual_seed(0)

TRAJECTORY_FILE = "trajectory.npy"
NUM_STEPS_ROLLOUT_EVAL = 500
NUM_BATCH_SIZE = 1

WEIGHTS_DIR = "weights"


def loss_fn(coord_, target_):
    return 1.0 - torch.exp(-torch.mean(torch.sqrt(torch.sum(torch.square(coord_ - target_), axis=0))))


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    target = torch.from_numpy(load_trajectory(TRAJECTORY_FILE, NUM_STEPS_ROLLOUT_EVAL, False)).to(
        device=device, dtype=float
    )
    model = VanillaPMTG().to(device=device, dtype=float)
    model.load_state_dict(torch.load(f"{WEIGHTS_DIR}/model-pmtg.pth"))
    model.eval()

    num_batches = NUM_BATCH_SIZE

    steps = NUM_STEPS_ROLLOUT_EVAL
    dt = 1.0 / steps

    test_loss = 0

    tg = TrajectoryGenerator(device=device, num_envs=num_batches)
    tg.reset()

    result = torch.zeros([steps, 2])

    feedback = torch.zeros([num_batches, 2]).to(device=device, dtype=float)

    with torch.no_grad():
        for i in range(steps):
            tg_phase = tg.get_phase().to(device=device, dtype=float)
            x = torch.cat([feedback, tg_phase], dim=1)
            y = target[i]

            # Compute prediction error
            y_ = model(x)

            u_tg = tg.get_action(dt, y_[:, 0], y_[:, 1])
            u_fb = y_[:, 2:]

            u = u_tg + u_fb

            feedback = u.detach()
            result[i] = feedback[0].cpu()
            test_loss += loss_fn(u, y).item()

    test_loss /= steps

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    result = result.numpy()
    view_trajectory_eval(result, target.to("cpu").numpy())


if __name__ == "__main__":
    test()

import numpy as np

from utils.visualize import view_trajectory

DEFAULT_SCALE = 1.0


def load_trajectory(file_name: str, num_steps: int, save_results: bool = True) -> np.ndarray:
    """Load paited trajectory and pre-processing

    Args:
        file_name (string): [description]
        num_steps (int): [description]

    Returns:
        trajectory with coordinates in shape (num_steps, 2)
    """
    trajectory = np.load(file_name).astype(np.float32)

    # 1. normalization
    w, h = (trajectory.max(axis=0) - trajectory.min(axis=0)).tolist()
    ratio = w / h

    if ratio >= 1:
        trajectory = (trajectory - trajectory.min(axis=0)) / w
        scales = np.array([[DEFAULT_SCALE, h / w]])
    else:
        trajectory = (trajectory - trajectory.min(axis=0)) / h
        scales = np.array([[w / h, DEFAULT_SCALE]])

    # [-DEFAULT_SCALE, DEFAULT_SCALE], ratio kepted
    trajectory = (trajectory - scales / 2) * 2 * DEFAULT_SCALE

    if save_results:
        view_trajectory(trajectory, title="original_trajectory")

    # 2. resampling / interpolation
    # Human hardly draw the curve in the constant speed, so a rough resampling (linear interpolation)
    #   should be used for pre-processing
    margin_right = sorted(np.where(trajectory[:, 0] == scales[0, 0])[0])
    margin_left = sorted(np.where(trajectory[:, 0] == -scales[0, 0])[0])

    assert len(margin_right) >= 1 and len(margin_left) >= 1

    ts = np.linspace(0, 1, num_steps)
    xs = scales[0, 0] * np.sin(2 * np.pi * ts)
    ys = np.zeros_like(xs)

    # from left to right
    ids = np.arange(num_steps // 2 + 1) - num_steps // 4
    ids_traj = np.arange(trajectory.shape[0] - margin_left[0] + margin_right[-1]) - (
        trajectory.shape[0] - margin_left[0]
    )
    ys[ids] = np.interp(xs[ids], trajectory[ids_traj, 0], trajectory[ids_traj, 1])

    # from right to left
    ys[-num_steps // 4 - 1 : num_steps // 4 : -1] = np.interp(
        xs[-num_steps // 4 - 1 : num_steps // 4 : -1],
        trajectory[margin_left[-1] : margin_right[0] : -1, 0],
        trajectory[margin_left[-1] : margin_right[0] : -1, 1],
    )

    trajectory_interp = np.zeros([num_steps, 2])
    trajectory_interp[:, 0], trajectory_interp[:, 1] = xs, ys

    if save_results:
        view_trajectory(trajectory_interp, title="interp_trajectory")

    return trajectory_interp

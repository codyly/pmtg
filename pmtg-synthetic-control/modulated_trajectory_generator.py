""" PMTG's trajectory generator for synthetic controll problem (Part 3 of the paper)

Author: Ren Liu
"""
import math
import torch


class TrajectoryGenerator:
    """Trajectory Generator to generate handwritten 8"""

    def __init__(self, device: torch.device, num_envs: int = 1) -> None:
        """Initialize settings of Trajectory Generator

        Args:
            device (torch.device): decide whether to use cpu or gpu
            num_envs (int, optional): Number of environment (batch size). Defaults to 1.
        """

        self.device = device
        self.num_envs = num_envs
        self.t = torch.zeros(num_envs, dtype=float, device=device)

    def get_action(self, dt: torch.Tensor, ax: torch.Tensor, ay: torch.Tensor) -> torch.Tensor:
        """Get action by trajectory generator (u_tg)

        Args:
            dt (num_envs,): time step
            ax (num_envs,): amplitude x
            ay (num_envs,): amplitude y

        Returns:
            action by trajectory generator (u_tg) in shape of (num_envs, 2)
        """

        theta = 2 * math.pi * self.t
        x = ax * torch.sin(theta)
        y = 0.5 * ay * torch.sin(theta) * torch.cos(theta)

        self.t += dt
        self.t[self.t > 1] = 0

        return torch.cat([x, y], dim=0).view(2, self.num_envs).T

    def get_phase(self) -> torch.Tensor:
        """Get current time (phase)"""
        return self.t.reshape([self.num_envs, 1])

    def reset(self) -> None:
        """Reset cached time (phase)"""
        self.t = torch.zeros(self.num_envs, dtype=float, device=self.device)

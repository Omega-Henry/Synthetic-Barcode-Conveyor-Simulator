"""
Trajectory abstractions for package motion simulation.
Decoupled from renderers to allow physics-based or 3D spline trajectories in the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from barcode_simulator.core.models import Point2D


class Trajectory(ABC):
    """Abstract trajectory interface."""

    @abstractmethod
    def get_position(self, t: float) -> Point2D:
        """Returns package center position (x, y) at time t in world/pixel coordinates."""
        pass

    @abstractmethod
    def get_velocity(self, t: float) -> float:
        """Returns instantaneous linear speed in px/s or world-units/s at time t."""
        pass

    @abstractmethod
    def get_velocity_vector(self, t: float) -> Point2D:
        """Returns instantaneous velocity vector (vx, vy) at time t."""
        pass

    @abstractmethod
    def is_active(self, t: float, canvas_width: float, package_width: float) -> bool:
        """Returns whether the package is within or near the active simulation canvas."""
        pass


class LinearTrajectory(Trajectory):
    """
    Linear 1D/2D conveyor trajectory with constant speed, optional acceleration, and noise jitter.
    x(t) = x0 + dir * (v0 * (t - t0) + 0.5 * a * (t - t0)^2)
    y(t) = y0 + jitter_y(t)
    """

    def __init__(
        self,
        start_time: float,
        start_x: float,
        start_y: float,
        speed: float,
        direction: str = "left_to_right",
        acceleration: float = 0.0,
        jitter_stddev: float = 0.0,
        jitter_seed: Optional[int] = None,
    ):
        self.t0 = float(start_time)
        self.x0 = float(start_x)
        self.y0 = float(start_y)
        self.speed = float(speed)
        self.direction_factor = 1.0 if direction == "left_to_right" else -1.0
        self.acceleration = float(acceleration)
        self.jitter_stddev = float(jitter_stddev)
        self._rng = np.random.default_rng(jitter_seed if jitter_seed is not None else 42)

    def get_position(self, t: float) -> Point2D:
        dt = max(0.0, t - self.t0)
        # s(t) = v0*dt + 0.5*a*dt^2
        dist = self.speed * dt + 0.5 * self.acceleration * (dt ** 2)
        x = self.x0 + self.direction_factor * dist
        y = self.y0
        if self.jitter_stddev > 0.0:
            # Deterministic pseudo-random offset using sine harmonics
            jitter_y = self.jitter_stddev * np.sin(t * 12.0 + self.y0 * 0.1)
            y += jitter_y
        return Point2D(float(x), float(y))

    def get_velocity(self, t: float) -> float:
        dt = max(0.0, t - self.t0)
        return float(max(0.0, self.speed + self.acceleration * dt))

    def get_velocity_vector(self, t: float) -> Point2D:
        v = self.get_velocity(t)
        return Point2D(float(self.direction_factor * v), 0.0)

    def is_active(self, t: float, canvas_width: float, package_width: float) -> bool:
        pos = self.get_position(t)
        margin = package_width * 1.5 + 100.0
        if self.direction_factor > 0:  # left to right
            return (pos.x >= -margin) and (pos.x <= canvas_width + margin)
        else:  # right to left
            return (pos.x <= canvas_width + margin) and (pos.x >= -margin)

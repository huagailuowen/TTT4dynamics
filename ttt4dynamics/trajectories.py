from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


Array2 = tuple[float, float]


@dataclass(frozen=True)
class TrajectoryState:
    xy: np.ndarray
    velocity_xy: np.ndarray
    yaw: float


@dataclass(frozen=True)
class TrajectorySpec:
    """Procedural closed-loop or periodic carrier trajectory.

    Coordinates are table-frame x/y coordinates in meters. The trajectory is
    deterministic for a given spec, but parameters should be sampled across
    cases/episodes to avoid memorization.
    """

    family: str = "frozen"
    center: Array2 = (0.0, 0.0)
    amplitude: Array2 = (0.10, 0.06)
    period: float = 6.0
    phase: float = 0.0
    direction: int = 1
    yaw: float = 0.0
    phase_y: float = math.pi / 3.0
    superellipse_power: float = 4.0
    harmonics: list[dict[str, float]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectorySpec":
        payload = dict(data)
        if "center" in payload:
            payload["center"] = tuple(payload["center"])
        if "amplitude" in payload:
            payload["amplitude"] = tuple(payload["amplitude"])
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "center": list(self.center),
            "amplitude": list(self.amplitude),
            "period": self.period,
            "phase": self.phase,
            "direction": self.direction,
            "yaw": self.yaw,
            "phase_y": self.phase_y,
            "superellipse_power": self.superellipse_power,
            "harmonics": list(self.harmonics),
        }


class ProceduralTrajectory:
    def __init__(self, spec: TrajectorySpec):
        if spec.period <= 0:
            raise ValueError(f"Trajectory period must be positive, got {spec.period}")
        if spec.direction not in {-1, 1}:
            raise ValueError(f"Trajectory direction must be -1 or 1, got {spec.direction}")
        self.spec = spec

    def sample(self, t: float) -> TrajectoryState:
        xy = self._xy(float(t))
        dt = max(1e-3, self.spec.period / 2000.0)
        velocity = (self._xy(float(t) + dt) - self._xy(float(t) - dt)) / (2.0 * dt)
        return TrajectoryState(xy=xy, velocity_xy=velocity, yaw=float(self.spec.yaw))

    def envelope(self, samples: int = 720) -> np.ndarray:
        ts = np.linspace(0.0, self.spec.period, num=max(samples, 16), endpoint=False)
        return np.stack([self._xy(float(t)) for t in ts], axis=0)

    def min_distance_to_point(self, xy: Array2, samples: int = 720) -> float:
        point = np.asarray(xy, dtype=np.float64)
        path = self.envelope(samples=samples)
        return float(np.linalg.norm(path - point[None, :], axis=1).min())

    def _theta(self, t: float) -> float:
        return self.spec.direction * (2.0 * math.pi * t / self.spec.period + self.spec.phase)

    def _rotate(self, xy: np.ndarray) -> np.ndarray:
        c = math.cos(self.spec.yaw)
        s = math.sin(self.spec.yaw)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
        return rot @ xy

    def _xy(self, t: float) -> np.ndarray:
        family = self.spec.family.lower()
        center = np.asarray(self.spec.center, dtype=np.float64)
        ax, ay = self.spec.amplitude
        theta = self._theta(t)

        if family == "frozen":
            local = np.zeros(2, dtype=np.float64)
        elif family == "line":
            local = np.asarray([ax * math.sin(theta), 0.0], dtype=np.float64)
        elif family == "ellipse":
            local = np.asarray([ax * math.cos(theta), ay * math.sin(theta)], dtype=np.float64)
        elif family == "figure8":
            local = np.asarray(
                [ax * math.sin(theta), ay * math.sin(theta) * math.cos(theta)],
                dtype=np.float64,
            )
        elif family == "lissajous":
            local = np.asarray(
                [
                    ax * math.sin(theta),
                    ay * math.sin(2.0 * theta + self.spec.phase_y),
                ],
                dtype=np.float64,
            )
        elif family == "irregular_loop":
            local = np.asarray(
                [
                    ax * (0.78 * math.cos(theta) + 0.22 * math.cos(2.0 * theta + 0.6)),
                    ay * (0.88 * math.sin(theta) + 0.18 * math.sin(3.0 * theta - 0.4)),
                ],
                dtype=np.float64,
            )
            for harmonic in self.spec.harmonics:
                order = float(harmonic.get("order", 2.0))
                hx = float(harmonic.get("x", 0.0))
                hy = float(harmonic.get("y", 0.0))
                phase = float(harmonic.get("phase", 0.0))
                local += np.asarray(
                    [ax * hx * math.cos(order * theta + phase), ay * hy * math.sin(order * theta + phase)],
                    dtype=np.float64,
                )
        elif family == "rounded_rectangle":
            power = max(2.0, float(self.spec.superellipse_power))
            c = math.cos(theta)
            s = math.sin(theta)
            local = np.asarray(
                [
                    ax * math.copysign(abs(c) ** (2.0 / power), c),
                    ay * math.copysign(abs(s) ** (2.0 / power), s),
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Unsupported trajectory family: {self.spec.family}")

        return center + self._rotate(local)

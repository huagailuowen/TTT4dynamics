from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from .trajectories import ProceduralTrajectory, TrajectorySpec


@dataclass(frozen=True)
class DynamicCarrierCase:
    """One dynamic carrier task case.

    The BDDL defines the static scene and symbolic objects. This case controls
    which object is the moving carrier, which object is the payload, where the
    static target is, and how the carrier moves.
    """

    case_id: str
    access_mode: str
    carrier_name: str
    payload_name: str
    target_xy: tuple[float, float]
    motion: TrajectorySpec
    bddl_file: str | None = None
    suite_name: str | None = None
    task_id: int | None = None
    target_radius: float = 0.055
    target_z: float | None = None
    target_z_tolerance: float = 0.12
    platform_radius: float = 0.12
    object_radius: float = 0.035
    safety_margin: float = 0.05
    payload_offset_xyz: tuple[float, float, float] | None = None
    grasp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.055)
    grasp_release_distance: float = 0.035
    grasp_release_height: float = 0.05
    detach_lift_height: float = 0.045
    control_freq: float = 20.0
    max_steps: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicCarrierCase":
        payload = dict(data)
        payload["target_xy"] = tuple(payload["target_xy"])
        payload["motion"] = TrajectorySpec.from_dict(payload["motion"])
        if payload.get("payload_offset_xyz") is not None:
            payload["payload_offset_xyz"] = tuple(payload["payload_offset_xyz"])
        if payload.get("grasp_offset_xyz") is not None:
            payload["grasp_offset_xyz"] = tuple(payload["grasp_offset_xyz"])
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "access_mode": self.access_mode,
            "carrier_name": self.carrier_name,
            "payload_name": self.payload_name,
            "target_xy": list(self.target_xy),
            "motion": self.motion.as_dict(),
            "bddl_file": self.bddl_file,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "target_radius": self.target_radius,
            "target_z": self.target_z,
            "target_z_tolerance": self.target_z_tolerance,
            "platform_radius": self.platform_radius,
            "object_radius": self.object_radius,
            "safety_margin": self.safety_margin,
            "payload_offset_xyz": (
                list(self.payload_offset_xyz) if self.payload_offset_xyz is not None else None
            ),
            "grasp_offset_xyz": list(self.grasp_offset_xyz),
            "grasp_release_distance": self.grasp_release_distance,
            "grasp_release_height": self.grasp_release_height,
            "detach_lift_height": self.detach_lift_height,
            "control_freq": self.control_freq,
            "max_steps": self.max_steps,
        }

    def frozen_variant(self, case_id_suffix: str = "_frozen") -> "DynamicCarrierCase":
        frozen_motion = replace(self.motion, family="frozen", amplitude=(0.0, 0.0))
        return replace(self, case_id=f"{self.case_id}{case_id_suffix}", motion=frozen_motion)

    def validate_target_separation(self) -> float:
        """Return min target-path distance, raising if the target is too close."""
        min_dist = ProceduralTrajectory(self.motion).min_distance_to_point(self.target_xy)
        required = self.platform_radius + self.object_radius + self.safety_margin + self.target_radius
        if min_dist <= required:
            raise ValueError(
                f"Case {self.case_id} target is too close to carrier trajectory: "
                f"min_dist={min_dist:.3f}, required>{required:.3f}. "
                "Move target_xy away from the path envelope."
            )
        return min_dist

    def resolved_bddl_file(self, root: Path) -> Path | None:
        if self.bddl_file is None:
            return None
        path = Path(self.bddl_file)
        if not path.is_absolute():
            path = root / path
        return path


def load_cases(path: str | Path) -> list[DynamicCarrierCase]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["cases"] if isinstance(data, dict) and "cases" in data else data
    return [DynamicCarrierCase.from_dict(item) for item in items]


def dump_cases(path: str | Path, cases: list[DynamicCarrierCase]) -> None:
    path = Path(path)
    payload = {"cases": [case.as_dict() for case in cases]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

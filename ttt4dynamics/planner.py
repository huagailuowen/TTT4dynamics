from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .dynamic_env import DynamicCarrierEnv


class PlannerPhase(str, Enum):
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    MOVE_TO_TARGET = "move_to_target"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"
    DONE = "done"


@dataclass(frozen=True)
class PlannerConfig:
    intercept_lead_s: float = 0.35
    position_gain: float = 10.0
    max_pos_action: float = 1.0
    xy_tolerance: float = 0.018
    target_xy_tolerance: float = 0.05
    z_tolerance: float = 0.03
    flat_approach_z: float = 1.04
    box_approach_z: float = 1.12
    grasp_z_offset: float = 0.045
    lift_z: float = 1.05
    place_z: float = 1.00
    retreat_z: float = 1.10
    grasp_hold_steps: int = 8
    release_hold_steps: int = 8
    max_steps: int = 500


class ScriptedDynamicCarrierPlanner:
    """Privileged phase-based expert for dynamic carrier demos."""

    def __init__(self, env: DynamicCarrierEnv, config: PlannerConfig | None = None):
        self.env = env
        self.config = config or PlannerConfig()
        self.phase = PlannerPhase.APPROACH
        self.phase_steps = 0
        self.last_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)

    def reset(self) -> None:
        self.phase = PlannerPhase.APPROACH
        self.phase_steps = 0
        self.last_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)

    def act(self, obs: dict) -> np.ndarray:
        eef = self.env.eef_position(obs)
        target = self._phase_target()
        gripper = self._phase_gripper()
        action = self._cartesian_action(eef, target, gripper)
        self._advance_phase_if_ready(eef, target)
        self.phase_steps += 1
        self.last_action = action
        return action

    def _phase_target(self) -> np.ndarray:
        grasp_pos = self.env.payload_grasp_position()
        access = self.env.case.access_mode.lower()
        approach_z = self.config.box_approach_z if "box" in access or "tray" in access else self.config.flat_approach_z

        if self.phase == PlannerPhase.APPROACH:
            xy = self.env.predict_payload_grasp_xy(self.env.t + self.config.intercept_lead_s)
            return np.asarray([xy[0], xy[1], approach_z], dtype=np.float64)
        if self.phase == PlannerPhase.DESCEND:
            xy = self.env.predict_payload_grasp_xy(self.env.t + 0.15)
            return np.asarray([xy[0], xy[1], grasp_pos[2]], dtype=np.float64)
        if self.phase == PlannerPhase.GRASP:
            return grasp_pos
        if self.phase == PlannerPhase.LIFT:
            eef = self.env.eef_position()
            return np.asarray([eef[0], eef[1], self.config.lift_z], dtype=np.float64)
        if self.phase == PlannerPhase.MOVE_TO_TARGET:
            tx, ty = self.env.case.target_xy
            return np.asarray([tx, ty, self.config.lift_z], dtype=np.float64)
        if self.phase == PlannerPhase.LOWER:
            tx, ty = self.env.case.target_xy
            return np.asarray([tx, ty, self._place_grasp_z()], dtype=np.float64)
        if self.phase == PlannerPhase.RELEASE:
            tx, ty = self.env.case.target_xy
            return np.asarray([tx, ty, self._place_grasp_z()], dtype=np.float64)
        if self.phase == PlannerPhase.RETREAT:
            tx, ty = self.env.case.target_xy
            return np.asarray([tx, ty, self.config.retreat_z], dtype=np.float64)
        return self.env.eef_position()

    def _place_grasp_z(self) -> float:
        if self.env.case.target_z is None:
            return float(self.config.place_z)
        grasp_offset = np.asarray(self.env.case.grasp_offset_xyz, dtype=np.float64)
        return float(self.env.case.target_z + grasp_offset[2])

    def _phase_gripper(self) -> float:
        if self.phase in {
            PlannerPhase.GRASP,
            PlannerPhase.LIFT,
            PlannerPhase.MOVE_TO_TARGET,
            PlannerPhase.LOWER,
        }:
            return 1.0
        return -1.0

    def _advance_phase_if_ready(self, eef: np.ndarray, target: np.ndarray) -> None:
        if self.phase == PlannerPhase.DONE:
            return

        xy_err = float(np.linalg.norm(eef[:2] - target[:2]))
        z_err = float(abs(eef[2] - target[2]))
        xy_tolerance = self.config.xy_tolerance
        if self.phase in {PlannerPhase.MOVE_TO_TARGET, PlannerPhase.LOWER, PlannerPhase.RELEASE}:
            xy_tolerance = max(self.config.target_xy_tolerance, float(self.env.case.target_radius) * 0.9)
        if self.phase == PlannerPhase.MOVE_TO_TARGET:
            at_target = xy_err <= xy_tolerance
        else:
            at_target = xy_err <= xy_tolerance and z_err <= self.config.z_tolerance

        next_phase = None
        if self.phase == PlannerPhase.APPROACH and at_target:
            next_phase = PlannerPhase.DESCEND
        elif self.phase == PlannerPhase.DESCEND and at_target:
            next_phase = PlannerPhase.GRASP
        elif self.phase == PlannerPhase.GRASP and self.phase_steps >= self.config.grasp_hold_steps:
            next_phase = PlannerPhase.LIFT
        elif self.phase == PlannerPhase.LIFT and at_target:
            next_phase = PlannerPhase.MOVE_TO_TARGET
        elif self.phase == PlannerPhase.MOVE_TO_TARGET and at_target:
            next_phase = PlannerPhase.LOWER
        elif self.phase == PlannerPhase.LOWER and at_target:
            next_phase = PlannerPhase.RELEASE
        elif self.phase == PlannerPhase.RELEASE and self.phase_steps >= self.config.release_hold_steps:
            next_phase = PlannerPhase.RETREAT
        elif self.phase == PlannerPhase.RETREAT and at_target:
            next_phase = PlannerPhase.DONE

        if next_phase is not None:
            self.phase = next_phase
            self.phase_steps = 0

    def _cartesian_action(self, eef: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        delta = target - eef
        pos_action = np.clip(
            self.config.position_gain * delta,
            -self.config.max_pos_action,
            self.config.max_pos_action,
        )
        action = np.zeros(7, dtype=np.float64)
        action[:3] = pos_action
        action[3:6] = 0.0
        action[-1] = float(gripper)
        return action

    def is_done(self) -> bool:
        return self.phase == PlannerPhase.DONE

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def ensure_libero_config(repo_root: str | Path) -> Path:
    """Create a repo-local LIBERO config before importing libero.libero."""
    repo_root = Path(repo_root).resolve()
    libero_root = repo_root.parent / "LIBERO" / "libero" / "libero"
    config_dir = repo_root / ".libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        payload = {
            "benchmark_root": str(libero_root),
            "bddl_files": str(libero_root / "bddl_files"),
            "init_states": str(libero_root / "init_files"),
            "datasets": str(libero_root.parent / "datasets"),
            "assets": str(libero_root / "assets"),
        }
        config_file.write_text(
            "\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n",
            encoding="utf-8",
        )
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    return config_file


@dataclass(frozen=True)
class LiberoPushBoxCase:
    case_id: str
    friction_mu: float
    domain: str = "default"
    friction_group: str = "default"
    geometry_id: str = ""
    init_xy: tuple[float, float] | None = None
    target_distance: float | None = None
    calibration: dict[str, Any] | None = None
    bddl_file: str = "generated_bddl/PUSH_BOX_FRICTION_push_the_cream_cheese_box_to_the_green_target.bddl"
    box_name: str = "cream_cheese_1"
    target_xy: tuple[float, float] = (0.03, 0.0)
    target_radius: float = 0.025
    max_steps: int = 320
    camera_resolution: int = 256
    control_freq: float = 20.0
    geom_friction_spin: float = 0.004
    geom_friction_roll: float = 0.0001
    joint_damping: float = 0.0005
    velocity_stop_threshold: float = 0.012
    pusher_approach_steps: int = 25
    pusher_descend_steps: int = 35
    pusher_push_steps: int = 22
    pusher_retreat_steps: int = 60
    pusher_settle_steps: int = 120
    pusher_approach_offset_xy: tuple[float, float] = (-0.12, 0.0)
    pusher_contact_offset_xy: tuple[float, float] = (-0.105, 0.0)
    pusher_push_distance_x: float = 0.10
    pusher_push_angle_deg: float = 0.0
    pusher_push_profile: str = "smootherstep"
    pusher_push_mode: str = "impulse"
    pusher_push_yz_hold_gain: float = 2.0
    pusher_push_yz_max_action: float = 0.25
    pusher_approach_z: float = 1.02
    pusher_contact_z: float = 0.915
    pusher_retreat_z: float = 1.04
    pusher_position_gain: float = 10.0
    pusher_prepare_position_gain: float = 4.0
    pusher_retreat_position_gain: float = 4.0
    pusher_max_pos_action: float = 1.5
    pusher_prepare_max_pos_action: float = 0.45
    pusher_retreat_max_pos_action: float = 0.75
    pusher_prepare_action_delta: float = 0.08
    pusher_push_action_delta: float = 0.18
    pusher_retreat_action_delta: float = 0.12
    pusher_settle_action_delta: float = 0.08
    pusher_compensate_non_push_controller_scale: bool = True
    pusher_non_push_reference_controller_scale: float = 8.0
    enable_controller_output_scaling: bool = False
    max_controller_output_scale: float = 4.0
    pusher_push_controller_scale: float = 1.0
    pusher_max_push_controller_scale: float = 20.0
    pusher_push_controller_scale_ramp_steps: int = 6
    pusher_push_action_start: float = 0.2
    pusher_push_action_end: float = 1.5
    pusher_push_accel_steps: int = 6
    pusher_gripper: float = 1.0
    controller_output_scale: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiberoPushBoxCase":
        payload = dict(data)
        for key in ("init_xy", "target_xy", "pusher_approach_offset_xy", "pusher_contact_offset_xy"):
            if payload.get(key) is not None:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "friction_mu": float(self.friction_mu),
            "domain": self.domain,
            "friction_group": self.friction_group,
            "geometry_id": self.geometry_id,
            "init_xy": list(self.init_xy) if self.init_xy is not None else None,
            "target_distance": float(self.target_distance) if self.target_distance is not None else None,
            "calibration": self.calibration,
            "bddl_file": self.bddl_file,
            "box_name": self.box_name,
            "target_xy": list(self.target_xy),
            "target_radius": float(self.target_radius),
            "max_steps": int(self.max_steps),
            "camera_resolution": int(self.camera_resolution),
            "control_freq": float(self.control_freq),
            "geom_friction_spin": float(self.geom_friction_spin),
            "geom_friction_roll": float(self.geom_friction_roll),
            "joint_damping": float(self.joint_damping),
            "velocity_stop_threshold": float(self.velocity_stop_threshold),
            "pusher_approach_steps": int(self.pusher_approach_steps),
            "pusher_descend_steps": int(self.pusher_descend_steps),
            "pusher_push_steps": int(self.pusher_push_steps),
            "pusher_retreat_steps": int(self.pusher_retreat_steps),
            "pusher_settle_steps": int(self.pusher_settle_steps),
            "pusher_approach_offset_xy": list(self.pusher_approach_offset_xy),
            "pusher_contact_offset_xy": list(self.pusher_contact_offset_xy),
            "pusher_push_distance_x": float(self.pusher_push_distance_x),
            "pusher_push_angle_deg": float(self.pusher_push_angle_deg),
            "pusher_push_profile": self.pusher_push_profile,
            "pusher_push_mode": self.pusher_push_mode,
            "pusher_push_yz_hold_gain": float(self.pusher_push_yz_hold_gain),
            "pusher_push_yz_max_action": float(self.pusher_push_yz_max_action),
            "pusher_approach_z": float(self.pusher_approach_z),
            "pusher_contact_z": float(self.pusher_contact_z),
            "pusher_retreat_z": float(self.pusher_retreat_z),
            "pusher_position_gain": float(self.pusher_position_gain),
            "pusher_prepare_position_gain": float(self.pusher_prepare_position_gain),
            "pusher_retreat_position_gain": float(self.pusher_retreat_position_gain),
            "pusher_max_pos_action": float(self.pusher_max_pos_action),
            "pusher_prepare_max_pos_action": float(self.pusher_prepare_max_pos_action),
            "pusher_retreat_max_pos_action": float(self.pusher_retreat_max_pos_action),
            "pusher_prepare_action_delta": float(self.pusher_prepare_action_delta),
            "pusher_push_action_delta": float(self.pusher_push_action_delta),
            "pusher_retreat_action_delta": float(self.pusher_retreat_action_delta),
            "pusher_settle_action_delta": float(self.pusher_settle_action_delta),
            "pusher_compensate_non_push_controller_scale": bool(self.pusher_compensate_non_push_controller_scale),
            "pusher_non_push_reference_controller_scale": float(self.pusher_non_push_reference_controller_scale),
            "enable_controller_output_scaling": bool(self.enable_controller_output_scaling),
            "max_controller_output_scale": float(self.max_controller_output_scale),
            "pusher_push_controller_scale": float(self.pusher_push_controller_scale),
            "pusher_max_push_controller_scale": float(self.pusher_max_push_controller_scale),
            "pusher_push_controller_scale_ramp_steps": int(self.pusher_push_controller_scale_ramp_steps),
            "pusher_push_action_start": float(self.pusher_push_action_start),
            "pusher_push_action_end": float(self.pusher_push_action_end),
            "pusher_push_accel_steps": int(self.pusher_push_accel_steps),
            "pusher_gripper": float(self.pusher_gripper),
            "controller_output_scale": float(self.controller_output_scale),
        }

    def with_friction(self, friction_mu: float, case_id: str | None = None) -> "LiberoPushBoxCase":
        return replace(
            self,
            friction_mu=float(friction_mu),
            case_id=case_id or f"{self.case_id}_mu{float(friction_mu):.3f}",
        )


@dataclass
class LiberoPushBoxStep:
    step: int
    box_xyz: np.ndarray
    box_vxy: np.ndarray
    distance_to_target: float
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "box_xyz": self.box_xyz.astype(float).tolist(),
            "box_vxy": self.box_vxy.astype(float).tolist(),
            "distance_to_target": float(self.distance_to_target),
            "success": bool(self.success),
        }


class LiberoPushBoxEnv:
    def __init__(self, case: LiberoPushBoxCase, *, repo_root: str | Path, seed: int | None = None):
        ensure_libero_config(repo_root)
        from libero.libero.envs import OffScreenRenderEnv
        import robosuite as suite

        self.case = case
        self.repo_root = Path(repo_root).resolve()
        self.seed = seed
        bddl_file = Path(case.bddl_file)
        if not bddl_file.is_absolute():
            bddl_file = self.repo_root / bddl_file
        if not bddl_file.exists():
            raise FileNotFoundError(f"Push-box BDDL not found: {bddl_file}")
        original_load_controller_config = suite.load_controller_config

        def load_scaled_controller_config(default_controller: str) -> dict[str, Any]:
            config = original_load_controller_config(default_controller=default_controller)
            scale = self._effective_controller_output_scale()
            if scale != 1.0 and "output_max" in config and "output_min" in config:
                output_max = np.asarray(config["output_max"], dtype=np.float64)
                output_min = np.asarray(config["output_min"], dtype=np.float64)
                output_max[:3] *= scale
                output_min[:3] *= scale
                config["output_max"] = output_max.tolist()
                config["output_min"] = output_min.tolist()
            return config

        suite.load_controller_config = load_scaled_controller_config
        try:
            self.env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                camera_heights=int(case.camera_resolution),
                camera_widths=int(case.camera_resolution),
                control_freq=float(case.control_freq),
            )
        finally:
            suite.load_controller_config = original_load_controller_config
        if seed is not None:
            self.env.seed(int(seed))
        self.step_count = 0
        self._last_obs: dict[str, Any] | None = None
        self._initial_box_xyz: np.ndarray | None = None
        self._last_scripted_action = np.zeros(7, dtype=np.float64)
        self._last_scripted_phase: str | None = None
        self._base_controller_output_min: np.ndarray | None = None
        self._base_controller_output_max: np.ndarray | None = None
        self._active_controller_output_scale: float | None = None

    @property
    def inner_env(self) -> Any:
        return self.env.env

    def reset(self) -> dict[str, Any]:
        self.step_count = 0
        self._last_scripted_action = np.zeros(7, dtype=np.float64)
        self._last_scripted_phase = None
        self._base_controller_output_min = None
        self._base_controller_output_max = None
        self._active_controller_output_scale = None
        obs = self.env.reset()
        self._apply_phase_controller_scale("approach")
        self._set_box_contact_dynamics()
        self._zero_box_velocity()
        self.inner_env.sim.forward()
        self._initial_box_xyz, _ = self.box_pose()
        self._last_obs = self._refresh_obs()
        return self._last_obs

    def close(self) -> None:
        self.env.close()

    def step(self, action: np.ndarray | None = None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        phase = self._scripted_phase()
        self._apply_phase_controller_scale(phase)
        if action is None:
            action = self._scripted_gripper_push_action(phase)
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float64))
        self.step_count += 1
        obs = self._refresh_obs()
        self._last_obs = obs
        step_info = self.step_info()
        info = dict(info)
        info["push_box"] = step_info.as_dict()
        info["push_box"]["action"] = np.asarray(action, dtype=np.float64).tolist()
        info["push_box"]["phase"] = phase
        info["push_box"]["eef_xyz"] = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist()
        info["push_box"]["pusher_target_xyz"] = self._scripted_target(phase).astype(float).tolist()
        return obs, float(reward), bool(done or step_info.success), info

    def rollout(self) -> tuple[list[dict[str, Any]], list[LiberoPushBoxStep]]:
        obs = self.reset()
        observations = [obs]
        steps = [self.step_info()]
        for _ in range(int(self.case.max_steps)):
            obs, _, _, _ = self.step()
            observations.append(obs)
            steps.append(self.step_info())
        return observations, steps

    def box_pose(self) -> tuple[np.ndarray, np.ndarray]:
        obj = self.inner_env.get_object(self.case.box_name)
        qpos = np.asarray(self.inner_env.sim.data.get_joint_qpos(obj.joints[-1]), dtype=np.float64)
        if qpos.shape[0] < 7:
            raise ValueError(f"{self.case.box_name} does not expose free-joint qpos: {qpos}")
        return qpos[:3].copy(), qpos[3:7].copy()

    def box_velocity(self) -> np.ndarray:
        obj = self.inner_env.get_object(self.case.box_name)
        qvel = np.asarray(self.inner_env.sim.data.get_joint_qvel(obj.joints[-1]), dtype=np.float64)
        if qvel.shape[0] < 6:
            return np.zeros(6, dtype=np.float64)
        return qvel.copy()

    def step_info(self) -> LiberoPushBoxStep:
        xyz, _ = self.box_pose()
        qvel = self.box_velocity()
        target_xy = np.asarray(self.case.target_xy, dtype=np.float64)
        distance = float(np.linalg.norm(xyz[:2] - target_xy))
        speed = float(np.linalg.norm(qvel[:2]))
        return LiberoPushBoxStep(
            step=int(self.step_count),
            box_xyz=xyz,
            box_vxy=qvel[:2].copy(),
            distance_to_target=distance,
            success=bool(distance <= float(self.case.target_radius) and speed <= float(self.case.velocity_stop_threshold)),
        )

    def _set_box_contact_dynamics(self) -> None:
        obj = self.inner_env.get_object(self.case.box_name)
        model = self.inner_env.sim.model
        geom_names = self._geom_names()
        box_geom_names = [name for name in geom_names if name and name.startswith(f"{obj.name}_")]
        if not box_geom_names:
            box_geom_names = [name for name in geom_names if name and self.case.box_name in name]
        table_geom_names = [name for name in geom_names if name in {"table_collision", "main_table_collision"}]
        target_geom_names = box_geom_names + table_geom_names
        for geom_name in target_geom_names:
            geom_id = model.geom_name2id(geom_name)
            model.geom_friction[geom_id] = np.asarray(
                [
                    float(self.case.friction_mu),
                    float(self.case.geom_friction_spin),
                    float(self.case.geom_friction_roll),
                ],
                dtype=np.float64,
            )
        if getattr(obj, "joints", None):
            for joint_name in obj.joints:
                joint_id = model.joint_name2id(joint_name)
                model.dof_damping[model.jnt_dofadr[joint_id] : model.jnt_dofadr[joint_id] + 6] = float(
                    self.case.joint_damping
                )

    def _geom_names(self) -> list[str]:
        model = self.inner_env.sim.model
        if hasattr(model, "geom_names"):
            return [str(name) for name in model.geom_names]
        return [model.geom_id2name(i) or "" for i in range(model.ngeom)]

    def _zero_box_velocity(self) -> None:
        obj = self.inner_env.get_object(self.case.box_name)
        self.inner_env.sim.data.set_joint_qvel(obj.joints[-1], np.zeros(6, dtype=np.float64))

    def _scripted_gripper_push_action(self, phase: str | None = None) -> np.ndarray:
        if phase is None:
            phase = self._scripted_phase()
        if phase == "push":
            obs = self._last_obs
            if obs is None:
                obs = self._refresh_obs()
            eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            target = self._scripted_target(phase)
            controller_scale = self._phase_controller_output_scale(phase)
            if str(self.case.pusher_push_mode).lower() == "impulse":
                action = self._impulse_push_action(eef, target)
            else:
                action = self._cartesian_action(
                    eef,
                    target,
                    float(self.case.pusher_gripper),
                    max_action=self._push_action_x(),
                    position_gain=float(self.case.pusher_position_gain) / max(1.0, controller_scale),
                )
            action = self._limit_scripted_action_delta(
                action,
                max_delta=float(self.case.pusher_push_action_delta),
            )
            self._last_scripted_phase = phase
            return action
        if phase == "settle":
            action = np.zeros(7, dtype=np.float64)
            action[-1] = float(self.case.pusher_gripper)
            action = self._limit_scripted_action_delta(
                action,
                max_delta=float(self.case.pusher_settle_action_delta),
            )
            self._last_scripted_phase = phase
            return action

        obs = self._last_obs
        if obs is None:
            obs = self._refresh_obs()
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        target = self._scripted_target(phase)
        if phase in {"approach", "descend"}:
            max_action = float(self.case.pusher_prepare_max_pos_action)
            position_gain = float(self.case.pusher_prepare_position_gain)
            max_delta = float(self.case.pusher_prepare_action_delta)
        else:
            max_action = float(self.case.pusher_retreat_max_pos_action)
            position_gain = float(self.case.pusher_retreat_position_gain)
            max_delta = float(self.case.pusher_retreat_action_delta)
        action = self._cartesian_action(
            eef,
            target,
            float(self.case.pusher_gripper),
            max_action=max_action,
            position_gain=position_gain,
        )
        action = self._compensate_non_push_controller_scale(action)
        action = self._limit_scripted_action_delta(action, max_delta=max_delta)
        self._last_scripted_phase = phase
        return action

    def _push_action_x(self) -> float:
        push_start = int(self.case.pusher_approach_steps) + int(self.case.pusher_descend_steps)
        push_idx = max(0, int(self.step_count) - push_start)
        accel_steps = max(1, int(self.case.pusher_push_accel_steps))
        alpha = min(1.0, float(push_idx + 1) / float(accel_steps))
        action = (1.0 - alpha) * float(self.case.pusher_push_action_start) + alpha * float(
            self.case.pusher_push_action_end
        )
        max_action = float(self.case.pusher_max_pos_action)
        return float(np.clip(action, -max_action, max_action))

    def _push_envelope(self) -> float:
        push_start = int(self.case.pusher_approach_steps) + int(self.case.pusher_descend_steps)
        push_idx = max(0, int(self.step_count) - push_start)
        push_steps = max(1, int(self.case.pusher_push_steps))
        progress = min(1.0, float(push_idx + 1) / float(push_steps))
        return max(0.0, float(np.sin(np.pi * progress)))

    def _impulse_push_action(self, eef: np.ndarray, target: np.ndarray) -> np.ndarray:
        action = np.zeros(7, dtype=np.float64)
        # One directional velocity pulse: no x-axis feedback, so contact noise cannot
        # make the pusher oscillate backward and forward.
        action[0] = float(self.case.pusher_push_action_end) * self._push_envelope()
        yz_delta = target[1:3] - eef[1:3]
        yz = float(self.case.pusher_push_yz_hold_gain) * yz_delta
        yz_limit = float(self.case.pusher_push_yz_max_action)
        action[1:3] = np.clip(yz, -yz_limit, yz_limit)
        action[:3] = np.clip(action[:3], -float(self.case.pusher_max_pos_action), float(self.case.pusher_max_pos_action))
        action[-1] = float(self.case.pusher_gripper)
        return action

    def _scripted_phase(self) -> str:
        descend_start = int(self.case.pusher_approach_steps)
        push_start = descend_start + int(self.case.pusher_descend_steps)
        retreat_start = push_start + int(self.case.pusher_push_steps)
        settle_start = retreat_start + int(self.case.pusher_retreat_steps)
        if self.step_count < descend_start:
            return "approach"
        if self.step_count < push_start:
            return "descend"
        if self.step_count < retreat_start:
            return "push"
        if self.step_count < settle_start:
            return "retreat"
        return "settle"

    def _scripted_target(self, phase: str) -> np.ndarray:
        if self._initial_box_xyz is None:
            self._initial_box_xyz, _ = self.box_pose()
        box = self._initial_box_xyz
        if phase == "approach":
            offset = self._oriented_xy_offset(self.case.pusher_approach_offset_xy)
            z = float(self.case.pusher_approach_z)
        elif phase == "descend":
            offset = self._oriented_xy_offset(self.case.pusher_contact_offset_xy)
            z = float(self.case.pusher_contact_z)
        elif phase == "push":
            push_start = int(self.case.pusher_approach_steps) + int(self.case.pusher_descend_steps)
            push_idx = max(0, int(self.step_count) - push_start)
            progress = min(1.0, float(push_idx + 1) / float(max(1, int(self.case.pusher_push_steps))))
            progress = self._push_progress(progress)
            offset = self._oriented_xy_offset(self.case.pusher_contact_offset_xy)
            offset = offset + self._push_direction_xy() * progress * float(self.case.pusher_push_distance_x)
            z = float(self.case.pusher_contact_z)
        else:
            offset = self._oriented_xy_offset(self.case.pusher_approach_offset_xy)
            z = float(self.case.pusher_retreat_z)
        return np.asarray([box[0] + offset[0], box[1] + offset[1], z], dtype=np.float64)

    def _push_direction_xy(self) -> np.ndarray:
        angle = np.deg2rad(float(self.case.pusher_push_angle_deg))
        return np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)

    def _oriented_xy_offset(self, offset_xy: tuple[float, float]) -> np.ndarray:
        local = np.asarray(offset_xy, dtype=np.float64)
        direction = self._push_direction_xy()
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        return direction * local[0] + lateral * local[1]

    def _cartesian_action(
        self,
        eef: np.ndarray,
        target: np.ndarray,
        gripper: float,
        *,
        max_action: float | None = None,
        position_gain: float | None = None,
    ) -> np.ndarray:
        delta = target - eef
        action_limit = float(self.case.pusher_max_pos_action if max_action is None else max_action)
        gain = float(self.case.pusher_position_gain if position_gain is None else position_gain)
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(
            gain * delta,
            -action_limit,
            action_limit,
        )
        action[-1] = gripper
        return action

    def _push_progress(self, progress: float) -> float:
        x = float(np.clip(progress, 0.0, 1.0))
        profile = str(self.case.pusher_push_profile).lower()
        if profile == "smoothstep":
            return x * x * (3.0 - 2.0 * x)
        if profile == "smootherstep":
            return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
        return x

    def _limit_scripted_action_delta(self, action: np.ndarray, *, max_delta: float) -> np.ndarray:
        smoothed = np.asarray(action, dtype=np.float64).copy()
        delta = smoothed[:3] - self._last_scripted_action[:3]
        smoothed[:3] = self._last_scripted_action[:3] + np.clip(delta, -float(max_delta), float(max_delta))
        smoothed[-1] = action[-1]
        self._last_scripted_action = smoothed.copy()
        return smoothed

    def _compensate_non_push_controller_scale(self, action: np.ndarray) -> np.ndarray:
        if not bool(self.case.pusher_compensate_non_push_controller_scale):
            return action
        scale = self._effective_controller_output_scale()
        reference_scale = max(1.0, float(self.case.pusher_non_push_reference_controller_scale))
        if scale <= reference_scale:
            return action
        compensated = np.asarray(action, dtype=np.float64).copy()
        compensated[:3] *= reference_scale / scale
        return compensated

    def _effective_controller_output_scale(self) -> float:
        if not bool(self.case.enable_controller_output_scaling):
            return 1.0
        return float(min(float(self.case.controller_output_scale), float(self.case.max_controller_output_scale)))

    def _phase_controller_output_scale(self, phase: str) -> float:
        if phase != "push":
            return 1.0
        push_start = int(self.case.pusher_approach_steps) + int(self.case.pusher_descend_steps)
        push_idx = max(0, int(self.step_count) - push_start)
        ramp_steps = max(1, int(self.case.pusher_push_controller_scale_ramp_steps))
        alpha = min(1.0, float(push_idx + 1) / float(ramp_steps))
        target = min(
            float(self.case.pusher_push_controller_scale),
            float(self.case.pusher_max_push_controller_scale),
        )
        return float((1.0 - alpha) + alpha * max(1.0, target))

    def _controller(self) -> Any | None:
        robots = getattr(self.inner_env, "robots", None)
        if not robots:
            return None
        return getattr(robots[0], "controller", None)

    def _apply_phase_controller_scale(self, phase: str) -> None:
        controller = self._controller()
        if controller is None or not hasattr(controller, "output_max") or not hasattr(controller, "output_min"):
            return
        if self._base_controller_output_min is None or self._base_controller_output_max is None:
            self._base_controller_output_min = np.asarray(controller.output_min, dtype=np.float64).copy()
            self._base_controller_output_max = np.asarray(controller.output_max, dtype=np.float64).copy()

        scale = self._phase_controller_output_scale(phase)
        if self._active_controller_output_scale is not None and np.isclose(self._active_controller_output_scale, scale):
            return

        output_min = self._base_controller_output_min.copy()
        output_max = self._base_controller_output_max.copy()
        dims = min(3, output_min.shape[0], output_max.shape[0])
        output_min[:dims] *= scale
        output_max[:dims] *= scale
        controller.output_min = output_min
        controller.output_max = output_max
        controller.action_scale = None
        controller.action_output_transform = None
        controller.action_input_transform = None
        self._active_controller_output_scale = scale

    def _refresh_obs(self) -> dict[str, Any]:
        self.inner_env.sim.forward()
        self.inner_env._post_process()
        self.inner_env._update_observables(force=True)
        return self.inner_env._get_observations()


def load_libero_push_box_cases(path: str | Path) -> list[LiberoPushBoxCase]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    return [LiberoPushBoxCase.from_dict(item) for item in items]


def dump_libero_push_box_cases(path: str | Path, cases: list[LiberoPushBoxCase]) -> None:
    path = Path(path)
    path.write_text(json.dumps({"cases": [case.as_dict() for case in cases]}, indent=2), encoding="utf-8")

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path("/home/yininghong/chenyuan/TTT-physics/repos/TTT4dynamics")
LIBERO_ROOT = REPO_ROOT.parent / "LIBERO"
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM"
LIBERO_CONFIG_PATH = LIBERO_ROOT / ".libero_config"
PIECEWISE_SCRIPT = REPO_ROOT / "tmp" / "piecewise_carrier_trajectory_demo_2026-07-05_hai-machine.py"
CASE_CONFIG = REPO_ROOT / "configs" / "dynamic_carrier_cases.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "lerobot"
    / "dynamic_carrier_piecewise_observe160_speed012_stable_release_100eps_2026-07-06_hai-machine"
)

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG_PATH))
for path in (REPO_ROOT, LIBERO_ROOT, FASTWAM_ROOT / "src", FASTWAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

from ttt4dynamics.cases import load_cases
from ttt4dynamics.dynamic_env import DynamicCarrierEnv, create_libero_env_for_case
from ttt4dynamics.planner import PlannerConfig, PlannerPhase, ScriptedDynamicCarrierPlanner
from ttt4dynamics.trajectories import TrajectoryState


FPS = 20
CAMERA_RESOLUTION = 224
OBSERVE_FRAMES = 160
EXECUTION_STEPS = 500
DEMO_SPEED = 0.12
DIRECT_INTERCEPT_LEAD_S = 0.75
DIRECT_GRASP_MAX_FRAMES = 24
DIRECT_INTERCEPT_EEF_SPEED_MPS = 0.24
DIRECT_INTERCEPT_SETTLE_S = 0.15
DIRECT_INTERCEPT_MIN_S = 0.30
DIRECT_INTERCEPT_MAX_S = 1.80
DIRECT_GRASP_Z_BIAS_M = -0.015
TARGET_RADIUS = 0.065
SAFE_RELEASE_PAYLOAD_Z = 0.935
POST_RELEASE_SETTLE_FRAMES = 16
TEMPLATE_GROUPS = [
    ("polygon_2", ["line_bounce"]),
    ("arc_polygon_2", ["arc_bounce_60deg", "line_plus_arc_90deg", "line_plus_arc_120deg"]),
    ("polygon_3", ["right_triangle", "obtuse_triangle", "acute_triangle"]),
    ("arc_polygon_3", ["triangle_with_arc_side"]),
    ("polygon_4", ["right_angle_quad", "trapezoid_quad", "kite_quad"]),
    ("arc_polygon_4", ["quad_with_arc_side"]),
    ("circle", ["circle_small_left", "circle_medium_right"]),
]
TEMPLATE_NAMES = [template_name for _, template_names in TEMPLATE_GROUPS for template_name in template_names]
TASK_PROMPT = [
    "dynamic carrier piecewise observe then pick place",
    "observe the moving tray trajectory, intercept the cream cheese box, and place it on the static target",
    "successful direct-intercept scripted demonstration",
    "success",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-id", default="ttt_dynamic_carrier_piecewise_observe160")
    parser.add_argument("--video-codec", default="h264")
    return parser.parse_args()


def load_piecewise_module() -> Any:
    spec = importlib.util.spec_from_file_location("piecewise_carrier_demo", PIECEWISE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.NOMINAL_SPEED = float(DEMO_SPEED)
    return module


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_") and not callable(item)
        }
    return repr(value)


def _trajectory_snapshot(template: Any, *, template_group: str, time_offset_s: float, xy_offset: np.ndarray) -> dict[str, Any]:
    return {
        "template_group": str(template_group),
        "template_name": str(template.name),
        "template_family": str(template.family),
        "template_description": str(template.description),
        "template_period_s": float(template.period),
        "template_raw": _jsonable(template),
        "source_script": str(PIECEWISE_SCRIPT),
        "speed_mps": float(DEMO_SPEED),
        "time_offset_s": float(time_offset_s),
        "xy_offset": np.asarray(xy_offset, dtype=np.float64).astype(float).tolist(),
        "start_phase": float((float(time_offset_s) / max(float(template.period), 1e-9)) % 1.0),
    }


def _safe_object_pose(env: DynamicCarrierEnv, object_name: str | None) -> dict[str, Any] | None:
    if not object_name:
        return None
    try:
        pos, quat = env.get_object_pose(object_name)
        return {
            "name": str(object_name),
            "position": np.asarray(pos, dtype=np.float64).astype(float).tolist(),
            "quat_wxyz": np.asarray(quat, dtype=np.float64).astype(float).tolist(),
        }
    except Exception as exc:
        return {"name": str(object_name), "error": repr(exc)}


def _target_virtual_pose(env: DynamicCarrierEnv) -> dict[str, Any]:
    target_z = env.case.target_z if env.case.target_z is not None else SAFE_RELEASE_PAYLOAD_Z
    return {
        "name": "virtual_target",
        "position": [float(env.case.target_xy[0]), float(env.case.target_xy[1]), float(target_z)],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "radius": float(env.case.target_radius),
    }


def _pose_to_vector(pose: dict[str, Any] | None) -> np.ndarray:
    if pose is None or "position" not in pose or "quat_wxyz" not in pose:
        return np.full(7, np.nan, dtype=np.float32)
    return np.asarray(pose["position"] + pose["quat_wxyz"], dtype=np.float32)


def _scene_object_poses(env: DynamicCarrierEnv) -> dict[str, Any]:
    return {
        "carrier": _safe_object_pose(env, getattr(env.case, "carrier_name", None)),
        "payload": _safe_object_pose(env, getattr(env.case, "payload_name", None)),
        "target": _target_virtual_pose(env),
    }


def _scene_pose_vector(env: DynamicCarrierEnv) -> np.ndarray:
    poses = _scene_object_poses(env)
    return np.concatenate(
        [
            _pose_to_vector(poses["carrier"]),
            _pose_to_vector(poses["payload"]),
            _pose_to_vector(poses["target"]),
        ],
        axis=0,
    ).astype(np.float32)


def _initial_conditions(
    env: DynamicCarrierEnv,
    obs: dict[str, Any],
    *,
    init_state: np.ndarray | None,
    reset_seed: int,
) -> dict[str, Any]:
    trajectory_state = env.trajectory.sample(env.t)
    phase = env.trajectory.phase(env.t) if hasattr(env.trajectory, "phase") else 0.0
    sim_init_state = np.asarray(env.get_sim_state(), dtype=np.float64)
    return {
        "reset_seed": int(reset_seed),
        "sim_init_state": sim_init_state.astype(float).tolist(),
        "sim_init_state_available": True,
        "source_init_state_from_create_env": (
            np.asarray(init_state, dtype=np.float64).astype(float).tolist() if init_state is not None else None
        ),
        "env_t": float(env.t),
        "trajectory_phase": float(phase),
        "trajectory_xy": trajectory_state.xy.astype(float).tolist(),
        "trajectory_velocity_xy": trajectory_state.velocity_xy.astype(float).tolist(),
        "trajectory_yaw": float(trajectory_state.yaw),
        "object_poses": _scene_object_poses(env),
        "eef_xyz": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).astype(float).tolist(),
        "eef_quat_wxyz": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).astype(float).tolist(),
    }


def build_quota_plan(episodes: int) -> list[dict[str, Any]]:
    group_count = len(TEMPLATE_GROUPS)
    group_base = int(episodes) // group_count
    group_remainder = int(episodes) % group_count
    plan = []
    for group_index, (group_name, template_names) in enumerate(TEMPLATE_GROUPS):
        group_total = group_base + int(group_index < group_remainder)
        template_base = group_total // len(template_names)
        template_remainder = group_total % len(template_names)
        for template_index, template_name in enumerate(template_names):
            target_episodes = template_base + int(template_index < template_remainder)
            if target_episodes > 0:
                plan.append(
                    {
                        "group": group_name,
                        "template_name": template_name,
                        "target_episodes": int(target_episodes),
                    }
                )
    return plan


class OffsetTemplateTrajectory:
    def __init__(self, template: Any, *, time_offset_s: float, xy_offset: np.ndarray):
        self.template = template
        self.time_offset_s = float(time_offset_s)
        self.xy_offset = np.asarray(xy_offset, dtype=np.float64)
        self.period = float(template.period)

    def sample(self, t: float) -> TrajectoryState:
        xy, velocity_xy = self.template.sample(float(t) + self.time_offset_s)
        return TrajectoryState(
            xy=np.asarray(xy, dtype=np.float64) + self.xy_offset,
            velocity_xy=np.asarray(velocity_xy, dtype=np.float64),
            yaw=0.0,
        )

    def phase(self, t: float) -> float:
        return float(((float(t) + self.time_offset_s) / max(self.period, 1e-9)) % 1.0)


class StableReleaseDynamicCarrierEnv(DynamicCarrierEnv):
    def __init__(self, base_env: Any, case: Any):
        super().__init__(base_env, case)
        self._stable_release_pose: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self, init_state: np.ndarray | None = None) -> dict[str, Any]:
        self._stable_release_pose = None
        return super().reset(init_state=init_state)

    def _zero_payload_velocity(self) -> None:
        obj = self.inner_env.get_object(self.case.payload_name)
        if not getattr(obj, "joints", None):
            return
        try:
            qvel = np.asarray(self.inner_env.sim.data.get_joint_qvel(obj.joints[-1]), dtype=np.float64)
            self.inner_env.sim.data.set_joint_qvel(obj.joints[-1], np.zeros_like(qvel))
        except Exception:
            return

    def _update_payload_attachment(self, action: np.ndarray, obs: dict[str, Any]) -> None:
        was_attached = bool(self.payload_attached_to_gripper)
        super()._update_payload_attachment(action, obs)
        released_now = was_attached and not bool(self.payload_attached_to_gripper) and bool(self.payload_detached)
        if released_now:
            pos = self._release_payload_pos.copy() if self._release_payload_pos is not None else self.payload_position()
            pos = np.asarray(pos, dtype=np.float64)
            pos[2] = max(float(pos[2]), SAFE_RELEASE_PAYLOAD_Z)
            quat = self._payload_quat.copy() if self._payload_quat is not None else self.get_object_pose(self.case.payload_name)[1]
            self._release_payload_pos = pos.copy()
            self._stable_release_pose = (pos.copy(), quat.copy())
            self.set_object_pose(self.case.payload_name, pos, quat)
            self._zero_payload_velocity()
            self.inner_env.sim.forward()

    def _apply_kinematic_motion(self, t: float, obs: dict[str, Any] | None = None) -> None:
        super()._apply_kinematic_motion(t, obs)
        if self._stable_release_pose is not None and not bool(self.payload_attached_to_gripper):
            pos, quat = self._stable_release_pose
            self.set_object_pose(self.case.payload_name, pos, quat)
            self._zero_payload_velocity()
            self.inner_env.sim.forward()


class AttachmentAwarePlanner(ScriptedDynamicCarrierPlanner):
    def _advance_phase_if_ready(self, eef: np.ndarray, target: np.ndarray) -> None:
        if self.phase == PlannerPhase.DONE:
            return

        xy_err = float(np.linalg.norm(eef[:2] - target[:2]))
        z_err = float(abs(eef[2] - target[2]))
        xy_tolerance = self.config.xy_tolerance
        if self.phase in {PlannerPhase.MOVE_TO_TARGET, PlannerPhase.LOWER, PlannerPhase.RELEASE}:
            xy_tolerance = max(self.config.target_xy_tolerance, float(self.env.case.target_radius) * 0.9)
        at_target = xy_err <= xy_tolerance if self.phase == PlannerPhase.MOVE_TO_TARGET else xy_err <= xy_tolerance and z_err <= self.config.z_tolerance

        next_phase = None
        if self.phase == PlannerPhase.APPROACH and at_target:
            next_phase = PlannerPhase.DESCEND
        elif self.phase == PlannerPhase.DESCEND and at_target:
            next_phase = PlannerPhase.GRASP
        elif self.phase == PlannerPhase.GRASP and bool(self.env.payload_attached_to_gripper):
            next_phase = PlannerPhase.LIFT
        elif self.phase == PlannerPhase.LIFT and at_target:
            next_phase = PlannerPhase.MOVE_TO_TARGET
        elif self.phase == PlannerPhase.MOVE_TO_TARGET and at_target:
            next_phase = PlannerPhase.LOWER
        elif self.phase == PlannerPhase.LOWER and at_target:
            next_phase = PlannerPhase.RELEASE
        elif self.phase == PlannerPhase.RELEASE and self.phase_steps >= self.config.release_hold_steps:
            next_phase = PlannerPhase.DONE

        if next_phase is not None:
            self.phase = next_phase
            self.phase_steps = 0


class DirectInterceptPlanner(AttachmentAwarePlanner):
    def __init__(
        self,
        env: DynamicCarrierEnv,
        config: PlannerConfig | None = None,
        *,
        intercept_lead_s: float = DIRECT_INTERCEPT_LEAD_S,
        grasp_max_frames: int = DIRECT_GRASP_MAX_FRAMES,
        eef_speed_mps: float = DIRECT_INTERCEPT_EEF_SPEED_MPS,
        settle_s: float = DIRECT_INTERCEPT_SETTLE_S,
    ):
        super().__init__(env, config)
        self.direct_intercept_lead_s = float(intercept_lead_s)
        self.grasp_max_frames = int(grasp_max_frames)
        self.eef_speed_mps = float(eef_speed_mps)
        self.settle_s = float(settle_s)
        self._locked_grasp_xy: np.ndarray | None = None
        self._locked_intercept_tau: float | None = None

    def reset(self) -> None:
        super().reset()
        self._locked_grasp_xy = None
        self._locked_intercept_tau = None

    def _lock_grasp_xy(self) -> np.ndarray:
        if self._locked_grasp_xy is None:
            eef = self.env.eef_position()
            access = self.env.case.access_mode.lower()
            approach_z = self.config.box_approach_z if "box" in access or "tray" in access else self.config.flat_approach_z
            grasp_z = float(self.env.payload_grasp_position()[2] + DIRECT_GRASP_Z_BIAS_M)
            tau = float(self.direct_intercept_lead_s)
            for _ in range(8):
                xy = self.env.predict_payload_grasp_xy(self.env.t + tau)
                approach_target = np.asarray([xy[0], xy[1], approach_z], dtype=np.float64)
                horizontal_and_approach = float(np.linalg.norm(approach_target - eef))
                descend = float(abs(approach_z - grasp_z))
                travel_time = (horizontal_and_approach + descend) / max(self.eef_speed_mps, 1e-6)
                tau = float(np.clip(travel_time + self.settle_s, DIRECT_INTERCEPT_MIN_S, DIRECT_INTERCEPT_MAX_S))
            self._locked_intercept_tau = tau
            self._locked_grasp_xy = self.env.predict_payload_grasp_xy(self.env.t + tau)
        return self._locked_grasp_xy.copy()

    def _phase_target(self) -> np.ndarray:
        if self.phase in {PlannerPhase.APPROACH, PlannerPhase.DESCEND, PlannerPhase.GRASP}:
            xy = self._lock_grasp_xy()
            access = self.env.case.access_mode.lower()
            approach_z = self.config.box_approach_z if "box" in access or "tray" in access else self.config.flat_approach_z
            grasp_z = float(self.env.payload_grasp_position()[2] + DIRECT_GRASP_Z_BIAS_M)
            if self.phase == PlannerPhase.APPROACH:
                return np.asarray([xy[0], xy[1], approach_z], dtype=np.float64)
            return np.asarray([xy[0], xy[1], grasp_z], dtype=np.float64)
        return super()._phase_target()

    def _advance_phase_if_ready(self, eef: np.ndarray, target: np.ndarray) -> None:
        if self.phase == PlannerPhase.GRASP and not bool(self.env.payload_attached_to_gripper):
            if self.phase_steps >= self.grasp_max_frames:
                self.phase = PlannerPhase.DONE
                self.phase_steps = 0
                return
        super()._advance_phase_if_ready(eef, target)


def _build_features(camera_resolution: int) -> dict[str, dict[str, Any]]:
    image_shape = (3, int(camera_resolution), int(camera_resolution))
    return {
        "observation.images.image": {"dtype": "video", "shape": image_shape, "names": ["channel", "height", "width"]},
        "observation.images.wrist_image": {"dtype": "video", "shape": image_shape, "names": ["channel", "height", "width"]},
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["eef_x", "eef_y", "eef_z", "eef_axis_x", "eef_axis_y", "eef_axis_z", "gripper_qpos_0", "gripper_qpos_1"],
        },
        "observation.carrier_state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["carrier_x", "carrier_y", "carrier_vx", "carrier_vy"],
        },
        "observation.payload_state": {
            "dtype": "float32",
            "shape": (5,),
            "names": ["payload_x", "payload_y", "payload_z", "attached", "detached"],
        },
        "observation.object_poses": {
            "dtype": "float32",
            "shape": (21,),
            "names": [
                "carrier_x",
                "carrier_y",
                "carrier_z",
                "carrier_qw",
                "carrier_qx",
                "carrier_qy",
                "carrier_qz",
                "payload_x",
                "payload_y",
                "payload_z",
                "payload_qw",
                "payload_qx",
                "payload_qy",
                "payload_qz",
                "target_x",
                "target_y",
                "target_z",
                "target_qw",
                "target_qx",
                "target_qy",
                "target_qz",
            ],
        },
        "observation.dynamic_time": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["t", "phase_sin", "phase_cos", "is_execute"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"],
        },
    }


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (xyz / sin_half * (2.0 * math.atan2(sin_half, w))).astype(np.float32)


def _obs_to_images(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
    return agent, wrist


def _obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _env_action_to_dataset_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[-1] = (1.0 - out[-1]) / 2.0
    return out


def _write_image_for_last_frame(dataset: LeRobotDataset, key: str, frame_index: int, image: np.ndarray) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=frame_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=95)


def _remove_current_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    episode_index = dataset.episode_buffer["episode_index"]
    for key in dataset.meta.video_keys:
        image_dir = dataset._get_image_file_path(
            episode_index=episode_index,
            image_key=key,
            frame_index=0,
        ).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def _frame_payload(
    env: StableReleaseDynamicCarrierEnv,
    obs: dict[str, Any],
    action: np.ndarray,
    *,
    frame_count: int,
    is_execute: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    agent, wrist = _obs_to_images(obs)
    trajectory = env.trajectory
    state = trajectory.sample(env.t)
    phase = trajectory.phase(env.t) if hasattr(trajectory, "phase") else 0.0
    payload = env.payload_position()
    object_poses = _scene_object_poses(env)
    frame = {
        "observation.images.image": agent,
        "observation.images.wrist_image": wrist,
        "observation.state": _obs_to_state(obs),
        "observation.carrier_state": np.asarray([state.xy[0], state.xy[1], state.velocity_xy[0], state.velocity_xy[1]], dtype=np.float32),
        "observation.payload_state": np.asarray(
            [payload[0], payload[1], payload[2], float(env.payload_attached_to_gripper), float(env.payload_detached)],
            dtype=np.float32,
        ),
        "observation.object_poses": np.concatenate(
            [
                _pose_to_vector(object_poses["carrier"]),
                _pose_to_vector(object_poses["payload"]),
                _pose_to_vector(object_poses["target"]),
            ],
            axis=0,
        ).astype(np.float32),
        "observation.dynamic_time": np.asarray(
            [env.t, math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase), float(is_execute)],
            dtype=np.float32,
        ),
        "action": _env_action_to_dataset_action(action),
    }
    row = {
        "frame": int(frame_count),
        "segment": "execute" if is_execute else "observe",
        "t": float(env.t),
        "trajectory_phase": float(phase),
        "carrier_xy": state.xy.astype(float).tolist(),
        "carrier_velocity_xy": state.velocity_xy.astype(float).tolist(),
        "carrier_yaw": float(state.yaw),
        "payload_xyz": payload.astype(float).tolist(),
        "object_poses": object_poses,
        "eef_xyz": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).astype(float).tolist(),
        "eef_quat_wxyz": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).astype(float).tolist(),
        "action_env": np.asarray(action, dtype=np.float64).astype(float).tolist(),
        "attached": bool(env.payload_attached_to_gripper),
        "detached": bool(env.payload_detached),
    }
    return frame, row


def _save_rollout(
    dataset: LeRobotDataset,
    *,
    case: Any,
    template_group: str,
    template: Any,
    time_offset_s: float,
    xy_offset: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    base_env, init_state, task_description = create_libero_env_for_case(
        case,
        repo_root=REPO_ROOT,
        camera_resolution=CAMERA_RESOLUTION,
        seed=seed,
    )
    env = StableReleaseDynamicCarrierEnv(base_env, case)
    env.trajectory = OffsetTemplateTrajectory(template, time_offset_s=time_offset_s, xy_offset=xy_offset)
    planner = DirectInterceptPlanner(
        env,
        PlannerConfig(
            intercept_lead_s=DIRECT_INTERCEPT_LEAD_S,
            position_gain=14.0,
            max_pos_action=1.0,
            xy_tolerance=0.04,
            target_xy_tolerance=0.035,
            z_tolerance=0.035,
            place_z=0.99,
            release_hold_steps=4,
        ),
    )
    rows = []
    frame_count = 0
    attach_ever = False
    success = False
    saved_episode_index = None
    trajectory_config = _trajectory_snapshot(
        template,
        template_group=template_group,
        time_offset_s=time_offset_s,
        xy_offset=xy_offset,
    )
    initial_snapshot = None
    no_op_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    try:
        _remove_current_episode_images(dataset)
        obs = env.reset(init_state=init_state)
        initial_snapshot = _initial_conditions(env, obs, init_state=init_state, reset_seed=seed)
        planner.reset()
        for _ in range(OBSERVE_FRAMES):
            frame, row = _frame_payload(env, obs, no_op_action, frame_count=frame_count, is_execute=False)
            dataset.add_frame(frame, task=TASK_PROMPT, timestamp=frame_count / FPS)
            _write_image_for_last_frame(dataset, "observation.images.image", frame_count, frame["observation.images.image"])
            _write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_count, frame["observation.images.wrist_image"])
            rows.append(row)
            frame_count += 1
            obs, _, _, _ = env.step(no_op_action)

        action_start_frame = int(frame_count)
        planner.reset()
        phase_counts: dict[str, int] = {}
        post_release_frames = 0
        for _ in range(EXECUTION_STEPS):
            phase = str(planner.phase.value)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            action = planner.act(obs)
            frame, row = _frame_payload(env, obs, action, frame_count=frame_count, is_execute=True)
            row["phase"] = phase
            row["locked_grasp_xy"] = planner._locked_grasp_xy.astype(float).tolist() if planner._locked_grasp_xy is not None else None
            row["locked_intercept_tau"] = float(planner._locked_intercept_tau) if planner._locked_intercept_tau is not None else None
            dataset.add_frame(frame, task=TASK_PROMPT, timestamp=frame_count / FPS)
            _write_image_for_last_frame(dataset, "observation.images.image", frame_count, frame["observation.images.image"])
            _write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_count, frame["observation.images.wrist_image"])
            rows.append(row)
            frame_count += 1
            obs, _, _, _ = env.step(action)
            attach_ever = bool(attach_ever or env.payload_attached_to_gripper)
            released_and_open = bool(env.payload_detached and not env.payload_attached_to_gripper)
            post_release_frames = post_release_frames + 1 if released_and_open else 0
            success = bool(env.check_success())
            if success and post_release_frames >= POST_RELEASE_SETTLE_FRAMES:
                break
            if planner.is_done():
                success = bool(env.check_success())
                if not success:
                    break

        if success and attach_ever:
            saved_episode_index = int(dataset.meta.total_episodes)
            dataset.save_episode()
        else:
            _remove_current_episode_images(dataset)
            dataset.clear_episode_buffer()

        return {
            "success": bool(success and attach_ever),
            "episode_index": saved_episode_index,
            "steps": int(frame_count),
            "action_start_frame": int(action_start_frame),
            "attach_ever": bool(attach_ever),
            "post_release_settle_frames_required": int(POST_RELEASE_SETTLE_FRAMES),
            "post_release_frames": int(post_release_frames),
            "phase_counts": phase_counts,
            "case": case.as_dict(),
            "template_group": str(template_group),
            "template_name": template.name,
            "template_family": template.family,
            "template_description": template.description,
            "trajectory_config": trajectory_config,
            "initial_conditions": initial_snapshot,
            "time_offset_s": float(time_offset_s),
            "xy_offset": np.asarray(xy_offset, dtype=np.float64).astype(float).tolist(),
            "task_description": task_description,
            "rows": rows,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    piecewise = load_piecewise_module()
    templates = {template.name: template for template in piecewise.make_templates()}
    quota_plan = build_quota_plan(int(args.episodes))
    base_case = next(case for case in load_cases(CASE_CONFIG) if case.case_id == "box_line_medium")
    rng = np.random.default_rng(int(args.seed))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=FPS,
        features=_build_features(CAMERA_RESOLUTION),
        use_videos=True,
        video_codec=args.video_codec,
        is_compute_episode_stats_image=False,
    )

    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "dynamic_carrier_piecewise_observe160_direct_intercept_stable_release_lerobot",
        "episodes_requested": int(args.episodes),
        "seed": int(args.seed),
        "camera_resolution": CAMERA_RESOLUTION,
        "fps": FPS,
        "observe_frames": OBSERVE_FRAMES,
        "execution_steps": EXECUTION_STEPS,
        "speed_mps": DEMO_SPEED,
        "target_radius": TARGET_RADIUS,
        "safe_release_payload_z": SAFE_RELEASE_PAYLOAD_Z,
        "direct_grasp_z_bias_m": DIRECT_GRASP_Z_BIAS_M,
        "post_release_settle_frames": POST_RELEASE_SETTLE_FRAMES,
        "post_release_policy": "After custom success first becomes true, keep recording no-motion open-gripper actions until the payload has been detached for this many frames.",
        "state_recording": {
            "lerobot_frame_features": {
                "observation.carrier_state": "[carrier_x, carrier_y, carrier_vx, carrier_vy] from the commanded trajectory at env.t",
                "observation.payload_state": "[payload_x, payload_y, payload_z, attached, detached]",
                "observation.object_poses": "carrier/payload actual MuJoCo poses and virtual target pose as xyz + quaternion(wxyz), 21 floats",
                "observation.dynamic_time": "[env_t, sin(2*pi*trajectory_phase), cos(2*pi*trajectory_phase), is_execute]",
            },
            "episode_metadata": {
                "trajectory_config": "topology group, template raw parameters, source script, speed, time_offset_s, xy_offset, start_phase",
                "initial_conditions": "reset_seed, full reset-time sim_init_state from env.get_sim_state(), initial trajectory state, initial object poses, initial eef pose",
                "rows": "per-frame trajectory state, object poses, eef pose, env action, attachment flags, and planner phase where available",
            },
            "reconstruction_contract": "A test case is reconstructable from case + trajectory_config + initial_conditions.sim_init_state + rows/frame states.",
        },
        "balanced_sampling": "balanced_by_topology_group_then_template; failed rollouts retry the same quota slot",
        "template_groups": [{"group": group_name, "template_names": template_names} for group_name, template_names in TEMPLATE_GROUPS],
        "template_names": TEMPLATE_NAMES,
        "quota_plan": quota_plan,
        "quota_progress": {item["template_name"]: 0 for item in quota_plan},
        "successes": [],
        "failures": [],
    }
    metadata_path = output / "dynamic_carrier_piecewise_generation_metadata.json"

    successes = 0
    attempts = 0
    exhausted_attempts = False
    for quota_item in quota_plan:
        template_group = str(quota_item["group"])
        template_name = str(quota_item["template_name"])
        target_episodes = int(quota_item["target_episodes"])
        template = templates[template_name]
        collected_for_template = 0
        while collected_for_template < target_episodes:
            if attempts >= int(args.max_attempts):
                exhausted_attempts = True
                break
            time_offset_s = float(rng.uniform(0.0, template.period))
            xy_offset = rng.uniform(-0.012, 0.012, size=2).astype(np.float64)
            case = replace(
                base_case,
                case_id=f"piecewise_{template_group}_{template.name}_a{attempts:04d}",
                target_radius=TARGET_RADIUS,
                max_steps=OBSERVE_FRAMES + EXECUTION_STEPS,
            )
            result = _save_rollout(
                dataset,
                case=case,
                template_group=template_group,
                template=template,
                time_offset_s=time_offset_s,
                xy_offset=xy_offset,
                seed=int(args.seed + attempts),
            )
            attempts += 1
            if result["success"]:
                successes += 1
                collected_for_template += 1
                metadata["quota_progress"][template_name] = int(collected_for_template)
                metadata["successes"].append(result)
                print(
                    f"[success] {successes:03d}/{int(args.episodes):03d} "
                    f"group={template_group} template={template.name} "
                    f"quota={collected_for_template:02d}/{target_episodes:02d} "
                    f"attempt={attempts:04d} steps={result['steps']}",
                    flush=True,
                )
            else:
                slim = {k: v for k, v in result.items() if k != "rows"}
                metadata["failures"].append(slim)
                print(
                    f"[failed] success={successes:03d}/{int(args.episodes):03d} "
                    f"group={template_group} template={template.name} "
                    f"quota={collected_for_template:02d}/{target_episodes:02d} "
                    f"attempt={attempts:04d} phase_counts={result.get('phase_counts')}",
                    flush=True,
                )
            metadata["attempts"] = int(attempts)
            metadata["episodes_collected"] = int(successes)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if exhausted_attempts:
            break

    if successes < int(args.episodes):
        raise RuntimeError(f"Only collected {successes}/{args.episodes} successful episodes after {attempts} attempts")
    print(f"dataset={output}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()

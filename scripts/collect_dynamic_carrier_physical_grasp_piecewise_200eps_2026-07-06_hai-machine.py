#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
import importlib.util
import json
import math
import os
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
PHYSICAL_PROBE_SCRIPT = REPO_ROOT / "tmp" / "physical_grasp_dynamic_carrier_probe_2026-07-06_hai-machine.py"
CASE_CONFIG = REPO_ROOT / "configs" / "dynamic_carrier_cases.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "lerobot" / "dynamic_carrier_physical_grasp_piecewise_formal_200eps_crf18_2026-07-06_hai-machine"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG_PATH))
for path in (REPO_ROOT, LIBERO_ROOT, FASTWAM_ROOT / "src", FASTWAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fastwam.datasets.lerobot.lerobot.lerobot_dataset as lerobot_dataset_module
from fastwam.datasets.lerobot.lerobot.datasets import video_utils as lerobot_video_utils

LeRobotDataset = lerobot_dataset_module.LeRobotDataset
from ttt4dynamics.cases import load_cases
from ttt4dynamics.dynamic_env import DynamicCarrierEnv, create_libero_env_for_case
from ttt4dynamics.planner import PlannerConfig, PlannerPhase
from ttt4dynamics.trajectories import TrajectoryState

FPS = 20
CAMERA_RESOLUTION = 224
OBSERVE_FRAMES = 160
EXECUTION_STEPS = 260
POST_DONE_SETTLE_FRAMES = 24
DEMO_SPEED = 0.12
FORMAL_SUCCESS_RADIUS_M = 0.060
VIDEO_CRF = 18
TASK_PROMPT = [
    "dynamic carrier physical grasp piecewise formal dataset",
    "observe a moving flat platform, physically grasp the cream cheese with the real gripper, and place it on the target",
    "piecewise trajectory dataset with real robosuite fingerpad grasp",
    "no fake attach during execution; physical gripper grasp is required",
]
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
PHASE_CANDIDATES = [0.25, 0.20, 0.30, 0.15, 0.35, 0.10, 0.40]
GRASP_XY_OFFSET_CANDIDATES = [
    (0.0, 0.0),
    (0.010, 0.0),
    (-0.010, 0.0),
    (0.0, 0.010),
    (0.0, -0.010),
    (0.015, 0.0),
    (-0.015, 0.0),
    (0.0, 0.015),
    (0.0, -0.015),
]
PLACE_XY_COMPENSATION_BY_TEMPLATE = {
    "obtuse_triangle": (0.050, 0.011),
    "kite_quad": (0.050, 0.011),
}
FORMAL_ROLLOUT_CONFIGS = [
    {"template_name": "line_bounce", "phase_fraction": 0.25, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "arc_bounce_60deg", "phase_fraction": 0.35, "grasp_xy_offset": (-0.01, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "line_plus_arc_90deg", "phase_fraction": 0.10, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "line_plus_arc_120deg", "phase_fraction": 0.10, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "right_triangle", "phase_fraction": 0.40, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "obtuse_triangle", "phase_fraction": 0.35, "grasp_xy_offset": (0.0, -0.01), "place_xy_compensation": (0.050, 0.011)},
    {"template_name": "acute_triangle", "phase_fraction": 0.30, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "triangle_with_arc_side", "phase_fraction": 0.15, "grasp_xy_offset": (0.0, 0.015), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "right_angle_quad", "phase_fraction": 0.25, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "trapezoid_quad", "phase_fraction": 0.10, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "kite_quad", "phase_fraction": 0.15, "grasp_xy_offset": (0.0, -0.01), "place_xy_compensation": (0.050, 0.011)},
    {"template_name": "quad_with_arc_side", "phase_fraction": 0.25, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "circle_small_left", "phase_fraction": 0.20, "grasp_xy_offset": (-0.01, 0.0), "place_xy_compensation": (0.0, 0.0)},
    {"template_name": "circle_medium_right", "phase_fraction": 0.30, "grasp_xy_offset": (0.0, 0.0), "place_xy_compensation": (0.0, 0.0)},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-id", default="ttt_dynamic_carrier_physical_grasp_piecewise_formal")
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--video-crf", type=int, default=VIDEO_CRF)
    return parser.parse_args()


def load_piecewise_module() -> Any:
    spec = importlib.util.spec_from_file_location("piecewise_carrier_demo", PIECEWISE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.NOMINAL_SPEED = float(DEMO_SPEED)
    return module


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_high_quality_video_encoder(crf: int) -> None:
    base_encode_video_frames = lerobot_video_utils.encode_video_frames

    def encode_video_frames_high_quality(*args: Any, **kwargs: Any) -> None:
        kwargs["crf"] = int(crf)
        return base_encode_video_frames(*args, **kwargs)

    lerobot_dataset_module.encode_video_frames = encode_video_frames_high_quality


def build_example_plan(episodes: int) -> list[dict[str, Any]]:
    base = [
        {"group": group_name, "template_name": template_name}
        for group_name, template_names in TEMPLATE_GROUPS
        for template_name in template_names
    ]
    if int(episodes) <= len(base):
        return base[: int(episodes)]
    plan = []
    while len(plan) < int(episodes):
        plan.extend(base)
    return plan[: int(episodes)]


def build_formal_plan(episodes: int) -> list[dict[str, Any]]:
    plan = []
    for index in range(int(episodes)):
        config = dict(FORMAL_ROLLOUT_CONFIGS[index % len(FORMAL_ROLLOUT_CONFIGS)])
        config["formal_config_index"] = int(index % len(FORMAL_ROLLOUT_CONFIGS))
        config["repeat_index"] = int(index // len(FORMAL_ROLLOUT_CONFIGS))
        plan.append(config)
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


class PhysicalPayloadDynamicCarrierEnv(DynamicCarrierEnv):
    """Moves carrier kinematically during observation, then releases payload to real MuJoCo physics.

    This disables DynamicCarrierEnv's fake payload attachment completely. After physical handoff, payload motion
    comes only from contacts with the gripper / scene.
    """

    def __init__(self, base_env: Any, case: Any):
        super().__init__(base_env, case)
        self.physics_payload_enabled = False

    def reset(self, init_state: np.ndarray | None = None) -> dict[str, Any]:
        self.physics_payload_enabled = False
        return super().reset(init_state=init_state)

    def enable_physical_payload(self) -> None:
        if self.physics_payload_enabled:
            return
        self.physics_payload_enabled = True
        self.payload_detached = True
        self.payload_attached_to_gripper = False
        self._release_payload_pos = None
        obj = self.inner_env.get_object(self.case.payload_name)
        if getattr(obj, "joints", None):
            try:
                qvel = np.asarray(self.inner_env.sim.data.get_joint_qvel(obj.joints[-1]), dtype=np.float64)
                qvel = np.zeros_like(qvel)
                qvel[:2] = self.carrier_velocity_xy()[:2]
                self.inner_env.sim.data.set_joint_qvel(obj.joints[-1], qvel)
            except Exception:
                pass
        self.inner_env.sim.forward()

    def _update_payload_attachment(self, action: np.ndarray, obs: dict[str, Any]) -> None:
        return

    def _apply_kinematic_motion(self, t: float, obs: dict[str, Any] | None = None) -> None:
        if self._carrier_z is None:
            return
        state = self.trajectory.sample(t)
        carrier_pos = np.asarray([state.xy[0], state.xy[1], self._carrier_z], dtype=np.float64)
        self.set_object_pose(self.case.carrier_name, carrier_pos, self._carrier_quat)
        if not self.physics_payload_enabled:
            offset = self._payload_offset if self._payload_offset is not None else np.zeros(3)
            payload_pos = carrier_pos + offset
            self.set_object_pose(self.case.payload_name, payload_pos, self._payload_quat)
        self.inner_env.sim.forward()

    def robosuite_grasping(self) -> bool:
        obj = self.inner_env.get_object(self.case.payload_name)
        try:
            return bool(self.inner_env._check_grasp(gripper=self.inner_env.robots[0].gripper, object_geoms=obj))
        except Exception:
            return False


class PhysicalDirectPlanner:
    def __init__(self, env: PhysicalPayloadDynamicCarrierEnv):
        self.env = env
        self.config = PlannerConfig(
            position_gain=14.0,
            max_pos_action=1.0,
            xy_tolerance=0.04,
            target_xy_tolerance=0.035,
            z_tolerance=0.035,
            box_approach_z=1.12,
            lift_z=1.12,
            place_z=0.99,
            release_hold_steps=24,
            grasp_hold_steps=28,
        )
        self.phase = PlannerPhase.APPROACH
        self.phase_steps = 0
        self.locked_grasp_xy: np.ndarray | None = None
        self.locked_intercept_tau: float | None = None

    def reset(self) -> None:
        self.phase = PlannerPhase.APPROACH
        self.phase_steps = 0
        self.locked_grasp_xy = None
        self.locked_intercept_tau = None

    def is_done(self) -> bool:
        return self.phase == PlannerPhase.DONE

    def _lock_grasp_xy(self) -> np.ndarray:
        if self.locked_grasp_xy is None:
            eef = self.env.eef_position()
            approach_z = self.config.box_approach_z
            grasp_z = float(self.env.payload_position()[2] + 0.010)
            tau = 0.75
            for _ in range(8):
                xy = self.env.predict_payload_xy(self.env.t + tau)
                approach = np.asarray([xy[0], xy[1], approach_z], dtype=np.float64)
                travel_time = (float(np.linalg.norm(approach - eef)) + abs(approach_z - grasp_z)) / 0.24
                tau = float(np.clip(travel_time + 0.15, 0.30, 1.80))
            self.locked_intercept_tau = tau
            self.locked_grasp_xy = self.env.predict_payload_xy(self.env.t + tau)
        return self.locked_grasp_xy.copy()

    def _target(self) -> np.ndarray:
        if self.phase in {PlannerPhase.APPROACH, PlannerPhase.DESCEND, PlannerPhase.GRASP}:
            xy = self._lock_grasp_xy()
            if self.phase == PlannerPhase.APPROACH:
                return np.asarray([xy[0], xy[1], self.config.box_approach_z], dtype=np.float64)
            return np.asarray([xy[0], xy[1], self.env.payload_position()[2] + 0.010], dtype=np.float64)
        if self.phase == PlannerPhase.LIFT:
            eef = self.env.eef_position()
            return np.asarray([eef[0], eef[1], self.config.lift_z], dtype=np.float64)
        if self.phase == PlannerPhase.MOVE_TO_TARGET:
            tx, ty = self.env.case.target_xy
            return np.asarray([tx, ty, self.config.lift_z], dtype=np.float64)
        if self.phase in {PlannerPhase.LOWER, PlannerPhase.RELEASE}:
            tx, ty = self.env.case.target_xy
            return np.asarray([tx, ty, self.config.place_z], dtype=np.float64)
        return self.env.eef_position()

    def _gripper(self) -> float:
        if self.phase in {PlannerPhase.GRASP, PlannerPhase.LIFT, PlannerPhase.MOVE_TO_TARGET, PlannerPhase.LOWER}:
            return 1.0
        return -1.0

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        if self.phase == PlannerPhase.GRASP and self.phase_steps >= 10:
            self.env.enable_physical_payload()
        eef = self.env.eef_position(obs)
        target = self._target()
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(self.config.position_gain * (target - eef), -1.0, 1.0)
        action[-1] = self._gripper()
        self._advance(eef, target)
        self.phase_steps += 1
        return action

    def _advance(self, eef: np.ndarray, target: np.ndarray) -> None:
        xy_err = float(np.linalg.norm(eef[:2] - target[:2]))
        z_err = float(abs(eef[2] - target[2]))
        target_xy_tol = max(self.config.target_xy_tolerance, float(self.env.case.target_radius) * 0.9)
        if self.phase == PlannerPhase.APPROACH and xy_err <= self.config.xy_tolerance and z_err <= self.config.z_tolerance:
            self.phase = PlannerPhase.DESCEND
            self.phase_steps = 0
        elif self.phase == PlannerPhase.DESCEND and xy_err <= self.config.xy_tolerance and z_err <= self.config.z_tolerance:
            self.phase = PlannerPhase.GRASP
            self.phase_steps = 0
        elif self.phase == PlannerPhase.GRASP and self.phase_steps >= self.config.grasp_hold_steps:
            self.phase = PlannerPhase.LIFT
            self.phase_steps = 0
        elif self.phase == PlannerPhase.LIFT and z_err <= self.config.z_tolerance:
            self.phase = PlannerPhase.MOVE_TO_TARGET
            self.phase_steps = 0
        elif self.phase == PlannerPhase.MOVE_TO_TARGET and xy_err <= target_xy_tol:
            self.phase = PlannerPhase.LOWER
            self.phase_steps = 0
        elif self.phase == PlannerPhase.LOWER and xy_err <= target_xy_tol and z_err <= self.config.z_tolerance:
            self.phase = PlannerPhase.RELEASE
            self.phase_steps = 0
        elif self.phase == PlannerPhase.RELEASE and self.phase_steps >= self.config.release_hold_steps:
            self.phase = PlannerPhase.DONE
            self.phase_steps = 0


def _build_features(camera_resolution: int) -> dict[str, dict[str, Any]]:
    image_shape = (3, int(camera_resolution), int(camera_resolution))
    return {
        "observation.images.image": {"dtype": "video", "shape": image_shape, "names": ["channel", "height", "width"]},
        "observation.images.wrist_image": {"dtype": "video", "shape": image_shape, "names": ["channel", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["eef_x", "eef_y", "eef_z", "eef_axis_x", "eef_axis_y", "eef_axis_z", "gripper_qpos_0", "gripper_qpos_1"]},
        "observation.carrier_state": {"dtype": "float32", "shape": (4,), "names": ["carrier_x", "carrier_y", "carrier_vx", "carrier_vy"]},
        "observation.payload_state": {"dtype": "float32", "shape": (6,), "names": ["payload_x", "payload_y", "payload_z", "physical_payload_enabled", "robosuite_grasping", "success_now"]},
        "observation.object_poses": {"dtype": "float32", "shape": (21,), "names": ["carrier_x", "carrier_y", "carrier_z", "carrier_qw", "carrier_qx", "carrier_qy", "carrier_qz", "payload_x", "payload_y", "payload_z", "payload_qw", "payload_qx", "payload_qy", "payload_qz", "target_x", "target_y", "target_z", "target_qw", "target_qx", "target_qy", "target_qz"]},
        "observation.dynamic_time": {"dtype": "float32", "shape": (4,), "names": ["t", "phase_sin", "phase_cos", "is_execute"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"]},
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
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8), np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)


def _obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate([np.asarray(obs["robot0_eef_pos"], dtype=np.float32), _quat_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)), np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)], axis=0).astype(np.float32)


def _env_action_to_dataset_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[-1] = (1.0 - out[-1]) / 2.0
    return out


def _write_image_for_last_frame(dataset: LeRobotDataset, key: str, frame_index: int, image: np.ndarray) -> None:
    path = dataset._get_image_file_path(episode_index=dataset.episode_buffer["episode_index"], image_key=key, frame_index=frame_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=95)


def _remove_current_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    episode_index = dataset.episode_buffer["episode_index"]
    for key in dataset.meta.video_keys:
        image_dir = dataset._get_image_file_path(episode_index=episode_index, image_key=key, frame_index=0).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def _pose_vec(pose: dict[str, Any]) -> np.ndarray:
    return np.asarray(pose["position"] + pose["quat_wxyz"], dtype=np.float32)


def _target_pose(case: Any) -> dict[str, Any]:
    target_z = case.target_z if case.target_z is not None else 0.935
    return {"name": "virtual_target", "position": [float(case.target_xy[0]), float(case.target_xy[1]), float(target_z)], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "radius": float(case.target_radius)}


def _object_poses(env: PhysicalPayloadDynamicCarrierEnv) -> dict[str, Any]:
    carrier_pos, carrier_quat = env.get_object_pose(env.case.carrier_name)
    payload_pos, payload_quat = env.get_object_pose(env.case.payload_name)
    return {
        "carrier": {"name": env.case.carrier_name, "position": carrier_pos.astype(float).tolist(), "quat_wxyz": carrier_quat.astype(float).tolist()},
        "payload": {"name": env.case.payload_name, "position": payload_pos.astype(float).tolist(), "quat_wxyz": payload_quat.astype(float).tolist()},
        "target": _target_pose(env.case),
    }


def _robosuite_grasping(env: Any) -> bool:
    obj = env.inner_env.get_object(env.case.payload_name)
    try:
        return bool(env.inner_env._check_grasp(gripper=env.inner_env.robots[0].gripper, object_geoms=obj))
    except Exception:
        return False


def _frame_payload(env: PhysicalPayloadDynamicCarrierEnv, obs: dict[str, Any], action: np.ndarray, *, frame_count: int, is_execute: bool, phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    agent, wrist = _obs_to_images(obs)
    trajectory_state = env.trajectory.sample(env.t)
    trajectory_phase = env.trajectory.phase(env.t) if hasattr(env.trajectory, "phase") else 0.0
    payload = env.payload_position()
    poses = _object_poses(env)
    grasping = _robosuite_grasping(env)
    success_now = bool(env.check_success())
    frame = {
        "observation.images.image": agent,
        "observation.images.wrist_image": wrist,
        "observation.state": _obs_to_state(obs),
        "observation.carrier_state": np.asarray([trajectory_state.xy[0], trajectory_state.xy[1], trajectory_state.velocity_xy[0], trajectory_state.velocity_xy[1]], dtype=np.float32),
        "observation.payload_state": np.asarray([payload[0], payload[1], payload[2], float(env.physics_payload_enabled), float(grasping), float(success_now)], dtype=np.float32),
        "observation.object_poses": np.concatenate([_pose_vec(poses["carrier"]), _pose_vec(poses["payload"]), _pose_vec(poses["target"])], axis=0).astype(np.float32),
        "observation.dynamic_time": np.asarray([env.t, math.sin(2.0 * math.pi * trajectory_phase), math.cos(2.0 * math.pi * trajectory_phase), float(is_execute)], dtype=np.float32),
        "action": _env_action_to_dataset_action(action),
    }
    row = {
        "frame": int(frame_count),
        "segment": "execute" if is_execute else "observe",
        "phase": str(phase),
        "t": float(env.t),
        "trajectory_phase": float(trajectory_phase),
        "carrier_xy": trajectory_state.xy.astype(float).tolist(),
        "carrier_velocity_xy": trajectory_state.velocity_xy.astype(float).tolist(),
        "payload_xyz": payload.astype(float).tolist(),
        "object_poses": poses,
        "eef_xyz": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).astype(float).tolist(),
        "eef_quat_wxyz": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).astype(float).tolist(),
        "action_env": np.asarray(action, dtype=np.float64).astype(float).tolist(),
        "physical_payload_enabled": bool(env.physics_payload_enabled),
        "robosuite_grasping": bool(grasping),
        "success_now": success_now,
    }
    return frame, row


def _trajectory_config(template: Any, time_offset_s: float, xy_offset: np.ndarray) -> dict[str, Any]:
    return {"template_name": str(template.name), "template_family": str(template.family), "template_description": str(template.description), "template_period_s": float(template.period), "time_offset_s": float(time_offset_s), "xy_offset": np.asarray(xy_offset, dtype=np.float64).astype(float).tolist(), "source_script": str(PIECEWISE_SCRIPT), "speed_mps": float(DEMO_SPEED)}


def save_rollout(dataset: LeRobotDataset, *, physical_probe: Any, template: Any, case: Any, seed: int, time_offset_s: float, xy_offset: np.ndarray, grasp_xy_offset: np.ndarray, place_xy_compensation: np.ndarray) -> dict[str, Any]:
    base_env, init_state, task_description = create_libero_env_for_case(case, repo_root=REPO_ROOT, camera_resolution=CAMERA_RESOLUTION, seed=seed)
    env = physical_probe.PhysicalPayloadDynamicCarrierEnv(base_env, case)
    env.trajectory = OffsetTemplateTrajectory(template, time_offset_s=time_offset_s, xy_offset=xy_offset)
    class TargetCompensatedPlanner(physical_probe.TimeBasedPhysicalDirectPlanner):
        def __init__(self, *planner_args: Any, place_xy_compensation: np.ndarray, **planner_kwargs: Any):
            super().__init__(*planner_args, **planner_kwargs)
            self.place_xy_compensation = np.asarray(place_xy_compensation, dtype=np.float64)

        def _target(self) -> np.ndarray:
            if self.phase == PlannerPhase.MOVE_TO_TARGET:
                tx, ty = self.env.case.target_xy
                comp = self.place_xy_compensation
                return np.asarray([tx + comp[0], ty + comp[1], self.config.lift_z], dtype=np.float64)
            if self.phase in {PlannerPhase.LOWER, PlannerPhase.RELEASE}:
                tx, ty = self.env.case.target_xy
                comp = self.place_xy_compensation
                return np.asarray([tx + comp[0], ty + comp[1], self.config.place_z], dtype=np.float64)
            return super()._target()

    planner = TargetCompensatedPlanner(
        None,
        env,
        physics_mode="grasp_after_close",
        grasp_z_bias=-0.045,
        handoff_after_grasp_steps=10,
        lift_z=1.12,
        grasp_xy_offset=tuple(np.asarray(grasp_xy_offset, dtype=np.float64).astype(float).tolist()),
        place_xy_compensation=place_xy_compensation,
    )
    planner.config = replace(planner.config, grasp_hold_steps=18, release_hold_steps=16)
    rows = []
    frame_count = 0
    no_op = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    try:
        _remove_current_episode_images(dataset)
        obs = env.reset(init_state=init_state)
        initial_conditions = {"reset_seed": int(seed), "sim_init_state": np.asarray(env.get_sim_state(), dtype=np.float64).astype(float).tolist(), "sim_init_state_available": True, "object_poses": _object_poses(env), "eef_xyz": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).astype(float).tolist(), "eef_quat_wxyz": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).astype(float).tolist()}
        for _ in range(OBSERVE_FRAMES):
            frame, row = _frame_payload(env, obs, no_op, frame_count=frame_count, is_execute=False, phase="observe")
            dataset.add_frame(frame, task=TASK_PROMPT, timestamp=frame_count / FPS)
            _write_image_for_last_frame(dataset, "observation.images.image", frame_count, frame["observation.images.image"])
            _write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_count, frame["observation.images.wrist_image"])
            rows.append(row)
            frame_count += 1
            obs, _, _, _ = env.step(no_op)
        planner.reset()
        phase_counts: dict[str, int] = {}
        for _ in range(EXECUTION_STEPS):
            phase = str(planner.phase.value)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            action = planner.act(obs)
            frame, row = _frame_payload(env, obs, action, frame_count=frame_count, is_execute=True, phase=phase)
            row["locked_grasp_xy"] = planner.locked_grasp_xy.astype(float).tolist() if planner.locked_grasp_xy is not None else None
            row["locked_intercept_tau"] = float(planner.locked_intercept_tau) if planner.locked_intercept_tau is not None else None
            dataset.add_frame(frame, task=TASK_PROMPT, timestamp=frame_count / FPS)
            _write_image_for_last_frame(dataset, "observation.images.image", frame_count, frame["observation.images.image"])
            _write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_count, frame["observation.images.wrist_image"])
            rows.append(row)
            frame_count += 1
            obs, _, _, _ = env.step(action)
            if planner.is_done():
                break
        for _ in range(POST_DONE_SETTLE_FRAMES):
            frame, row = _frame_payload(env, obs, no_op, frame_count=frame_count, is_execute=True, phase="post_done_settle")
            dataset.add_frame(frame, task=TASK_PROMPT, timestamp=frame_count / FPS)
            _write_image_for_last_frame(dataset, "observation.images.image", frame_count, frame["observation.images.image"])
            _write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_count, frame["observation.images.wrist_image"])
            rows.append(row)
            frame_count += 1
            obs, _, _, _ = env.step(no_op)
        payload_zs = [float(row["payload_xyz"][2]) for row in rows]
        initial_payload_z = float(payload_zs[0])
        max_payload_lift = float(max(payload_zs) - initial_payload_z)
        grasp_true_frames = int(sum(int(row.get("robosuite_grasping", False)) for row in rows))
        final_payload = env.payload_position()
        target_xy = np.asarray(case.target_xy, dtype=np.float64)
        final_xy_error = float(np.linalg.norm(final_payload[:2] - target_xy))
        success_radius = max(float(case.target_radius), float(FORMAL_SUCCESS_RADIUS_M))
        success = bool(final_xy_error <= success_radius and grasp_true_frames >= 20 and max_payload_lift >= 0.04)
        if success:
            episode_index = int(dataset.meta.total_episodes)
            dataset.save_episode()
        else:
            episode_index = None
            _remove_current_episode_images(dataset)
            dataset.clear_episode_buffer()
        trajectory_config = _trajectory_config(template, time_offset_s, xy_offset)
        trajectory_config["grasp_xy_offset"] = np.asarray(grasp_xy_offset, dtype=np.float64).astype(float).tolist()
        trajectory_config["place_xy_compensation"] = np.asarray(place_xy_compensation, dtype=np.float64).astype(float).tolist()
        return {"success": success, "episode_index": episode_index, "steps": int(frame_count), "final_payload_xyz": final_payload.astype(float).tolist(), "final_xy_error_m": final_xy_error, "success_radius_m": success_radius, "max_payload_lift_m": max_payload_lift, "initial_payload_z": initial_payload_z, "robosuite_grasp_true_frames": grasp_true_frames, "phase_counts": phase_counts, "case": case.as_dict(), "trajectory_config": trajectory_config, "initial_conditions": initial_conditions, "task_description": task_description, "rows": rows}
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
    install_high_quality_video_encoder(int(args.video_crf))
    piecewise = load_piecewise_module()
    physical_probe = import_module(PHYSICAL_PROBE_SCRIPT, "physical_grasp_probe_successful")
    templates = {item.name: item for item in piecewise.make_templates()}
    case = next(case for case in load_cases(CASE_CONFIG) if case.case_id == "flat_line_medium")
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=output, fps=FPS, features=_build_features(CAMERA_RESOLUTION), use_videos=True, video_codec=args.video_codec, is_compute_episode_stats_image=False)
    plan = build_formal_plan(int(args.episodes))
    metadata = {"created_at": dt.datetime.now().isoformat(), "dataset_type": "dynamic_carrier_physical_grasp_piecewise_formal_no_fake_attach_execution", "episodes_requested": int(args.episodes), "seed": int(args.seed), "camera_resolution": CAMERA_RESOLUTION, "fps": FPS, "observe_frames": OBSERVE_FRAMES, "post_done_settle_frames": POST_DONE_SETTLE_FRAMES, "case_id": case.case_id, "template_plan": plan, "formal_rollout_configs": FORMAL_ROLLOUT_CONFIGS, "video_codec": str(args.video_codec), "video_crf": int(args.video_crf), "formal_success_radius_m": FORMAL_SUCCESS_RADIUS_M, "physical_grasp_policy": {"fake_attach": False, "robosuite_grasp_required": True, "grasp_target": "payload_position_z + 0.010 m; calibrated from official LIBERO static grasp probe", "intercept_policy": "fixed stable intercept phase and grasp xy offset per trajectory template from validated 14-template rollout", "placement_policy": "template-specific EEF target compensation is recorded when physical grasp has systematic payload-to-EEF offset", "physics_mode": "enable physical payload after 10 close-grasp frames, lift after 18 grasp-phase frames", "success_criterion": "formal threshold for this physical-grasp dataset: final xy within max(case target_radius, 0.060m), robosuite _check_grasp true for >=20 frames, payload lift >=4cm"}, "successes": [], "failures": []}
    metadata_path = output / "dynamic_carrier_physical_grasp_piecewise_formal_metadata.json"
    successes = 0
    attempts = 0
    while successes < int(args.episodes) and attempts < int(args.max_attempts):
        entry = plan[successes]
        template = templates[entry["template_name"]]
        phase_fraction = float(entry["phase_fraction"])
        grasp_xy_offset = np.asarray(entry["grasp_xy_offset"], dtype=np.float64)
        time_offset_s = float(phase_fraction * template.period)
        xy_offset = np.zeros(2, dtype=np.float64)
        place_xy_compensation = np.asarray(entry["place_xy_compensation"], dtype=np.float64)
        result = save_rollout(dataset, physical_probe=physical_probe, template=template, case=case, seed=int(args.seed + attempts), time_offset_s=time_offset_s, xy_offset=xy_offset, grasp_xy_offset=grasp_xy_offset, place_xy_compensation=place_xy_compensation)
        attempts += 1
        result["formal_index"] = int(successes)
        result["formal_config_index"] = int(entry["formal_config_index"])
        result["repeat_index"] = int(entry["repeat_index"])
        result["trajectory_config"]["phase_fraction"] = float(phase_fraction)
        if result["success"]:
            successes += 1
            metadata["successes"].append(result)
            print(f"[success] {successes:03d}/{int(args.episodes):03d} attempt={attempts:04d} config={int(entry['formal_config_index']):02d} repeat={int(entry['repeat_index']):02d} template={template.name} phase={phase_fraction:.2f} grasp_offset={grasp_xy_offset.tolist()} grasp={result['robosuite_grasp_true_frames']} lift={result['max_payload_lift_m']:.3f} final_xy={result['final_xy_error_m']:.3f} steps={result['steps']}", flush=True)
        else:
            slim = {key: value for key, value in result.items() if key != "rows"}
            metadata["failures"].append(slim)
            print(f"[failed] success={successes:03d}/{int(args.episodes):03d} attempt={attempts:04d} config={int(entry['formal_config_index']):02d} repeat={int(entry['repeat_index']):02d} template={template.name} phase={phase_fraction:.2f} grasp_offset={grasp_xy_offset.tolist()} grasp={result['robosuite_grasp_true_frames']} lift={result['max_payload_lift_m']:.3f} final_xy={result['final_xy_error_m']:.3f}", flush=True)
        metadata["attempts"] = int(attempts)
        metadata["episodes_collected"] = int(successes)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if successes < int(args.episodes):
        raise RuntimeError(f"Only collected {successes}/{args.episodes} successful physical-grasp episodes after {attempts} attempts")
    print(f"dataset={output}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()

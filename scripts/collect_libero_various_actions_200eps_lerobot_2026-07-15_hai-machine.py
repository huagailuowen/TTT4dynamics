#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
LIBERO_REPO = REPO_ROOT.parent / "LIBERO"
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM-TTT"
for path in (REPO_ROOT, SCRIPTS_DIR, LIBERO_REPO, FASTWAM_ROOT, FASTWAM_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

BASE_COLLECTOR_SCRIPT = (
    SCRIPTS_DIR / "collect_libero_push_box_formal_6fric_50pair_35_35_direct_lerobot_hai-machine.py"
)
DEMO_SCRIPT = SCRIPTS_DIR / "render_libero_action_world_model_diverse_demos_2026-07-15_hai-machine.py"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "various_actions"
    / "libero_mu0100_various_actions_200eps_lerobot_2026-07-15_hai-machine"
)

EPISODES_PER_FAMILY = 8
BASE_SEED = 20260715
SIM_SEED = 20260715
TASK_PROMPT = [
    "explore robot action dynamics on a tabletop",
    "move the robot along diverse paths around and with a box",
    "observe how commanded actions change the robot and the scene",
    "action-conditioned world-model interaction trajectory",
]

FREE_FAMILIES = [
    "free_circle_cw",
    "free_circle_ccw",
    "free_ellipse_cw",
    "free_ellipse_ccw",
    "free_square_cw",
    "free_square_ccw",
    "free_triangle_cw",
    "free_triangle_ccw",
    "free_spiral_cw",
    "free_spiral_ccw",
    "free_figure8_forward",
    "free_figure8_reverse",
]

CONTACT_FAMILIES = [
    "straight_push_slow",
    "straight_push_medium",
    "tap_medium",
    "ram_fast",
    "lateral_push_pos_y",
    "lateral_push_neg_y",
    "diagonal_push_pos30",
    "diagonal_push_neg30",
    "diagonal_push_pos60",
    "diagonal_push_neg60",
    "curved_contact_sweep_cw",
    "curved_contact_sweep_ccw",
    "vertical_press_or_grasp",
]

FAMILIES = FREE_FAMILIES + CONTACT_FAMILIES
EXPECTED_EPISODES = len(FAMILIES) * EPISODES_PER_FAMILY


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_COLLECTOR_SCRIPT, "various_actions_base_collector_hai_machine")
demo = load_module(DEMO_SCRIPT, "various_actions_demo_base_hai_machine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect 200 diverse fixed-friction LIBERO action episodes in LeRobot format.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def gripper_profile(mode: int) -> Callable[[int, int], float]:
    def value(index: int, count: int) -> float:
        if mode == 0:
            return -1.0
        if mode == 1:
            return 1.0
        if mode == 2:
            return -1.0 if index < count // 2 else 1.0
        return 1.0 if index < count // 2 else -1.0

    return value


class LeRobotEpisodeRecorder:
    def __init__(self, *, env: Any, dataset: Any):
        self.env = env
        self.dataset = dataset
        self.obs = env.reset()
        self.initial_box_xyz, _ = env.box_pose()
        self.rows: list[dict[str, Any]] = []
        self.frame_index = 0
        self.last_gripper = -1.0
        self.phase_counts: Counter[str] = Counter()
        base.remove_current_episode_images(dataset)

    def _add_frame(self, obs: dict[str, Any], action: np.ndarray, phase: str) -> None:
        agent, wrist = base._obs_to_images(obs)
        frame = {
            "observation.images.image": agent,
            "observation.images.wrist_image": wrist,
            "observation.state": base._obs_to_state(obs),
            "action": base._env_action_to_fastwam_action(np.asarray(action, dtype=np.float32)),
        }
        self.dataset.add_frame(
            frame,
            task=TASK_PROMPT,
            timestamp=float(self.frame_index) / float(demo.FPS),
        )
        base.write_image_for_last_frame(
            self.dataset,
            "observation.images.image",
            self.frame_index,
            agent,
        )
        base.write_image_for_last_frame(
            self.dataset,
            "observation.images.wrist_image",
            self.frame_index,
            wrist,
        )
        self.phase_counts[phase] += 1
        self.frame_index += 1

    def step(self, action: np.ndarray, phase: str) -> None:
        command = np.asarray(action, dtype=np.float64).copy()
        command[:6] = np.clip(command[:6], -1.0, 1.0)
        command[-1] = float(np.clip(command[-1], -1.0, 1.0))
        obs_t = self.obs
        eef_t = np.asarray(obs_t["robot0_eef_pos"], dtype=np.float64)
        box_xyz_t, _ = self.env.box_pose()
        box_qvel_t = self.env.box_velocity()

        self._add_frame(obs_t, command, phase)
        obs_tp1, _, _, _ = self.env.step(command)
        eef_tp1 = np.asarray(obs_tp1["robot0_eef_pos"], dtype=np.float64)
        box_xyz_tp1, _ = self.env.box_pose()
        box_qvel_tp1 = self.env.box_velocity()
        self.rows.append(
            {
                "phase": phase,
                "action_env": command,
                "action_fastwam": base._env_action_to_fastwam_action(command),
                "eef_xyz_t": eef_t,
                "eef_xyz_tp1": eef_tp1,
                "box_xyz_t": box_xyz_t,
                "box_xyz_tp1": box_xyz_tp1,
                "box_qvel_t": box_qvel_t,
                "box_qvel_tp1": box_qvel_tp1,
                "robot_box_contact": demo.robot_box_contact(self.env),
                "robosuite_grasping": demo.grasping(self.env),
            }
        )
        self.last_gripper = float(command[-1])
        self.obs = obs_tp1

    def add_terminal_observation(self) -> None:
        terminal_action = np.zeros(7, dtype=np.float64)
        terminal_action[-1] = float(self.last_gripper)
        self._add_frame(self.obs, terminal_action, "terminal_observation")

    def hold(self, steps: int, *, gripper: float, phase: str) -> None:
        for _ in range(int(steps)):
            action = np.zeros(7, dtype=np.float64)
            action[-1] = float(gripper)
            self.step(action, phase)

    def move_to(
        self,
        target_xyz: np.ndarray | tuple[float, float, float],
        *,
        steps: int,
        gripper: float,
        phase: str,
        gain: float = 3.5,
        max_action: float = 0.20,
    ) -> None:
        target = np.asarray(target_xyz, dtype=np.float64)
        for _ in range(int(steps)):
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
            action = np.zeros(7, dtype=np.float64)
            action[:3] = np.clip(float(gain) * (target - eef), -float(max_action), float(max_action))
            action[-1] = float(gripper)
            self.step(action, phase)

    def follow_points(
        self,
        points_xyz: np.ndarray,
        *,
        gripper_fn: Callable[[int, int], float],
        phase: str,
        gain: float,
        max_action: float,
    ) -> None:
        points = np.asarray(points_xyz, dtype=np.float64)
        for index, target in enumerate(points):
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
            action = np.zeros(7, dtype=np.float64)
            action[:3] = np.clip(float(gain) * (target - eef), -float(max_action), float(max_action))
            action[-1] = float(gripper_fn(index, len(points)))
            self.step(action, phase)

    def track_line(
        self,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        *,
        steps: int,
        gripper: float,
        phase: str,
        gain: float,
        max_action: float,
    ) -> None:
        start = np.asarray(start_xyz, dtype=np.float64)
        end = np.asarray(end_xyz, dtype=np.float64)
        points = []
        for index in range(int(steps)):
            alpha = demo.smootherstep(float(index + 1) / float(max(1, steps)))
            points.append((1.0 - alpha) * start + alpha * end)
        self.follow_points(
            np.asarray(points),
            gripper_fn=lambda _index, _count: float(gripper),
            phase=phase,
            gain=gain,
            max_action=max_action,
        )

    def directional_pulse(
        self,
        direction_xy: np.ndarray,
        amplitude: float,
        *,
        line_xy: np.ndarray,
        contact_z: float,
        phase: str,
    ) -> None:
        direction = demo.normalized(direction_xy)
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
        lateral_error = float(np.dot(np.asarray(line_xy, dtype=np.float64) - eef[:2], lateral))
        action = np.zeros(7, dtype=np.float64)
        action[:2] = float(amplitude) * direction
        action[:2] += float(np.clip(3.0 * lateral_error, -0.10, 0.10)) * lateral
        action[2] = float(np.clip(3.0 * (float(contact_z) - eef[2]), -0.12, 0.12))
        action[-1] = 1.0
        self.step(action, phase)

    def contact_triggered_pulse(
        self,
        direction_xy: np.ndarray,
        *,
        peak: float,
        line_xy: np.ndarray,
        contact_z: float,
        max_precontact_steps: int,
        hold_after_contact: int,
        phase: str,
    ) -> dict[str, Any]:
        contact_seen = False
        first_contact_step: int | None = None
        post_contact_count = 0
        for local_step in range(int(max_precontact_steps) + int(hold_after_contact)):
            amplitude = 0.5 * float(peak) if local_step == 0 else float(peak)
            self.directional_pulse(
                direction_xy,
                amplitude,
                line_xy=line_xy,
                contact_z=contact_z,
                phase=phase,
            )
            row = self.rows[-1]
            displacement = float(np.linalg.norm(row["box_xyz_tp1"][:2] - self.initial_box_xyz[:2]))
            box_speed = float(np.linalg.norm(row["box_qvel_tp1"][:2]))
            event = bool(row["robot_box_contact"] or displacement > 0.001 or box_speed > 0.03)
            if event and not contact_seen:
                contact_seen = True
                first_contact_step = local_step
            elif contact_seen:
                post_contact_count += 1
                if post_contact_count >= int(hold_after_contact):
                    break
        self.directional_pulse(
            direction_xy,
            0.5 * float(peak),
            line_xy=line_xy,
            contact_z=contact_z,
            phase=f"{phase}_brake",
        )
        self.directional_pulse(
            direction_xy,
            0.0,
            line_xy=line_xy,
            contact_z=contact_z,
            phase=f"{phase}_brake",
        )
        return {
            "contact_seen": contact_seen,
            "first_contact_local_step": first_contact_step,
            "post_contact_steps": post_contact_count,
        }

    def summary(self) -> dict[str, Any]:
        final_box_xyz, _ = self.env.box_pose()
        if self.rows:
            actions = np.asarray([row["action_env"] for row in self.rows], dtype=np.float64)
            box_speeds = np.asarray([np.linalg.norm(row["box_qvel_tp1"][:2]) for row in self.rows])
            eef_speeds = np.asarray(
                [np.linalg.norm(row["eef_xyz_tp1"] - row["eef_xyz_t"]) * demo.FPS for row in self.rows]
            )
        else:
            actions = np.zeros((0, 7), dtype=np.float64)
            box_speeds = np.zeros(0, dtype=np.float64)
            eef_speeds = np.zeros(0, dtype=np.float64)
        return {
            "steps_with_actions": len(self.rows),
            "frames_in_lerobot_episode": int(self.frame_index),
            "terminal_observation_added": True,
            "phase_counts": dict(self.phase_counts),
            "initial_box_xyz_m": self.initial_box_xyz,
            "final_box_xyz_m": final_box_xyz,
            "final_box_displacement_m": float(np.linalg.norm(final_box_xyz[:2] - self.initial_box_xyz[:2])),
            "max_box_planar_speed_mps": float(np.max(box_speeds)) if box_speeds.size else 0.0,
            "max_eef_speed_mps": float(np.max(eef_speeds)) if eef_speeds.size else 0.0,
            "max_abs_action_by_dim": np.max(np.abs(actions), axis=0) if actions.size else np.zeros(7),
            "robot_box_contact_steps": int(sum(bool(row["robot_box_contact"]) for row in self.rows)),
            "robosuite_grasping_steps": int(sum(bool(row["robosuite_grasping"]) for row in self.rows)),
        }


def interpolate_polygon(vertices: np.ndarray, count: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    edge_count = len(vertices) - 1
    per_edge = max(2, int(math.ceil(float(count) / float(edge_count))))
    points = []
    for edge in range(edge_count):
        for alpha in np.linspace(0.0, 1.0, per_edge, endpoint=False):
            points.append((1.0 - alpha) * vertices[edge] + alpha * vertices[edge + 1])
    points.append(vertices[-1])
    return np.asarray(points[:count], dtype=np.float64)


def make_free_path(
    family: str,
    *,
    box_xyz: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    center = np.asarray(box_xyz[:2], dtype=np.float64) + rng.uniform(-0.012, 0.012, size=2)
    z = float(rng.uniform(1.07, 1.13))
    count = int(rng.choice([56, 64, 72]))
    phase = float(rng.uniform(-math.pi, math.pi))
    direction = -1.0 if family.endswith("_cw") or family.endswith("_reverse") else 1.0

    if "circle" in family:
        radius = float(rng.uniform(0.10, 0.145))
        theta = phase + direction * np.linspace(0.0, 2.0 * math.pi, count)
        xy = np.stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)], axis=1)
        params = {"shape": "circle", "radius_m": radius}
    elif "ellipse" in family:
        radius_x = float(rng.uniform(0.12, 0.16))
        radius_y = float(rng.uniform(0.065, 0.10))
        theta = phase + direction * np.linspace(0.0, 2.0 * math.pi, count)
        xy = np.stack(
            [center[0] + radius_x * np.cos(theta), center[1] + radius_y * np.sin(theta)],
            axis=1,
        )
        params = {"shape": "ellipse", "radius_x_m": radius_x, "radius_y_m": radius_y}
    elif "square" in family or "triangle" in family:
        sides = 4 if "square" in family else 3
        radius = float(rng.uniform(0.115, 0.15))
        angles = phase + direction * np.linspace(0.0, 2.0 * math.pi, sides + 1)
        vertices = np.stack(
            [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)],
            axis=1,
        )
        xy = interpolate_polygon(vertices, count)
        params = {"shape": "square" if sides == 4 else "triangle", "radius_m": radius}
    elif "spiral" in family:
        radius_start = float(rng.uniform(0.13, 0.16))
        radius_end = float(rng.uniform(0.055, 0.075))
        progress = np.linspace(0.0, 1.0, count)
        theta = phase + direction * 3.0 * math.pi * progress
        radius = (1.0 - progress) * radius_start + progress * radius_end
        xy = np.stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)], axis=1)
        params = {
            "shape": "spiral",
            "turns": 1.5,
            "radius_start_m": radius_start,
            "radius_end_m": radius_end,
        }
    else:
        radius_x = float(rng.uniform(0.12, 0.16))
        radius_y = float(rng.uniform(0.07, 0.105))
        u = phase + direction * np.linspace(0.0, 2.0 * math.pi, count)
        xy = np.stack(
            [center[0] + radius_x * np.sin(u), center[1] + radius_y * np.sin(2.0 * u)],
            axis=1,
        )
        params = {"shape": "figure8", "radius_x_m": radius_x, "radius_y_m": radius_y}

    points = np.concatenate([xy, np.full((len(xy), 1), z, dtype=np.float64)], axis=1)
    params.update(
        {
            "center_xy_m": center,
            "z_m": z,
            "path_steps": len(points),
            "sequence_direction": "clockwise_or_reverse" if direction < 0.0 else "counterclockwise_or_forward",
            "phase_rad": phase,
        }
    )
    return points, params


def run_free_episode(
    recorder: LeRobotEpisodeRecorder,
    *,
    family: str,
    repetition: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    points, params = make_free_path(family, box_xyz=recorder.initial_box_xyz, rng=rng)
    grip_fn = gripper_profile(repetition % 4)
    initial_gripper = grip_fn(0, len(points))
    recorder.hold(4, gripper=initial_gripper, phase="initial_observation")
    recorder.move_to(
        points[0],
        steps=26,
        gripper=initial_gripper,
        phase="move_to_trajectory_start",
        max_action=0.17,
    )
    recorder.follow_points(
        points,
        gripper_fn=grip_fn,
        phase=family,
        gain=float(rng.uniform(2.8, 3.4)),
        max_action=float(rng.uniform(0.10, 0.14)),
    )
    recorder.hold(8, gripper=grip_fn(len(points) - 1, len(points)), phase="trajectory_settle")
    params["gripper_profile_mode"] = int(repetition % 4)
    return params


def approach_for_push(
    recorder: LeRobotEpisodeRecorder,
    direction_xy: np.ndarray,
    *,
    offset_m: float,
) -> np.ndarray:
    direction = demo.normalized(direction_xy)
    behind_xy = recorder.initial_box_xyz[:2] - direction * float(offset_m)
    recorder.move_to(
        (float(behind_xy[0]), float(behind_xy[1]), 1.04),
        steps=26,
        gripper=1.0,
        phase="approach_above",
        max_action=0.18,
    )
    recorder.move_to(
        (float(behind_xy[0]), float(behind_xy[1]), demo.CONTACT_Z),
        steps=22,
        gripper=1.0,
        phase="descend_behind_box",
        max_action=0.13,
    )
    return behind_xy


def finish_interaction(recorder: LeRobotEpisodeRecorder) -> None:
    recorder.hold(16, gripper=1.0, phase="observe_object_motion")
    eef = np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64)
    recorder.move_to(
        (float(eef[0]), float(eef[1]), 1.06),
        steps=16,
        gripper=1.0,
        phase="lift_clear",
        max_action=0.14,
    )
    recorder.hold(6, gripper=1.0, phase="final_settle")


def run_line_push(
    recorder: LeRobotEpisodeRecorder,
    *,
    direction: np.ndarray,
    slow: bool,
    rng: np.random.Generator,
    phase: str,
) -> dict[str, Any]:
    offset = float(rng.uniform(0.112, 0.122) if slow else rng.uniform(0.125, 0.145))
    behind_xy = approach_for_push(recorder, direction, offset_m=offset)
    start = np.asarray([behind_xy[0], behind_xy[1], demo.CONTACT_Z], dtype=np.float64)
    distance = float(rng.uniform(0.145, 0.18) if slow else rng.uniform(0.175, 0.215))
    end = start.copy()
    end[:2] += demo.normalized(direction) * distance
    steps = int(rng.integers(38, 46) if slow else rng.integers(30, 38))
    recorder.track_line(
        start,
        end,
        steps=steps,
        gripper=1.0,
        phase=phase,
        gain=float(rng.uniform(2.7, 3.2)),
        max_action=float(rng.uniform(0.065, 0.085) if slow else rng.uniform(0.105, 0.14)),
    )
    finish_interaction(recorder)
    return {"offset_m": offset, "track_distance_m": distance, "track_steps": steps}


def run_tap(
    recorder: LeRobotEpisodeRecorder,
    *,
    fast: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    offset = float(rng.uniform(0.15, 0.165) if fast else rng.uniform(0.135, 0.15))
    approach_for_push(recorder, direction, offset_m=offset)
    peak = float(rng.uniform(0.42, 0.50) if fast else rng.uniform(0.24, 0.34))
    hold_after_contact = int(rng.integers(2, 4) if fast else rng.integers(2, 5))
    event = recorder.contact_triggered_pulse(
        direction,
        peak=peak,
        line_xy=recorder.initial_box_xyz[:2],
        contact_z=demo.CONTACT_Z,
        max_precontact_steps=34 if fast else 38,
        hold_after_contact=hold_after_contact,
        phase="ram_fast" if fast else "tap_medium",
    )
    finish_interaction(recorder)
    return {
        "offset_m": offset,
        "peak_action": peak,
        "hold_after_contact": hold_after_contact,
        "contact_event": event,
    }


def run_directional_position_push(
    recorder: LeRobotEpisodeRecorder,
    *,
    angle_deg: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    angle = math.radians(float(angle_deg))
    direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    offset = float(rng.uniform(0.135, 0.155))
    behind_xy = approach_for_push(recorder, direction, offset_m=offset)
    start = np.asarray([behind_xy[0], behind_xy[1], demo.CONTACT_Z], dtype=np.float64)
    distance = float(rng.uniform(0.18, 0.22))
    end = start.copy()
    end[:2] += direction * distance
    steps = int(rng.integers(36, 45))
    recorder.track_line(
        start,
        end,
        steps=steps,
        gripper=1.0,
        phase=f"directional_push_{angle_deg:+.0f}deg",
        gain=float(rng.uniform(2.8, 3.3)),
        max_action=float(rng.uniform(0.10, 0.14)),
    )
    finish_interaction(recorder)
    return {"angle_deg": angle_deg, "offset_m": offset, "track_distance_m": distance, "track_steps": steps}


def run_curved_contact_sweep(
    recorder: LeRobotEpisodeRecorder,
    *,
    clockwise: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sign = -1.0 if clockwise else 1.0
    center = recorder.initial_box_xyz[:2].copy()
    start_angle = math.pi
    start_radius = float(rng.uniform(0.12, 0.135))
    start_xy = center + start_radius * np.asarray([math.cos(start_angle), math.sin(start_angle)])
    recorder.move_to(
        (float(start_xy[0]), float(start_xy[1]), 1.04),
        steps=26,
        gripper=1.0,
        phase="curve_approach_above",
        max_action=0.18,
    )
    recorder.move_to(
        (float(start_xy[0]), float(start_xy[1]), demo.CONTACT_Z),
        steps=22,
        gripper=1.0,
        phase="curve_descend",
        max_action=0.13,
    )
    radial_contact_radius = float(rng.uniform(0.015, 0.030))
    radial_target_xy = center + radial_contact_radius * np.asarray(
        [math.cos(start_angle), math.sin(start_angle)],
        dtype=np.float64,
    )
    recorder.track_line(
        np.asarray([start_xy[0], start_xy[1], demo.CONTACT_Z], dtype=np.float64),
        np.asarray([radial_target_xy[0], radial_target_xy[1], demo.CONTACT_Z], dtype=np.float64),
        steps=30,
        gripper=1.0,
        phase="curve_radial_contact_entry",
        gain=3.0,
        max_action=0.08,
    )
    recorder.move_to(
        np.asarray([radial_target_xy[0], radial_target_xy[1], demo.CONTACT_Z], dtype=np.float64),
        steps=24,
        gripper=1.0,
        phase="curve_establish_contact",
        gain=3.0,
        max_action=0.08,
    )
    count = int(rng.integers(46, 57))
    arc = float(rng.uniform(0.48 * math.pi, 0.70 * math.pi))
    progress = np.linspace(0.0, 1.0, count)
    theta = start_angle + sign * arc * progress
    contact_radius = float(rng.uniform(0.075, 0.088))
    gain = float(rng.uniform(2.8, 3.3))
    max_action = float(rng.uniform(0.10, 0.135))
    phase = "curved_contact_sweep_cw" if clockwise else "curved_contact_sweep_ccw"
    for angle in theta:
        box_xyz_now, _ = recorder.env.box_pose()
        target_xy = box_xyz_now[:2] + contact_radius * np.asarray(
            [math.cos(float(angle)), math.sin(float(angle))],
            dtype=np.float64,
        )
        eef = np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64)
        target = np.asarray([target_xy[0], target_xy[1], demo.CONTACT_Z], dtype=np.float64)
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(gain * (target - eef), -max_action, max_action)
        action[-1] = 1.0
        recorder.step(action, phase)
    finish_interaction(recorder)
    return {
        "clockwise": clockwise,
        "start_radius_m": start_radius,
        "radial_contact_radius_m": radial_contact_radius,
        "dynamic_arc_contact_radius_m": contact_radius,
        "arc_rad": arc,
        "path_steps": count,
    }


def run_top_press(recorder: LeRobotEpisodeRecorder, *, rng: np.random.Generator) -> dict[str, Any]:
    box = recorder.initial_box_xyz.copy()
    recorder.hold(4, gripper=1.0, phase="initial_observation")
    recorder.move_to(
        (float(box[0]), float(box[1]), 1.12),
        steps=30,
        gripper=1.0,
        phase="press_approach",
        max_action=0.18,
    )
    recorder.move_to(
        (float(box[0]), float(box[1]), float(box[2] + 0.055)),
        steps=26,
        gripper=1.0,
        phase="press_descend",
        max_action=0.12,
    )
    depth = float(rng.uniform(-0.030, -0.012))
    recorder.move_to(
        (float(box[0]), float(box[1]), float(box[2] + depth)),
        steps=24,
        gripper=1.0,
        phase="press_down",
        gain=2.5,
        max_action=0.08,
    )
    hold_steps = int(rng.integers(8, 15))
    recorder.hold(hold_steps, gripper=1.0, phase="hold_pressure")
    eef = np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64)
    recorder.move_to(
        (float(eef[0]), float(eef[1]), 1.12),
        steps=26,
        gripper=1.0,
        phase="release_press",
        max_action=0.15,
    )
    recorder.hold(8, gripper=1.0, phase="final_settle")
    return {"interaction_variant": "top_press", "target_depth_relative_to_box_center_m": depth, "hold_steps": hold_steps}


def run_grasp_lift_place(recorder: LeRobotEpisodeRecorder, *, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    box = recorder.initial_box_xyz.copy()
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    recorder.move_to(
        (float(box[0]), float(box[1]), 1.12),
        steps=33,
        gripper=-1.0,
        phase="grasp_approach",
        max_action=0.18,
    )
    recorder.move_to(
        (float(box[0]), float(box[1]), float(box[2] - 0.035)),
        steps=34,
        gripper=-1.0,
        phase="grasp_descend",
        max_action=0.12,
    )
    recorder.hold(28, gripper=1.0, phase="close_gripper")
    recorder.move_to(
        (float(box[0]), float(box[1]), 1.12),
        steps=34,
        gripper=1.0,
        phase="lift_box",
        max_action=0.16,
    )
    candidates = [
        (0.10, 0.10),
        (0.10, -0.08),
        (0.12, 0.00),
        (0.08, 0.12),
    ]
    base_offset = np.asarray(candidates[repetition % len(candidates)], dtype=np.float64)
    place_offset = base_offset + rng.uniform(-0.008, 0.008, size=2)
    place_xy = box[:2] + place_offset
    recorder.move_to(
        (float(place_xy[0]), float(place_xy[1]), 1.12),
        steps=30,
        gripper=1.0,
        phase="carry_box",
        max_action=0.14,
    )
    recorder.move_to(
        (float(place_xy[0]), float(place_xy[1]), 0.99),
        steps=27,
        gripper=1.0,
        phase="lower_box",
        max_action=0.10,
    )
    recorder.hold(22, gripper=-1.0, phase="open_gripper_drop")
    eef = np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64)
    recorder.move_to(
        (float(eef[0]), float(eef[1]), 1.12),
        steps=26,
        gripper=-1.0,
        phase="clear_after_drop",
        max_action=0.14,
    )
    recorder.hold(8, gripper=-1.0, phase="final_settle")
    return {"interaction_variant": "grasp_lift_place", "place_offset_xy_m": place_offset}


def run_contact_episode(
    recorder: LeRobotEpisodeRecorder,
    *,
    family: str,
    repetition: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if family not in {"vertical_press_or_grasp"}:
        recorder.hold(4, gripper=1.0, phase="initial_observation")
    if family == "straight_push_slow":
        return run_line_push(
            recorder,
            direction=np.asarray([1.0, 0.0]),
            slow=True,
            rng=rng,
            phase="straight_push_slow",
        )
    if family == "straight_push_medium":
        return run_line_push(
            recorder,
            direction=np.asarray([1.0, 0.0]),
            slow=False,
            rng=rng,
            phase="straight_push_medium",
        )
    if family == "tap_medium":
        return run_tap(recorder, fast=False, rng=rng)
    if family == "ram_fast":
        return run_tap(recorder, fast=True, rng=rng)
    angle_by_family = {
        "lateral_push_pos_y": 90.0,
        "lateral_push_neg_y": -90.0,
        "diagonal_push_pos30": 30.0,
        "diagonal_push_neg30": -30.0,
        "diagonal_push_pos60": 60.0,
        "diagonal_push_neg60": -60.0,
    }
    if family in angle_by_family:
        return run_directional_position_push(
            recorder,
            angle_deg=angle_by_family[family],
            rng=rng,
        )
    if family == "curved_contact_sweep_cw":
        return run_curved_contact_sweep(recorder, clockwise=True, rng=rng)
    if family == "curved_contact_sweep_ccw":
        return run_curved_contact_sweep(recorder, clockwise=False, rng=rng)
    if repetition % 2 == 0:
        return run_top_press(recorder, rng=rng)
    return run_grasp_lift_place(recorder, repetition=repetition, rng=rng)


def make_plan() -> list[dict[str, Any]]:
    plan = []
    for family_index, family in enumerate(FAMILIES):
        for repetition in range(EPISODES_PER_FAMILY):
            plan.append(
                {
                    "family": family,
                    "family_index": family_index,
                    "family_repetition": repetition,
                    "parameter_seed": BASE_SEED + family_index * 100 + repetition,
                }
            )
    rng = np.random.default_rng(BASE_SEED)
    rng.shuffle(plan)
    return plan


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    base.patch_lerobot_video_crf(demo.VIDEO_CRF)
    dataset = base.create_dataset(
        output,
        repo_id="libero_mu0100_various_actions_200eps_hai_machine",
    )
    bddl_file = demo.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="various_actions_mu0100_hidden",
        init_xy=demo.INIT_XY,
        target_xy=demo.TARGET_XY,
        init_half_size=0.002,
        target_radius=0.025,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )
    case = demo.build_demo_case(bddl_file)
    plan = make_plan()
    if len(plan) != EXPECTED_EPISODES or EXPECTED_EPISODES != 200:
        raise RuntimeError(f"Expected exactly 200 planned episodes, got {len(plan)}")

    episode_rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_mu0100_various_actions_200eps_lerobot_2026-07-15_hai-machine",
        "purpose": "action-conditioned world-model training with diverse commanded EEF trajectories and object interactions",
        "episode_count_expected": EXPECTED_EPISODES,
        "friction_mu": demo.FRICTION_MU,
        "target_visible": False,
        "camera_resolution": demo.CAMERA_RESOLUTION,
        "fps": demo.FPS,
        "video_crf": demo.VIDEO_CRF,
        "controller_scale": demo.CONTROLLER_SCALE,
        "sim_seed_fixed_for_same_scene": SIM_SEED,
        "action_names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"],
        "rotation_action_policy": "dax=day=daz=0; this dataset isolates translation and gripper effects",
        "alignment": "Each action_t is stored with observation_t before env.step(action_t); a final terminal observation is appended so every effective action has observation_tp1.",
        "task_prompt_policy": "one generic prompt for every episode; action-family labels are metadata only to prevent text leakage",
        "families": FAMILIES,
        "episodes_per_family": EPISODES_PER_FAMILY,
        "episodes": episode_rows,
    }
    manifest = {
        "created_at": metadata["created_at"],
        "output": str(output),
        "expected_episodes": EXPECTED_EPISODES,
        "friction_mu": demo.FRICTION_MU,
        "families": FAMILIES,
        "episodes_per_family": EPISODES_PER_FAMILY,
        "episodes": episode_rows,
    }

    def autosave() -> None:
        write_json(output / "collection_manifest.json", manifest)
        base.write_dataset_metadata(
            output,
            base.to_jsonable(metadata),
            base.to_jsonable(episode_rows),
        )

    autosave()
    for collection_index, item in enumerate(plan):
        family = str(item["family"])
        repetition = int(item["family_repetition"])
        parameter_seed = int(item["parameter_seed"])
        rng = np.random.default_rng(parameter_seed)
        env = demo.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=SIM_SEED)
        try:
            recorder = LeRobotEpisodeRecorder(env=env, dataset=dataset)
            if family in FREE_FAMILIES:
                parameters = run_free_episode(
                    recorder,
                    family=family,
                    repetition=repetition,
                    rng=rng,
                )
                interaction_group = "free_space_trajectory"
            else:
                parameters = run_contact_episode(
                    recorder,
                    family=family,
                    repetition=repetition,
                    rng=rng,
                )
                interaction_group = "object_interaction"
            recorder.add_terminal_observation()
            episode_index = int(dataset.meta.total_episodes)
            summary = recorder.summary()
            interaction_observed = bool(
                int(summary["robot_box_contact_steps"]) > 0
                or float(summary["final_box_displacement_m"]) > 0.001
                or float(summary["max_box_planar_speed_mps"]) > 0.03
                or int(summary["robosuite_grasping_steps"]) > 0
            )
            if interaction_group == "object_interaction" and not interaction_observed:
                raise RuntimeError(
                    f"Object-interaction episode has no observable interaction: family={family} "
                    f"repetition={repetition} collection_index={collection_index} "
                    f"contact={summary['robot_box_contact_steps']} "
                    f"displacement={summary['final_box_displacement_m']:.6f} "
                    f"max_box_speed={summary['max_box_planar_speed_mps']:.6f}"
                )
            dataset.save_episode()
        finally:
            env.close()

        row = {
            "episode_index": episode_index,
            "collection_index": collection_index,
            "family": family,
            "family_repetition": repetition,
            "interaction_group": interaction_group,
            "parameter_seed": parameter_seed,
            "sim_seed": SIM_SEED,
            "friction_mu": demo.FRICTION_MU,
            "target_visible": False,
            "parameters": parameters,
            "metrics": summary,
        }
        episode_rows.append(row)
        print(
            f"[{collection_index + 1:03d}/{EXPECTED_EPISODES:03d}] ep={episode_index:03d} "
            f"family={family} frames={summary['frames_in_lerobot_episode']} "
            f"box={summary['final_box_displacement_m'] * 100.0:.1f}cm "
            f"contact={summary['robot_box_contact_steps']} grasp={summary['robosuite_grasping_steps']}",
            flush=True,
        )
        autosave()

    summary = {
        "completed_at": dt.datetime.now().isoformat(),
        "output": str(output),
        "episode_count": len(episode_rows),
        "expected_episode_count": EXPECTED_EPISODES,
        "count_by_family": dict(Counter(row["family"] for row in episode_rows)),
        "count_by_interaction_group": dict(Counter(row["interaction_group"] for row in episode_rows)),
        "friction_mu": demo.FRICTION_MU,
        "target_visible": False,
        "total_lerobot_frames": int(sum(row["metrics"]["frames_in_lerobot_episode"] for row in episode_rows)),
    }
    write_json(output / "collection_summary.json", summary)
    autosave()
    print(json.dumps(base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

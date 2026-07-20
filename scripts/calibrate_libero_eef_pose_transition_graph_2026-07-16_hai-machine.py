#!/usr/bin/env python3
"""Collect a LeRobot calibration set for safe 6D EEF pose transitions.

This is deliberately separate from the formal workspace-rich dataset.  Every
attempt, including a failed reachability attempt, is recorded as a LeRobot
episode.  The emitted JSON graph is the allow-list for the later formal
collector.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
RICH_SCRIPT = HERE / "collect_libero_workspace_rich_eef_300eps_lerobot_2026-07-16_hai-machine.py"
DEFAULT_OUTPUT = (
    HERE.parent
    / "data/various_actions/calibration/"
    "libero_eef_pose_transition_graph_lerobot_2026-07-16_hai-machine"
)
DEFAULT_GRAPH = DEFAULT_OUTPUT.parent / "eef_pose_transition_graph_2026-07-16_hai-machine.json"

POSITION_TOLERANCE_M = 0.020
RETURN_POSITION_TOLERANCE_M = 0.015
ORIENTATION_TOLERANCE_DEG = 6.0
MIN_JOINT_LIMIT_MARGIN_RAD = 0.025
MAX_REALIZED_EEF_STEP_M = 0.030


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rich = load_module(RICH_SCRIPT, "workspace_rich_calibration_base")


def rotation_xyz_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([roll, pitch, yaw])
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


def orientation_distance_deg(target: np.ndarray, current: np.ndarray) -> float:
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(math.acos(cosine)))


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


def robot_from(recorder: Any) -> Any:
    return recorder.env.inner_env.robots[0]


def current_orientation(recorder: Any) -> np.ndarray:
    return np.asarray(robot_from(recorder).controller.ee_ori_mat, dtype=np.float64).copy()


@dataclass
class SafetyTrace:
    min_joint_limit_margin_rad: float = math.inf
    max_joint_speed_rad_s: float = 0.0
    max_realized_eef_step_m: float = 0.0
    sample_count: int = 0
    previous_eef: np.ndarray | None = field(default=None, repr=False)

    def sample(self, recorder: Any) -> None:
        robot = robot_from(recorder)
        qpos = np.asarray(robot._joint_positions, dtype=np.float64)
        qvel = np.asarray(robot._joint_velocities, dtype=np.float64)
        joint_ranges = np.asarray(
            robot.sim.model.jnt_range[np.asarray(robot._ref_joint_indexes, dtype=np.int64)],
            dtype=np.float64,
        )
        finite = np.isfinite(joint_ranges).all(axis=1) & (joint_ranges[:, 1] > joint_ranges[:, 0])
        if np.any(finite):
            margins = np.minimum(
                qpos[finite] - joint_ranges[finite, 0],
                joint_ranges[finite, 1] - qpos[finite],
            )
            self.min_joint_limit_margin_rad = min(
                self.min_joint_limit_margin_rad, float(np.min(margins))
            )
        self.max_joint_speed_rad_s = max(self.max_joint_speed_rad_s, float(np.max(np.abs(qvel))))
        eef = rich.current_eef(recorder)
        if self.previous_eef is not None:
            self.max_realized_eef_step_m = max(
                self.max_realized_eef_step_m, float(np.linalg.norm(eef - self.previous_eef))
            )
        self.previous_eef = eef.copy()
        self.sample_count += 1

    def as_dict(self) -> dict[str, Any]:
        margin = self.min_joint_limit_margin_rad
        return {
            "sample_count": self.sample_count,
            "min_joint_limit_margin_rad": None if not math.isfinite(margin) else float(margin),
            "max_joint_speed_rad_s": float(self.max_joint_speed_rad_s),
            "max_realized_eef_step_m": float(self.max_realized_eef_step_m),
        }


def move_pose(
    recorder: Any,
    target_xyz: np.ndarray,
    target_orientation: np.ndarray,
    *,
    phase: str,
    trace: SafetyTrace,
    max_steps: int = 90,
    gripper: float = -1.0,
) -> dict[str, Any]:
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    start_xyz = rich.current_eef(recorder)
    start_orientation = current_orientation(recorder)
    stable_steps = 0
    steps_used = 0
    trace.sample(recorder)

    for step in range(max_steps):
        actual_xyz = rich.current_eef(recorder)
        actual_orientation = current_orientation(recorder)
        position_delta = target_xyz - actual_xyz
        rotation_delta = orientation_error(target_orientation, actual_orientation)
        position_error_m = float(np.linalg.norm(position_delta))
        orientation_error_deg = orientation_distance_deg(target_orientation, actual_orientation)

        if position_error_m <= 0.008 and orientation_error_deg <= 2.5:
            stable_steps += 1
        else:
            stable_steps = 0
        if stable_steps >= 4:
            break

        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(position_delta * 3.2, -0.16, 0.16)
        action[3:6] = np.clip(rotation_delta / 0.5, -0.12, 0.12)
        action[6] = gripper
        recorder.step(action, phase=phase)
        trace.sample(recorder)
        steps_used = step + 1

    for _ in range(4):
        recorder.step(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float32), phase=f"{phase}_settle")
        trace.sample(recorder)

    final_xyz = rich.current_eef(recorder)
    final_orientation = current_orientation(recorder)
    return {
        "phase": phase,
        "steps": steps_used + 4,
        "start_xyz_m": start_xyz.tolist(),
        "target_xyz_m": target_xyz.tolist(),
        "achieved_xyz_m": final_xyz.tolist(),
        "position_error_m": float(np.linalg.norm(target_xyz - final_xyz)),
        "orientation_error_deg": orientation_distance_deg(target_orientation, final_orientation),
        "requested_translation_m": float(np.linalg.norm(target_xyz - start_xyz)),
        "realized_translation_m": float(np.linalg.norm(final_xyz - start_xyz)),
        "start_orientation_wxyz": matrix_to_quaternion_wxyz(start_orientation),
        "target_orientation_wxyz": matrix_to_quaternion_wxyz(target_orientation),
        "achieved_orientation_wxyz": matrix_to_quaternion_wxyz(final_orientation),
    }


ORIENTATION_PRESETS: list[dict[str, Any]] = [
    {"name": "tool_pitch_forward_15", "rpy_deg": [0.0, 15.0, 0.0], "use": "push_or_scrape"},
    {"name": "tool_pitch_forward_25", "rpy_deg": [0.0, 25.0, 0.0], "use": "push_or_scrape"},
    {"name": "tool_pitch_backward_15", "rpy_deg": [0.0, -15.0, 0.0], "use": "pull_or_retract"},
    {"name": "tool_roll_left_20", "rpy_deg": [20.0, 0.0, 0.0], "use": "stir_or_sweep"},
    {"name": "tool_roll_right_20", "rpy_deg": [-20.0, 0.0, 0.0], "use": "stir_or_sweep"},
    {"name": "tool_roll_left_35", "rpy_deg": [35.0, 0.0, 0.0], "use": "tilt_or_pour"},
    {"name": "tool_roll_right_35", "rpy_deg": [-35.0, 0.0, 0.0], "use": "tilt_or_pour"},
    {"name": "tool_yaw_left_30", "rpy_deg": [0.0, 0.0, 30.0], "use": "shake_or_wipe"},
    {"name": "tool_yaw_right_30", "rpy_deg": [0.0, 0.0, -30.0], "use": "shake_or_wipe"},
    {"name": "tool_oblique_20", "rpy_deg": [20.0, 20.0, 20.0], "use": "oblique_hammer_or_poke"},
]


def make_position_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    xs = [-0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.14]
    ys = [-0.24, -0.12, 0.00, 0.12, 0.24]
    zs = [1.00, 1.12, 1.24]
    for x in xs:
        for y in ys:
            for z in zs:
                cases.append(
                    {
                        "type": "position_roundtrip",
                        "id": f"position_x{x:+.2f}_y{y:+.2f}_z{z:.2f}",
                        "target_xyz_m": [x, y, z],
                        "orientation": {"name": "reset_upright", "rpy_deg": [0.0, 0.0, 0.0]},
                    }
                )
    return cases


def make_orientation_cases() -> list[dict[str, Any]]:
    anchors = [
        [-0.30, -0.16, 1.10],
        [-0.30, 0.16, 1.10],
        [-0.16, -0.18, 1.08],
        [-0.16, 0.18, 1.08],
        [0.00, -0.14, 1.12],
        [0.00, 0.14, 1.12],
    ]
    cases: list[dict[str, Any]] = []
    for preset in ORIENTATION_PRESETS:
        for anchor_index, anchor in enumerate(anchors):
            cases.append(
                {
                    "type": "orientation_roundtrip",
                    "id": f"{preset['name']}_anchor_{anchor_index:02d}",
                    "target_xyz_m": anchor,
                    "orientation": preset,
                }
            )
    return cases


def make_route_cases() -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260716)
    pool = np.asarray(
        [
            [-0.36, -0.20, 1.06], [-0.36, 0.00, 1.18], [-0.36, 0.20, 1.10],
            [-0.22, -0.22, 1.20], [-0.22, 0.00, 1.04], [-0.22, 0.22, 1.16],
            [-0.08, -0.20, 1.10], [-0.08, 0.00, 1.22], [-0.08, 0.20, 1.06],
            [0.06, -0.18, 1.16], [0.06, 0.00, 1.06], [0.06, 0.18, 1.20],
        ],
        dtype=np.float64,
    )
    cases: list[dict[str, Any]] = []
    for route_index in range(20):
        indices = rng.choice(len(pool), size=4, replace=False)
        cases.append(
            {
                "type": "multi_node_route",
                "id": f"route_{route_index:02d}",
                "waypoints_xyz_m": pool[indices].tolist(),
                "orientation": {"name": "reset_upright", "rpy_deg": [0.0, 0.0, 0.0]},
            }
        )
    return cases


CALIBRATION_CASES = make_position_cases() + make_orientation_cases() + make_route_cases()
RESULTS: list[dict[str, Any]] = []


def segment_pass(segment: dict[str, Any]) -> bool:
    return bool(
        segment["position_error_m"] <= POSITION_TOLERANCE_M
        and segment["orientation_error_deg"] <= ORIENTATION_TOLERANCE_DEG
    )


def safety_pass(trace: SafetyTrace) -> bool:
    margin = trace.min_joint_limit_margin_rad
    return bool(
        math.isfinite(margin)
        and margin >= MIN_JOINT_LIMIT_MARGIN_RAD
        and trace.max_realized_eef_step_m <= MAX_REALIZED_EEF_STEP_M
    )


def run_calibration_case(recorder: Any, case: dict[str, Any]) -> dict[str, Any]:
    trace = SafetyTrace()
    home_xyz = rich.current_eef(recorder)
    home_orientation = current_orientation(recorder)
    segments: list[dict[str, Any]] = []

    if case["type"] in {"position_roundtrip", "orientation_roundtrip"}:
        target_xyz = np.asarray(case["target_xyz_m"], dtype=np.float64)
        rpy = case["orientation"]["rpy_deg"]
        target_orientation = rotation_xyz_deg(*rpy) @ home_orientation
        segments.append(
            move_pose(
                recorder,
                target_xyz,
                target_orientation,
                phase="calibration_outbound",
                trace=trace,
            )
        )
        segments.append(
            move_pose(
                recorder,
                home_xyz,
                home_orientation,
                phase="calibration_return",
                trace=trace,
            )
        )
    else:
        for waypoint_index, waypoint in enumerate(case["waypoints_xyz_m"]):
            segments.append(
                move_pose(
                    recorder,
                    np.asarray(waypoint, dtype=np.float64),
                    home_orientation,
                    phase=f"calibration_route_{waypoint_index:02d}",
                    trace=trace,
                )
            )
        segments.append(
            move_pose(
                recorder,
                home_xyz,
                home_orientation,
                phase="calibration_route_return",
                trace=trace,
            )
        )

    return_segment = segments[-1]
    all_segments_pass = all(segment_pass(segment) for segment in segments)
    return_pass = bool(
        return_segment["position_error_m"] <= RETURN_POSITION_TOLERANCE_M
        and return_segment["orientation_error_deg"] <= ORIENTATION_TOLERANCE_DEG
    )
    result = {
        "calibration_case": case,
        "home_xyz_m": home_xyz.tolist(),
        "home_orientation_wxyz": matrix_to_quaternion_wxyz(home_orientation),
        "segments": segments,
        "safety": trace.as_dict(),
        "all_segments_pass": all_segments_pass,
        "return_pass": return_pass,
        "roundtrip_pass": bool(all_segments_pass and return_pass and safety_pass(trace)),
        "thresholds": {
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "return_position_tolerance_m": RETURN_POSITION_TOLERANCE_M,
            "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
            "min_joint_limit_margin_rad": MIN_JOINT_LIMIT_MARGIN_RAD,
            "max_realized_eef_step_m": MAX_REALIZED_EEF_STEP_M,
        },
    }
    RESULTS.append(result)
    return result


def make_plan() -> list[dict[str, Any]]:
    return [
        {
            "family": "eef_pose_transition_calibration",
            "family_index": 0,
            "repetition": index,
            "family_repetition": index,
            "behavior_group": "eef_pose_calibration",
            "parameter_seed": 20260716 + index,
        }
        for index in range(len(CALIBRATION_CASES))
    ]


def run_family(recorder: Any, family: str, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    del rng
    if family != "eef_pose_transition_calibration":
        raise ValueError(f"Unexpected family: {family}")
    return run_calibration_case(recorder, CALIBRATION_CASES[repetition])


def prompts(family: str) -> list[str]:
    del family
    return [
        "Calibrate a safe end-effector pose and return to the reset pose.",
        "Reach the requested tool pose, hold it, and safely return.",
        "Verify this position and orientation through a measured round trip.",
        "Execute and validate a safe six-dimensional end-effector transition.",
    ]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_graph(path: Path) -> None:
    passed = [result for result in RESULTS if result["roundtrip_pass"]]
    graph = {
        "schema": "libero_eef_pose_transition_graph_v1",
        "created_for": "hai-machine",
        "controller": "OSC_POSE",
        "action_semantics": "relative EEF translation and axis-angle rotation at native 20 Hz",
        "quaternion_order": "wxyz",
        "attempt_count": len(RESULTS),
        "pass_count": len(passed),
        "fail_count": len(RESULTS) - len(passed),
        "thresholds": {
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "return_position_tolerance_m": RETURN_POSITION_TOLERANCE_M,
            "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
            "min_joint_limit_margin_rad": MIN_JOINT_LIMIT_MARGIN_RAD,
            "max_realized_eef_step_m": MAX_REALIZED_EEF_STEP_M,
        },
        "passed_case_ids": [item["calibration_case"]["id"] for item in passed],
        "results": RESULTS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(graph), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph-output", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rich.FAMILY_COUNTS = {"eef_pose_transition_calibration": len(CALIBRATION_CASES)}
    rich.EXPECTED_EPISODES = len(CALIBRATION_CASES)
    rich.GROUP_BY_FAMILY = {"eef_pose_transition_calibration": "eef_pose_calibration"}
    rich.TASK_PROMPTS = {"eef_pose_transition_calibration": prompts("")}
    rich.make_plan = make_plan
    rich.run_family = run_family
    rich.episode_task_prompts = prompts
    rich.observable_interaction = lambda recorder: True

    forwarded = [str(RICH_SCRIPT), "--output", str(args.output)]
    if args.overwrite:
        forwarded.append("--overwrite")
    original_argv = sys.argv
    try:
        sys.argv = forwarded
        rich.main()
    finally:
        sys.argv = original_argv
        write_graph(args.graph_output)


if __name__ == "__main__":
    main()

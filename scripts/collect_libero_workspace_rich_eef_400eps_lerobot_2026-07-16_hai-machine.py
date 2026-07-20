#!/usr/bin/env python3
"""Collect the formal 400-episode workspace-rich OSC_POSE LeRobot dataset.

The collector consumes the measured EEF transition graph produced on this
machine.  The final 100 episodes are long imagined-tool action sequences: the
robot performs task-shaped motions with validated lateral EEF orientations,
but no visible tool asset is inserted into the scene.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CALIBRATION_SCRIPT = HERE / "calibrate_libero_eef_pose_transition_graph_2026-07-16_hai-machine.py"
DEFAULT_GRAPH = (
    HERE.parent
    / "data/various_actions/calibration/eef_pose_transition_graph_2026-07-16_hai-machine.json"
)
DEFAULT_OUTPUT = (
    HERE.parent
    / "data/various_actions/"
    "libero_mu0100_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine"
)
EXPECTED_EPISODES = 400
FORMAL_WARNING_JOINT_MARGIN_RAD = 0.025
FORMAL_HARD_JOINT_MARGIN_RAD = -0.010


class RetryableEpisodeError(RuntimeError):
    """A rollout-quality failure that must discard and repeat this setting."""


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration = load_module(CALIBRATION_SCRIPT, "eef_pose_calibration_for_formal_collection")
rich = calibration.rich


FAMILY_COUNTS: dict[str, int] = {
    "validated_multi_node_route": 60,
    "validated_pose_star_tour": 40,
    "short_poke": 20,
    "impulse_tap": 20,
    "strong_ram": 20,
    "position_push": 20,
    "contact_sweep_press": 20,
    "direct_grasp_far_place": 35,
    "push_then_grasp_far_place": 35,
    "grasp_carry_waypoint_place": 30,
    "imagined_tool_micro_sequence": 100,
}

GROUP_BY_FAMILY = {
    "validated_multi_node_route": "workspace_exploration",
    "validated_pose_star_tour": "workspace_exploration",
    "short_poke": "object_contact",
    "impulse_tap": "object_contact",
    "strong_ram": "object_contact",
    "position_push": "object_contact",
    "contact_sweep_press": "object_contact",
    "direct_grasp_far_place": "grasp_relocation",
    "push_then_grasp_far_place": "grasp_relocation",
    "grasp_carry_waypoint_place": "grasp_relocation",
    "imagined_tool_micro_sequence": "imagined_tool_micro_actions",
}

BASE_TASK_PROMPTS: dict[str, list[str]] = {
    "validated_multi_node_route": [
        "Traverse a broad sequence of verified end-effector waypoints.",
        "Move safely through several distant poses across the workspace.",
        "Follow a long calibrated route over the table and return.",
        "Explore the reachable workspace through verified pose transitions.",
    ],
    "validated_pose_star_tour": [
        "Visit several verified poses across the workspace and return between visits.",
        "Reach multiple distant table locations through the safe home pose.",
        "Perform a broad star-shaped end-effector workspace tour.",
        "Move accurately among several calibrated workspace targets.",
    ],
    "short_poke": [
        "Approach the block and give it a short controlled poke.",
        "Tap the block briefly with a small forward impulse.",
        "Make a short dynamic contact with the block.",
        "Poke the block once and stop cleanly.",
    ],
    "impulse_tap": [
        "Strike the block once with a medium impulse.",
        "Use one quick controlled tap to move the block.",
        "Deliver a medium-speed single collision to the block.",
        "Approach, tap the block dynamically, and stop.",
    ],
    "strong_ram": [
        "Ram the block once with a strong controlled push.",
        "Use a high-speed single impact to send the block forward.",
        "Deliver one strong dynamic strike without repeated contact.",
        "Approach at speed and hit the block with a strong impulse.",
    ],
    "position_push": [
        "Push the block steadily using position control.",
        "Maintain contact and move the block with a controlled push.",
        "Execute a smooth position-controlled block push.",
        "Approach the block and push it steadily along the table.",
    ],
    "contact_sweep_press": [
        "Sweep or press the block with a controlled contact trajectory.",
        "Perform a curved contact motion against the block.",
        "Use a deliberate sweeping or pressing interaction on the block.",
        "Contact the block and follow a shaped manipulation path.",
    ],
    "direct_grasp_far_place": [
        "Grasp the block, carry it far across the table, and place it down.",
        "Pick up the block and relocate it to a distant table position.",
        "Lift the block, transport it across the workspace, and release it.",
        "Execute a direct grasp followed by a long carry and placement.",
    ],
    "push_then_grasp_far_place": [
        "Push the block, then grasp it and place it far away.",
        "First move the block by pushing, then pick it up and relocate it.",
        "Combine a block push with a subsequent grasp and distant placement.",
        "Push, re-approach, grasp, carry, and release the block.",
    ],
    "grasp_carry_waypoint_place": [
        "Grasp the block and carry it through several waypoints before placing it.",
        "Pick up the block, follow a long carry path, and release it.",
        "Transport the grasped block through multiple poses to a distant target.",
        "Lift, carry through the workspace, and place the block down.",
    ],
}

MICRO_PRIMITIVES = [
    "stir_circle",
    "stir_figure_eight",
    "shake_lateral",
    "shake_vertical",
    "hammer_tap",
    "wipe_raster",
    "scrape_stroke",
    "twist_wrist",
    "tilt_pour",
    "press_poke",
    "scoop_lift",
    "pinch_reposition",
]

PRIMITIVE_TO_USE = {
    "stir_circle": "stir_or_sweep",
    "stir_figure_eight": "stir_or_sweep",
    "shake_lateral": "shake_or_wipe",
    "shake_vertical": "shake_or_wipe",
    "hammer_tap": "oblique_hammer_or_poke",
    "wipe_raster": "shake_or_wipe",
    "scrape_stroke": "push_or_scrape",
    "twist_wrist": "shake_or_wipe",
    "tilt_pour": "tilt_or_pour",
    "press_poke": "oblique_hammer_or_poke",
    "scoop_lift": "push_or_scrape",
    "pinch_reposition": "reset_upright",
}


def build_balanced_micro_schedule() -> list[list[str]]:
    targets = {
        primitive: 34 if index < 4 else 33
        for index, primitive in enumerate(MICRO_PRIMITIVES)
    }
    remaining = dict(targets)
    rng = np.random.default_rng(910_000)
    schedule: list[list[str]] = []
    for _ in range(FAMILY_COUNTS["imagined_tool_micro_sequence"]):
        tie_break = {primitive: float(rng.random()) for primitive in MICRO_PRIMITIVES}
        ranked = sorted(
            (primitive for primitive in MICRO_PRIMITIVES if remaining[primitive] > 0),
            key=lambda primitive: (-remaining[primitive], tie_break[primitive]),
        )
        sequence = ranked[:4]
        rng.shuffle(sequence)
        for primitive in sequence:
            remaining[primitive] -= 1
        schedule.append(sequence)
    if any(remaining.values()) or any(len(set(sequence)) != 4 for sequence in schedule):
        raise RuntimeError(f"Invalid balanced micro-action schedule: {remaining}")
    return schedule


MICRO_SCHEDULE = build_balanced_micro_schedule()


def micro_sequence_for(repetition: int) -> list[str]:
    return list(MICRO_SCHEDULE[repetition])


TASK_PROMPTS = dict(BASE_TASK_PROMPTS)
TASK_PROMPTS["imagined_tool_micro_sequence"] = [
    "Perform four distinct imagined-tool micro-actions at verified workspace poses.",
    "Move among safe poses and demonstrate four common manipulation primitives.",
    "Execute a varied sequence of four task-shaped end-effector motions.",
    "Demonstrate four calibrated micro-actions without a visible physical tool.",
]


GRAPH_DATA: dict[str, Any] = {}
POSITION_CASES: list[dict[str, Any]] = []
ORIENTATION_CASES: list[dict[str, Any]] = []
ROUTE_CASES: list[dict[str, Any]] = []


def load_validated_graph(path: Path) -> None:
    global GRAPH_DATA, POSITION_CASES, ORIENTATION_CASES, ROUTE_CASES
    GRAPH_DATA = json.loads(path.read_text(encoding="utf-8"))
    if GRAPH_DATA.get("attempt_count") != 185:
        raise RuntimeError(f"Incomplete calibration graph: {path}")
    passed = [item for item in GRAPH_DATA["results"] if item["roundtrip_pass"]]
    POSITION_CASES = [
        item["calibration_case"]
        for item in passed
        if item["calibration_case"]["type"] == "position_roundtrip"
    ]
    ORIENTATION_CASES = [
        item["calibration_case"]
        for item in passed
        if item["calibration_case"]["type"] == "orientation_roundtrip"
    ]
    ROUTE_CASES = [
        item["calibration_case"]
        for item in passed
        if item["calibration_case"]["type"] == "multi_node_route"
    ]
    if len(POSITION_CASES) < 60 or len(ORIENTATION_CASES) < 50 or len(ROUTE_CASES) != 20:
        raise RuntimeError(
            "Calibration graph does not provide enough safe poses: "
            f"position={len(POSITION_CASES)} orientation={len(ORIENTATION_CASES)} "
            f"routes={len(ROUTE_CASES)}"
        )


def pose_for_case(case: dict[str, Any], home_orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_xyz = np.asarray(case["target_xyz_m"], dtype=np.float64)
    rpy = case["orientation"]["rpy_deg"]
    target_orientation = calibration.rotation_xyz_deg(*rpy) @ home_orientation
    return target_xyz, target_orientation


def checked_move(
    recorder: Any,
    target_xyz: np.ndarray,
    target_orientation: np.ndarray,
    *,
    phase: str,
    trace: Any,
    returning_home: bool = False,
) -> dict[str, Any]:
    segment = calibration.move_pose(
        recorder,
        target_xyz,
        target_orientation,
        phase=phase,
        trace=trace,
    )
    position_limit = (
        calibration.RETURN_POSITION_TOLERANCE_M
        if returning_home
        else calibration.POSITION_TOLERANCE_M
    )
    if (
        segment["position_error_m"] > position_limit
        or segment["orientation_error_deg"] > calibration.ORIENTATION_TOLERANCE_DEG
    ):
        raise RetryableEpisodeError(
            f"EEF runtime gate failed in {phase}: "
            f"position={segment['position_error_m']:.4f}m "
            f"orientation={segment['orientation_error_deg']:.2f}deg"
        )
    return segment


def require_safe_trace(trace: Any, family: str) -> None:
    metrics = trace.as_dict()
    margin = metrics["min_joint_limit_margin_rad"]
    if (
        margin is None
        or margin < FORMAL_HARD_JOINT_MARGIN_RAD
        or metrics["max_realized_eef_step_m"] > calibration.MAX_REALIZED_EEF_STEP_M
    ):
        raise RetryableEpisodeError(f"EEF safety gate failed for {family}: {metrics}")
    if margin < FORMAL_WARNING_JOINT_MARGIN_RAD:
        print(
            f"[safety-warning] family={family} "
            f"joint_margin={margin:.6f}rad metrics={metrics}",
            flush=True,
        )


def run_validated_route(
    recorder: Any,
    repetition: int,
    rng: np.random.Generator,
    *,
    retry_attempt: int = 0,
) -> dict[str, Any]:
    trace = calibration.SafetyTrace()
    home_xyz = rich.current_eef(recorder)
    home_orientation = calibration.current_orientation(recorder)
    route = (
        ROUTE_CASES[repetition % len(ROUTE_CASES)]
        if retry_attempt == 0
        else ROUTE_CASES[int(rng.integers(0, len(ROUTE_CASES)))]
    )
    segments: list[dict[str, Any]] = []
    for waypoint_index, waypoint in enumerate(route["waypoints_xyz_m"]):
        segments.append(
            checked_move(
                recorder,
                np.asarray(waypoint, dtype=np.float64),
                home_orientation,
                phase=f"validated_route_{waypoint_index:02d}",
                trace=trace,
            )
        )

    variant = repetition // len(ROUTE_CASES)
    if variant == 1:
        case = POSITION_CASES[(repetition * 7) % len(POSITION_CASES)]
        target_xyz, target_orientation = pose_for_case(case, home_orientation)
        segments.append(
            checked_move(
                recorder,
                home_xyz,
                home_orientation,
                phase="validated_route_mid_home",
                trace=trace,
                returning_home=True,
            )
        )
        segments.append(
            checked_move(
                recorder,
                target_xyz,
                target_orientation,
                phase="validated_route_extra_position",
                trace=trace,
            )
        )
    elif variant == 2:
        case = ORIENTATION_CASES[(repetition * 11) % len(ORIENTATION_CASES)]
        target_xyz, target_orientation = pose_for_case(case, home_orientation)
        segments.append(
            checked_move(
                recorder,
                home_xyz,
                home_orientation,
                phase="validated_route_mid_home",
                trace=trace,
                returning_home=True,
            )
        )
        segments.append(
            checked_move(
                recorder,
                target_xyz,
                target_orientation,
                phase="validated_route_extra_tool_pose",
                trace=trace,
            )
        )

    segments.append(
        checked_move(
            recorder,
            home_xyz,
            home_orientation,
            phase="validated_route_return_home",
            trace=trace,
            returning_home=True,
        )
    )
    require_safe_trace(trace, "validated_multi_node_route")
    return {
        "graph_case_id": route["id"],
        "route_variant": variant,
        "segments": segments,
        "safety": trace.as_dict(),
        "runtime_gate_pass": True,
    }


def run_pose_star(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    trace = calibration.SafetyTrace()
    home_xyz = rich.current_eef(recorder)
    home_orientation = calibration.current_orientation(recorder)
    position_indices = rng.choice(len(POSITION_CASES), size=3, replace=False)
    orientation_index = int(rng.integers(0, len(ORIENTATION_CASES)))
    cases = [POSITION_CASES[int(index)] for index in position_indices]
    cases.insert(repetition % 4, ORIENTATION_CASES[orientation_index])
    segments: list[dict[str, Any]] = []

    for visit_index, case in enumerate(cases):
        target_xyz, target_orientation = pose_for_case(case, home_orientation)
        segments.append(
            checked_move(
                recorder,
                target_xyz,
                target_orientation,
                phase=f"pose_star_visit_{visit_index:02d}",
                trace=trace,
            )
        )
        segments.append(
            checked_move(
                recorder,
                home_xyz,
                home_orientation,
                phase=f"pose_star_home_{visit_index:02d}",
                trace=trace,
                returning_home=True,
            )
        )

    require_safe_trace(trace, "validated_pose_star_tour")
    return {
        "visited_case_ids": [case["id"] for case in cases],
        "segments": segments,
        "safety": trace.as_dict(),
        "runtime_gate_pass": True,
    }


def primitive_path(
    primitive: str,
    anchor: np.ndarray,
    base_orientation: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    positions: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    grippers: list[float] = []

    def append(position: np.ndarray, orientation: np.ndarray = base_orientation, gripper: float = -1.0) -> None:
        clipped = np.asarray(position, dtype=np.float64).copy()
        clipped[0] = np.clip(clipped[0], -0.43, 0.12)
        clipped[1] = np.clip(clipped[1], -0.27, 0.27)
        clipped[2] = np.clip(clipped[2], 0.99, 1.25)
        positions.append(clipped)
        orientations.append(np.asarray(orientation, dtype=np.float64))
        grippers.append(float(gripper))

    if primitive == "stir_circle":
        for t in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
            append(anchor + [0.035 * np.cos(t), 0.035 * np.sin(t), 0.004 * np.sin(2.0 * t)])
    elif primitive == "stir_figure_eight":
        for t in np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False):
            append(anchor + [0.042 * np.sin(t), 0.025 * np.sin(2.0 * t), 0.0])
    elif primitive == "shake_lateral":
        angle = float(rng.uniform(-np.pi, np.pi))
        direction = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        values: list[float] = []
        current = 0.0
        for _ in range(3):
            values.extend(np.linspace(current, 0.035, 5).tolist())
            values.extend([0.035] * 4)
            values.extend(np.linspace(0.035, -0.035, 9).tolist())
            values.extend([-0.035] * 4)
            current = -0.035
        values.extend(np.linspace(current, 0.0, 5).tolist())
        for value in values:
            append(anchor + direction * value)
    elif primitive == "shake_vertical":
        values = []
        current = 0.0
        for _ in range(3):
            values.extend(np.linspace(current, 0.028, 5).tolist())
            values.extend([0.028] * 4)
            values.extend(np.linspace(0.028, -0.020, 8).tolist())
            values.extend([-0.020] * 4)
            current = -0.020
        values.extend(np.linspace(current, 0.0, 5).tolist())
        for value in values:
            append(anchor + [0.0, 0.0, value])
    elif primitive == "hammer_tap":
        for _ in range(5):
            for value in [0.035, 0.015, -0.025, 0.010, 0.035]:
                append(anchor + [0.012, 0.0, value])
    elif primitive == "wipe_raster":
        for row, y_offset in enumerate(np.linspace(-0.040, 0.040, 5)):
            xs = np.linspace(-0.055, 0.055, 6)
            if row % 2:
                xs = xs[::-1]
            for x_offset in xs:
                append(anchor + [x_offset, y_offset, 0.0])
    elif primitive == "scrape_stroke":
        for _ in range(4):
            for x_offset in np.linspace(-0.045, 0.055, 8):
                append(anchor + [x_offset, 0.0, -0.006])
            append(anchor + [-0.045, 0.0, 0.020])
    elif primitive == "twist_wrist":
        for yaw in np.tile([0.0, 8.0, 16.0, 8.0, 0.0, -8.0, -16.0, -8.0], 3):
            append(anchor, calibration.rotation_xyz_deg(0.0, 0.0, float(yaw)) @ base_orientation)
    elif primitive == "tilt_pour":
        for roll in [0.0, 5.0, 10.0, 14.0, 14.0, 10.0, 5.0, 0.0] * 3:
            append(anchor + [0.0, 0.0, 0.004 * math.sin(math.radians(roll))], calibration.rotation_xyz_deg(float(roll), 0.0, 0.0) @ base_orientation)
    elif primitive == "press_poke":
        for _ in range(5):
            for offset in [[0.0, 0.0, 0.025], [0.010, 0.0, 0.0], [0.020, 0.0, -0.025], [0.010, 0.0, 0.0]]:
                append(anchor + offset)
    elif primitive == "scoop_lift":
        for t in np.linspace(0.0, 1.0, 28):
            append(anchor + [-0.030 + 0.060 * t, 0.0, 0.035 * (t**2)])
        for t in np.linspace(1.0, 0.0, 12):
            append(anchor + [-0.030 + 0.060 * t, 0.0, 0.035 * (t**2)])
    elif primitive == "pinch_reposition":
        for cycle in range(5):
            direction = -1.0 if cycle % 2 else 1.0
            for gripper in [-1.0, 1.0, 1.0, -1.0]:
                append(anchor + [0.0, direction * 0.025, 0.010 * (gripper > 0.0)], gripper=gripper)
    else:
        raise ValueError(primitive)

    return np.asarray(positions), orientations, np.asarray(grippers, dtype=np.float32)


def follow_micro_path(
    recorder: Any,
    positions: np.ndarray,
    orientations: list[np.ndarray],
    grippers: np.ndarray,
    *,
    phase: str,
    trace: Any,
) -> dict[str, Any]:
    actual_positions: list[np.ndarray] = [rich.current_eef(recorder)]
    start_orientation = calibration.current_orientation(recorder)
    max_orientation_change_deg = 0.0
    for index, (target_xyz, target_orientation, gripper) in enumerate(
        zip(positions, orientations, grippers)
    ):
        for _ in range(2):
            actual_xyz = rich.current_eef(recorder)
            actual_orientation = calibration.current_orientation(recorder)
            position_delta = target_xyz - actual_xyz
            rotation_delta = calibration.orientation_error(target_orientation, actual_orientation)
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(position_delta * 3.5, -0.14, 0.14)
            action[3:6] = np.clip(rotation_delta / 0.5, -0.10, 0.10)
            action[6] = float(gripper)
            recorder.step(action, phase=f"{phase}_{index:03d}")
            trace.sample(recorder)
            actual_positions.append(rich.current_eef(recorder))
            max_orientation_change_deg = max(
                max_orientation_change_deg,
                calibration.orientation_distance_deg(
                    start_orientation, calibration.current_orientation(recorder)
                ),
            )

    actual = np.asarray(actual_positions)
    span = np.ptp(actual, axis=0)
    return {
        "commanded_points": len(positions),
        "control_steps": len(positions) * 2,
        "actual_eef_span_xyz_m": span.tolist(),
        "actual_eef_span_norm_m": float(np.linalg.norm(span)),
        "actual_orientation_span_deg": float(max_orientation_change_deg),
    }


def select_micro_case(
    primitive: str,
    repetition: int,
    primitive_index: int,
    *,
    rng: np.random.Generator | None = None,
    resample: bool = False,
) -> dict[str, Any]:
    required_use = PRIMITIVE_TO_USE[primitive]
    if required_use == "reset_upright":
        conservative = [
            case
            for case in POSITION_CASES
            if case["target_xyz_m"][2] <= 1.12
            and -0.36 <= case["target_xyz_m"][0] <= 0.06
            and abs(case["target_xyz_m"][1]) <= 0.20
        ]
        if resample and rng is not None:
            return conservative[int(rng.integers(0, len(conservative)))]
        return conservative[(repetition * 5 + primitive_index * 11) % len(conservative)]
    matching = [
        case for case in ORIENTATION_CASES if case["orientation"].get("use") == required_use
    ]
    if primitive == "scoop_lift":
        central = [
            case
            for case in matching
            if case["orientation"]["name"] == "tool_pitch_forward_15"
            and abs(case["target_xyz_m"][0]) <= 0.18
            and abs(case["target_xyz_m"][1]) <= 0.18
        ]
        if central:
            matching = central
    if resample and rng is not None:
        return matching[int(rng.integers(0, len(matching)))]
    return matching[(repetition * 7 + primitive_index * 3) % len(matching)]


def run_micro_sequence(
    recorder: Any,
    repetition: int,
    rng: np.random.Generator,
    *,
    retry_attempt: int = 0,
) -> dict[str, Any]:
    trace = calibration.SafetyTrace()
    home_xyz = rich.current_eef(recorder)
    home_orientation = calibration.current_orientation(recorder)
    sequence = micro_sequence_for(repetition)
    primitive_results: list[dict[str, Any]] = []

    for primitive_index, primitive in enumerate(sequence):
        case = select_micro_case(
            primitive,
            repetition,
            primitive_index,
            rng=rng,
            resample=retry_attempt > 0,
        )
        anchor, tool_orientation = pose_for_case(case, home_orientation)
        approach = checked_move(
            recorder,
            anchor,
            tool_orientation,
            phase=f"micro_{primitive_index:02d}_{primitive}_approach",
            trace=trace,
        )
        positions, orientations, grippers = primitive_path(
            primitive, anchor, tool_orientation, rng
        )
        motion = follow_micro_path(
            recorder,
            positions,
            orientations,
            grippers,
            phase=f"micro_{primitive_index:02d}_{primitive}",
            trace=trace,
        )
        settle = checked_move(
            recorder,
            anchor,
            tool_orientation,
            phase=f"micro_{primitive_index:02d}_{primitive}_settle",
            trace=trace,
        )
        unrotate = checked_move(
            recorder,
            anchor,
            home_orientation,
            phase=f"micro_{primitive_index:02d}_{primitive}_unrotate",
            trace=trace,
        )
        returned = checked_move(
            recorder,
            home_xyz,
            home_orientation,
            phase=f"micro_{primitive_index:02d}_{primitive}_return",
            trace=trace,
            returning_home=True,
        )
        if motion["actual_eef_span_norm_m"] < 0.010 and motion["actual_orientation_span_deg"] < 3.0:
            raise RetryableEpisodeError(f"Micro action did not visibly move: {primitive}")
        primitive_results.append(
            {
                "primitive": primitive,
                "validated_case_id": case["id"],
                "imagined_tool_frame": True,
                "visible_tool_asset": False,
                "tool_orientation_preset": case["orientation"],
                "approach": approach,
                "motion": motion,
                "settle": settle,
                "unrotate": unrotate,
                "return": returned,
            }
        )

    require_safe_trace(trace, "imagined_tool_micro_sequence")
    return {
        "instruction": "Move through safe locations and perform this imagined-tool sequence: "
        + ", ".join(name.replace("_", " ") for name in sequence)
        + ".",
        "primitive_sequence": sequence,
        "primitive_results": primitive_results,
        "imagined_tool_only": True,
        "visible_tool_asset": False,
        "safety": trace.as_dict(),
        "runtime_gate_pass": True,
    }


def run_family_once(
    recorder: Any,
    family: str,
    repetition: int,
    rng: np.random.Generator,
    *,
    retry_attempt: int = 0,
) -> dict[str, Any]:
    if family == "validated_multi_node_route":
        return run_validated_route(
            recorder, repetition, rng, retry_attempt=retry_attempt
        )
    if family == "validated_pose_star_tour":
        return run_pose_star(recorder, repetition, rng)
    if family == "short_poke":
        return rich.run_event_pulse(
            recorder,
            repetition,
            rng,
            family=family,
            peak_range=(0.14, 0.24),
            offset_range=(0.17, 0.18),
            post_contact_range=(2, 3),
        )
    if family == "impulse_tap":
        return rich.run_event_pulse(
            recorder,
            repetition,
            rng,
            family=family,
            peak_range=(0.24, 0.36),
            offset_range=(0.17, 0.18),
            post_contact_range=(3, 4),
        )
    if family == "strong_ram":
        return rich.run_event_pulse(
            recorder,
            repetition,
            rng,
            family=family,
            peak_range=(0.36, 0.50),
            offset_range=(0.17, 0.18),
            post_contact_range=(3, 4),
        )
    if family == "position_push":
        return rich.run_position_push(recorder, repetition, rng)
    if family == "contact_sweep_press":
        return rich.run_contact_sweep_press(recorder, repetition, rng)
    if family == "direct_grasp_far_place":
        return rich.run_direct_grasp_far_place(recorder, repetition, rng)
    if family == "push_then_grasp_far_place":
        return rich.run_push_then_grasp_far_place(recorder, repetition, rng)
    if family == "grasp_carry_waypoint_place":
        return rich.run_grasp_carry_waypoint_place(recorder, repetition, rng)
    if family == "imagined_tool_micro_sequence":
        return run_micro_sequence(
            recorder, repetition, rng, retry_attempt=retry_attempt
        )
    raise ValueError(family)


def reset_retry_attempt(recorder: Any) -> None:
    recorder.dataset.clear_episode_buffer()
    fresh = type(recorder)(env=recorder.env, dataset=recorder.dataset)
    recorder.__dict__.clear()
    recorder.__dict__.update(fresh.__dict__)


def run_family(recorder: Any, family: str, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    attempt = 0
    while True:
        if attempt:
            reset_retry_attempt(recorder)
        if attempt == 0:
            attempt_rng = rng
            attempt_seed = None
        else:
            family_code = sum((index + 1) * ord(char) for index, char in enumerate(family))
            attempt_seed = int(
                (20260716 + family_code * 1009 + repetition * 1000003 + attempt * 10000019)
                % (2**63 - 1)
            )
            attempt_rng = np.random.default_rng(attempt_seed)
        try:
            result = run_family_once(
                recorder,
                family,
                repetition,
                attempt_rng,
                retry_attempt=attempt,
            )
            result["rollout_attempts"] = attempt + 1
            result["resampled_retries"] = attempt
            result["accepted_retry_seed"] = attempt_seed
            return result
        except RetryableEpisodeError as error:
            attempt += 1
            next_family_code = sum(
                (index + 1) * ord(char) for index, char in enumerate(family)
            )
            next_seed = int(
                (20260716 + next_family_code * 1009 + repetition * 1000003 + attempt * 10000019)
                % (2**63 - 1)
            )
            print(
                f"[retry] family={family} repetition={repetition} "
                f"attempt={attempt} new_setting=true next_seed={next_seed} reason={error}",
                flush=True,
            )


def make_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for family_index, (family, count) in enumerate(FAMILY_COUNTS.items()):
        for repetition in range(count):
            plan.append(
                {
                    "family": family,
                    "family_index": family_index,
                    "family_repetition": repetition,
                    "behavior_group": GROUP_BY_FAMILY[family],
                    "parameter_seed": 20260716 + family_index * 10_000 + repetition,
                }
            )
    rng = np.random.default_rng(20260716)
    rng.shuffle(plan)
    return plan


def episode_task_prompts(family: str) -> list[str]:
    return TASK_PROMPTS[family]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_validated_graph(args.graph)
    if sum(FAMILY_COUNTS.values()) != EXPECTED_EPISODES:
        raise RuntimeError("Formal family counts do not sum to 400")

    rich.FAMILY_COUNTS = FAMILY_COUNTS
    rich.EXPECTED_EPISODES = EXPECTED_EPISODES
    rich.GROUP_BY_FAMILY = GROUP_BY_FAMILY
    rich.TASK_PROMPTS = TASK_PROMPTS
    rich.make_plan = make_plan
    rich.run_family = run_family
    rich.episode_task_prompts = episode_task_prompts
    rich.observable_interaction = lambda recorder: True

    forwarded = [str(rich.__file__), "--output", str(args.output)]
    if args.overwrite:
        forwarded.append("--overwrite")
    original_argv = sys.argv
    try:
        sys.argv = forwarded
        rich.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

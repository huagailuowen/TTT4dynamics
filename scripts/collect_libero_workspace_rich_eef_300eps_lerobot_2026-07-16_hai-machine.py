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
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
LIBERO_REPO = REPO_ROOT.parent / "LIBERO"
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM-TTT"
for path in (REPO_ROOT, SCRIPTS_DIR, LIBERO_REPO, FASTWAM_ROOT, FASTWAM_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SOURCE_COLLECTOR = SCRIPTS_DIR / "collect_libero_various_actions_200eps_lerobot_2026-07-15_hai-machine.py"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "various_actions"
    / "libero_mu0100_workspace_rich_eef_300eps_lerobot_2026-07-16_hai-machine"
)

EXPECTED_EPISODES = 300
BASE_SEED = 20260716
SIM_SEED = 20260716

# Conservative, reachable airspace above the classic LIBERO table. Contact
# motions use the measured object pose and are allowed below FREE_Z_MIN.
WORKSPACE_X = (-0.46, 0.40)
WORKSPACE_Y = (-0.30, 0.30)
FREE_Z = (0.98, 1.28)
PLACE_X = (-0.38, 0.32)
PLACE_Y = (-0.24, 0.24)
X_CELLS = 4
Y_CELLS = 3
Z_BANDS = 3

FAMILY_COUNTS = {
    # 100 broad free-space EEF episodes.
    "workspace_waypoint_tour": 40,
    "global_axis_polyline": 20,
    "global_geometric_path": 20,
    "far_reach_gripper_pose": 10,
    "orientation_workspace_motion": 10,
    # 100 object-contact dynamics episodes.
    "short_poke": 25,
    "impulse_tap": 20,
    "strong_ram": 20,
    "position_push": 25,
    "contact_sweep_press": 10,
    # 100 grasping and long-horizon composition episodes.
    "direct_grasp_far_place": 40,
    "push_then_grasp_far_place": 35,
    "grasp_carry_waypoint_place": 25,
}

GROUP_BY_FAMILY = {
    **{family: "workspace_exploration" for family in list(FAMILY_COUNTS)[:5]},
    **{family: "object_contact" for family in list(FAMILY_COUNTS)[5:10]},
    **{family: "grasp_and_transport" for family in list(FAMILY_COUNTS)[10:]},
}

TASK_PROMPTS = {
    "workspace_waypoint_tour": [
        "move the gripper through widely separated positions across the whole table workspace",
        "visit several distant points at different heights above the table",
        "explore the full tabletop airspace with a sequence of end-effector motions",
        "move the robot hand between multiple far-apart workspace locations",
    ],
    "global_axis_polyline": [
        "move the gripper along long straight and piecewise-linear paths across the table",
        "sweep the end effector across distant table regions using axis-aligned segments",
        "follow a broad zigzag or raster path through the tabletop workspace",
        "execute long horizontal and vertical end-effector translations",
    ],
    "global_geometric_path": [
        "trace a geometric path at a sampled location and height above the table",
        "move around a circle polygon spiral or figure eight in free space",
        "follow a broad geometric trajectory without contacting the object",
        "draw a smooth multi-directional path with the robot hand above the table",
    ],
    "far_reach_gripper_pose": [
        "reach to distant workspace corners and open and close the gripper",
        "extend the robot arm far across the table and exercise the gripper",
        "move between far-apart poses while alternating gripper states",
        "perform empty-space reach and grasp motions at distant table locations",
    ],
    "orientation_workspace_motion": [
        "translate through the workspace while changing end-effector orientation",
        "move to distant points and rotate the robot hand around different axes",
        "combine broad end-effector translations with controlled orientation changes",
        "explore position and rotation effects throughout the tabletop airspace",
    ],
    "short_poke": [
        "approach the box from one side and give it a short controlled poke",
        "make brief contact with the box using a low-force end-effector thrust",
        "tap the box lightly from a sampled planar direction",
        "perform a short contact impulse and then clear the object",
    ],
    "impulse_tap": [
        "strike the box with a medium impulse from a sampled direction",
        "approach and tap the object using a brief event-triggered thrust",
        "apply a medium dynamic push and stop shortly after first contact",
        "give the box one clear impulse and observe its resulting motion",
    ],
    "strong_ram": [
        "hit the box with a strong short thrust and then brake the end effector",
        "perform a high-force event-triggered ram from a sampled direction",
        "push the object dynamically with a brief high-amplitude command",
        "apply one forceful impact and observe the box move freely",
    ],
    "position_push": [
        "push the box along a controlled straight path from a sampled direction",
        "maintain contact while moving the object across a chosen table direction",
        "execute a slow or medium position-controlled push of the box",
        "approach behind the object and steadily push it for a sampled distance",
    ],
    "contact_sweep_press": [
        "perform a varied contact sequence such as sweeping pressing or two intentional pushes",
        "interact with the box using a curved side top or multi-stage contact motion",
        "execute a contact-rich manipulation around the object",
        "approach the box and apply a structured sweep press or repeated interaction",
    ],
    "direct_grasp_far_place": [
        "grasp the box lift it carry it to a distant table region and place it down",
        "pick up the object and transport it far across the table before releasing it",
        "perform a complete grasp lift long carry lower and place sequence",
        "move the box between distant table locations using a stable grasp",
    ],
    "push_then_grasp_far_place": [
        "push the box to a new position then grasp it and place it far away",
        "combine a short push with object reacquisition lifting transport and placement",
        "first move the object by contact then pick it up and carry it across the table",
        "execute push settle grasp lift far carry and release as one long task",
    ],
    "grasp_carry_waypoint_place": [
        "grasp the box and carry it through several high workspace waypoints before placing it",
        "pick up the object move it through a multi-segment path and set it down far away",
        "transport the grasped box across multiple table regions at different heights",
        "perform a long-horizon grasp carry waypoint tour and placement",
    ],
}

DESIGN_REFERENCES = [
    {
        "title": "World Action Models are Zero-shot Policies",
        "url": "https://arxiv.org/abs/2602.15922",
        "applied_principle": "diverse non-repetitive data, long episodes with multiple subtasks, native frame-rate video-action alignment",
    },
    {
        "title": "ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models",
        "url": "https://arxiv.org/abs/2605.08567",
        "applied_principle": "cover distinct kinematic and rigid-body interaction regimes with controlled action variation",
    },
    {
        "title": "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset",
        "url": "https://arxiv.org/abs/2403.12945",
        "applied_principle": "task and scene-state diversity rather than repeated trajectories from one narrow pose distribution",
    },
    {
        "title": "BridgeData V2: A Dataset for Robot Learning at Scale",
        "url": "https://arxiv.org/abs/2308.12952",
        "applied_principle": "multi-skill language-compatible trajectories and broad state coverage improve generalization",
    },
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source = load_module(SOURCE_COLLECTOR, "workspace_rich_eef_source_collector_hai_machine")
base = source.base
demo = source.demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a new 300-episode broad-workspace OSC_POSE LIBERO dataset in LeRobot format."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def episode_task_prompts(family: str) -> list[str]:
    if GROUP_BY_FAMILY[family] == "workspace_exploration":
        return [
            f"move through broad workspace poses, then {prompt}, and finish at other distant poses"
            for prompt in TASK_PROMPTS[family]
        ]
    return [
        f"{prompt}, then move the gripper through several distant workspace poses"
        for prompt in TASK_PROMPTS[family]
    ]


def current_eef(recorder: Any) -> np.ndarray:
    return np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64).copy()


def current_box(recorder: Any) -> np.ndarray:
    xyz, _ = recorder.env.box_pose()
    return np.asarray(xyz, dtype=np.float64).copy()


def current_box_yaw(recorder: Any) -> float:
    _, quat = recorder.env.box_pose()
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def clip_free_target(target_xyz: np.ndarray, box_xy: np.ndarray) -> np.ndarray:
    target = np.asarray(target_xyz, dtype=np.float64).copy()
    target[0] = float(np.clip(target[0], *WORKSPACE_X))
    target[1] = float(np.clip(target[1], *WORKSPACE_Y))
    target[2] = float(np.clip(target[2], *FREE_Z))
    if float(np.linalg.norm(target[:2] - box_xy)) < 0.09 and target[2] < 1.055:
        target[2] = 1.075
    return target


def move_and_measure(
    recorder: Any,
    target_xyz: np.ndarray | tuple[float, float, float],
    *,
    gripper: float,
    phase: str,
    rng: np.random.Generator,
    max_action: float = 0.18,
    steps: int | None = None,
    free_space: bool = True,
) -> dict[str, Any]:
    target = np.asarray(target_xyz, dtype=np.float64)
    if free_space:
        target = clip_free_target(target, current_box(recorder)[:2])
    start = current_eef(recorder)
    distance = float(np.linalg.norm(target - start))
    if steps is None:
        steps = int(np.clip(math.ceil(distance / 0.014) + 12, 20, 76))
    recorder.move_to(
        target,
        steps=int(steps),
        gripper=float(gripper),
        phase=phase,
        gain=float(rng.uniform(3.0, 3.6)),
        max_action=float(max_action),
    )
    achieved = current_eef(recorder)
    return {
        "phase": phase,
        "start_xyz_m": start,
        "target_xyz_m": target,
        "achieved_xyz_m": achieved,
        "target_error_m": float(np.linalg.norm(achieved - target)),
        "commanded_path_length_m": distance,
        "steps": int(steps),
        "gripper_env_command": float(gripper),
    }


def sample_workspace_point(
    rng: np.random.Generator,
    *,
    cell_index: int | None = None,
    height_band: int | None = None,
) -> np.ndarray:
    if cell_index is None:
        cell_index = int(rng.integers(0, X_CELLS * Y_CELLS))
    if height_band is None:
        height_band = int(rng.integers(0, Z_BANDS))
    ix = int(cell_index) % X_CELLS
    iy = int(cell_index) // X_CELLS
    x_edges = np.linspace(WORKSPACE_X[0], WORKSPACE_X[1], X_CELLS + 1)
    y_edges = np.linspace(WORKSPACE_Y[0], WORKSPACE_Y[1], Y_CELLS + 1)
    z_edges = np.linspace(FREE_Z[0], FREE_Z[1], Z_BANDS + 1)
    margin_xy = 0.014
    margin_z = 0.010
    return np.asarray(
        [
            rng.uniform(x_edges[ix] + margin_xy, x_edges[ix + 1] - margin_xy),
            rng.uniform(y_edges[iy] + margin_xy, y_edges[iy + 1] - margin_xy),
            rng.uniform(z_edges[height_band] + margin_z, z_edges[height_band + 1] - margin_z),
        ],
        dtype=np.float64,
    )


def run_workspace_context_segment(
    recorder: Any,
    repetition: int,
    rng: np.random.Generator,
    *,
    prefix: str,
    count: int = 2,
) -> dict[str, Any]:
    prefix_offset = 0 if prefix == "episode_prelude" else 7
    movements = []
    cells = []
    for index in range(int(count)):
        cell = (repetition * 5 + prefix_offset + index * 6) % (X_CELLS * Y_CELLS)
        cells.append(cell)
        target = sample_workspace_point(
            rng,
            cell_index=cell,
            height_band=(repetition + prefix_offset + index) % Z_BANDS,
        )
        gripper = -1.0 if (repetition + prefix_offset + index) % 2 == 0 else 1.0
        movements.append(
            move_and_measure(
                recorder,
                target,
                gripper=gripper,
                phase=f"{prefix}_workspace_pose_{index}",
                rng=rng,
            )
        )
        recorder.hold(4, gripper=gripper, phase=f"{prefix}_workspace_pose_{index}_hold")
    return {"cells": cells, "movements": movements}


def run_workspace_waypoint_tour(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    count = int(rng.integers(6, 9))
    required_cell = repetition % (X_CELLS * Y_CELLS)
    remaining = [cell for cell in range(X_CELLS * Y_CELLS) if cell != required_cell]
    rng.shuffle(remaining)
    cells = [required_cell] + remaining[: count - 1]
    rng.shuffle(cells)
    movements = []
    for index, cell in enumerate(cells):
        target = sample_workspace_point(rng, cell_index=cell, height_band=(repetition + index) % Z_BANDS)
        gripper = -1.0 if (repetition + index) % 3 else 1.0
        movements.append(
            move_and_measure(
                recorder,
                target,
                gripper=gripper,
                phase=f"workspace_waypoint_{index}",
                rng=rng,
            )
        )
        recorder.hold(3, gripper=gripper, phase=f"workspace_waypoint_{index}_hold")
    return {"sampled_cells": cells, "waypoint_count": count, "movements": movements}


def run_global_axis_polyline(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    pattern = ["x_sweep", "y_sweep", "diagonal", "raster", "height_staircase"][repetition % 5]
    x0, x1 = WORKSPACE_X[0] + 0.025, WORKSPACE_X[1] - 0.025
    y0, y1 = WORKSPACE_Y[0] + 0.025, WORKSPACE_Y[1] - 0.025
    z0 = float(rng.uniform(1.01, 1.08))
    z1 = float(rng.uniform(1.19, 1.26))
    if pattern == "x_sweep":
        y = float(rng.uniform(y0, y1))
        targets = [(x0, y, z0), (x1, y, z0), (x0, y, z1), (x1, y, z1)]
    elif pattern == "y_sweep":
        x = float(rng.uniform(x0, x1))
        targets = [(x, y0, z0), (x, y1, z0), (x, y0, z1), (x, y1, z1)]
    elif pattern == "diagonal":
        targets = [(x0, y0, z0), (x1, y1, z1), (x0, y1, z1), (x1, y0, z0)]
    elif pattern == "raster":
        targets = [(x0, y0, z0), (x1, y0, z0), (x1, 0.0, z1), (x0, 0.0, z1), (x0, y1, z0), (x1, y1, z0)]
    else:
        targets = [(x0, y0, z0), (x0, y0, z1), (0.0, 0.0, z0), (0.0, 0.0, z1), (x1, y1, z0), (x1, y1, z1)]
    movements = []
    for index, target in enumerate(targets):
        gripper = -1.0 if index % 2 == 0 else 1.0
        movements.append(
            move_and_measure(
                recorder,
                target,
                gripper=gripper,
                phase=f"global_{pattern}_{index}",
                rng=rng,
            )
        )
        recorder.hold(2, gripper=gripper, phase=f"global_{pattern}_{index}_hold")
    return {"pattern": pattern, "movements": movements}


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


def run_global_geometric_path(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    shape = ["circle", "ellipse", "spiral", "figure8", "square", "triangle"][repetition % 6]
    count = int(rng.integers(48, 69))
    rx = float(rng.uniform(0.075, 0.17))
    ry = float(rng.uniform(0.060, min(0.145, rx)))
    center = np.asarray(
        [
            rng.uniform(WORKSPACE_X[0] + rx + 0.015, WORKSPACE_X[1] - rx - 0.015),
            rng.uniform(WORKSPACE_Y[0] + ry + 0.015, WORKSPACE_Y[1] - ry - 0.015),
        ],
        dtype=np.float64,
    )
    z = float(rng.uniform(1.02, 1.25))
    phase = float(rng.uniform(-math.pi, math.pi))
    direction = -1.0 if repetition % 2 else 1.0
    progress = np.linspace(0.0, 1.0, count)
    theta = phase + direction * 2.0 * math.pi * progress
    if shape == "circle":
        radius = min(rx, ry)
        xy = np.stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)], axis=1)
    elif shape == "ellipse":
        xy = np.stack([center[0] + rx * np.cos(theta), center[1] + ry * np.sin(theta)], axis=1)
    elif shape == "spiral":
        radius = (1.0 - 0.65 * progress)
        xy = np.stack([center[0] + rx * radius * np.cos(1.5 * theta), center[1] + ry * radius * np.sin(1.5 * theta)], axis=1)
    elif shape == "figure8":
        xy = np.stack([center[0] + rx * np.sin(theta), center[1] + ry * np.sin(2.0 * theta)], axis=1)
    else:
        sides = 4 if shape == "square" else 3
        angles = phase + direction * np.linspace(0.0, 2.0 * math.pi, sides + 1)
        vertices = np.stack([center[0] + rx * np.cos(angles), center[1] + ry * np.sin(angles)], axis=1)
        xy = interpolate_polygon(vertices, count)
    points = np.concatenate([xy, np.full((len(xy), 1), z, dtype=np.float64)], axis=1)
    points = np.asarray([clip_free_target(point, current_box(recorder)[:2]) for point in points])
    initial_gripper = -1.0 if repetition % 4 < 2 else 1.0
    start_move = move_and_measure(
        recorder,
        points[0],
        gripper=initial_gripper,
        phase="move_to_global_geometry_start",
        rng=rng,
    )
    recorder.follow_points(
        points,
        gripper_fn=source.gripper_profile(repetition % 4),
        phase=f"global_geometry_{shape}",
        gain=float(rng.uniform(2.8, 3.4)),
        max_action=float(rng.uniform(0.10, 0.15)),
    )
    recorder.hold(6, gripper=source.gripper_profile(repetition % 4)(count - 1, count), phase="geometry_settle")
    return {
        "shape": shape,
        "center_xy_m": center,
        "radius_x_m": rx,
        "radius_y_m": ry,
        "z_m": z,
        "path_steps": count,
        "direction": "clockwise" if direction < 0 else "counterclockwise",
        "start_move": start_move,
    }


def run_far_reach_gripper_pose(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    corners = [
        np.asarray([WORKSPACE_X[0] + 0.025, WORKSPACE_Y[0] + 0.025, 1.04]),
        np.asarray([WORKSPACE_X[1] - 0.025, WORKSPACE_Y[1] - 0.025, 1.23]),
        np.asarray([WORKSPACE_X[0] + 0.025, WORKSPACE_Y[1] - 0.025, 1.20]),
        np.asarray([WORKSPACE_X[1] - 0.025, WORKSPACE_Y[0] + 0.025, 1.03]),
    ]
    corners = corners[repetition % 4 :] + corners[: repetition % 4]
    movements = []
    for index, target in enumerate(corners):
        movements.append(
            move_and_measure(
                recorder,
                target,
                gripper=-1.0,
                phase=f"far_reach_{index}",
                rng=rng,
            )
        )
        recorder.hold(5, gripper=1.0, phase=f"far_empty_grasp_{index}")
        recorder.hold(5, gripper=-1.0, phase=f"far_empty_release_{index}")
    return {"corner_order": [target.tolist() for target in corners], "movements": movements}


def run_orientation_workspace_motion(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    movements = []
    rotations = []
    for index in range(3):
        target = sample_workspace_point(
            rng,
            cell_index=(repetition * 3 + index * 5) % (X_CELLS * Y_CELLS),
            height_band=(repetition + index) % Z_BANDS,
        )
        gripper = -1.0 if index != 1 else 1.0
        movements.append(
            move_and_measure(
                recorder,
                target,
                gripper=gripper,
                phase=f"orientation_move_{index}",
                rng=rng,
            )
        )
        axis = (repetition + index) % 3
        amplitude = float(rng.uniform(0.035, 0.070))
        pulse_steps = int(rng.integers(8, 13))
        for sign, phase_name in ((1.0, "rotate_out"), (-1.0, "rotate_back")):
            for local_step in range(pulse_steps):
                action = np.zeros(7, dtype=np.float64)
                envelope = math.sin(math.pi * float(local_step + 1) / float(pulse_steps + 1))
                action[3 + axis] = sign * amplitude * envelope
                action[-1] = gripper
                recorder.step(action, f"orientation_{axis}_{index}_{phase_name}")
        rotations.append({"axis": axis, "peak_action": amplitude, "steps_each_direction": pulse_steps})
    recorder.hold(6, gripper=-1.0, phase="orientation_final_settle")
    return {"movements": movements, "rotations": rotations}


def planar_direction(repetition: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    angle = 2.0 * math.pi * float(repetition % 16) / 16.0 + math.radians(float(rng.uniform(-5.0, 5.0)))
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64), math.degrees(angle)


def approach_box(recorder: Any, direction: np.ndarray, offset_m: float, rng: np.random.Generator, phase: str) -> dict[str, Any]:
    box = current_box(recorder)
    direction = demo.normalized(direction)
    behind = box[:2] - direction * float(offset_m)
    above = np.asarray([behind[0], behind[1], 1.055], dtype=np.float64)
    contact = np.asarray([behind[0], behind[1], demo.CONTACT_Z], dtype=np.float64)
    first = move_and_measure(
        recorder,
        above,
        gripper=1.0,
        phase=f"{phase}_approach_above",
        rng=rng,
        max_action=0.18,
        free_space=False,
    )
    second = move_and_measure(
        recorder,
        contact,
        gripper=1.0,
        phase=f"{phase}_descend",
        rng=rng,
        max_action=0.13,
        steps=24,
        free_space=False,
    )
    return {"box_xyz_m": box, "behind_xy_m": behind, "above_move": first, "descend_move": second}


def finish_contact(recorder: Any, rng: np.random.Generator, phase: str) -> None:
    recorder.hold(10, gripper=1.0, phase=f"{phase}_observe")
    eef = current_eef(recorder)
    move_and_measure(
        recorder,
        (float(eef[0]), float(eef[1]), 1.085),
        gripper=1.0,
        phase=f"{phase}_clear",
        rng=rng,
        max_action=0.15,
        steps=20,
        free_space=False,
    )
    recorder.hold(4, gripper=1.0, phase=f"{phase}_settle")


def run_event_pulse(
    recorder: Any,
    repetition: int,
    rng: np.random.Generator,
    *,
    family: str,
    peak_range: tuple[float, float],
    offset_range: tuple[float, float],
    post_contact_range: tuple[int, int],
) -> dict[str, Any]:
    recorder.hold(4, gripper=1.0, phase="initial_observation")
    direction, angle_deg = planar_direction(repetition, rng)
    offset = float(rng.uniform(*offset_range))
    approach = approach_box(recorder, direction, offset, rng, family)
    peak = float(rng.uniform(*peak_range))
    post_contact = int(rng.integers(post_contact_range[0], post_contact_range[1] + 1))
    box_before = current_box(recorder)
    event = recorder.contact_triggered_pulse(
        direction,
        peak=peak,
        line_xy=box_before[:2],
        contact_z=demo.CONTACT_Z,
        max_precontact_steps=38,
        hold_after_contact=post_contact,
        phase=family,
    )
    finish_contact(recorder, rng, family)
    return {
        "direction_xy": direction,
        "angle_deg": angle_deg,
        "offset_m": offset,
        "peak_action": peak,
        "hold_after_contact": post_contact,
        "approach": approach,
        "contact_event": event,
    }


def run_position_push(
    recorder: Any,
    repetition: int,
    rng: np.random.Generator,
    *,
    phase: str = "position_push",
    direction_override: np.ndarray | None = None,
    short: bool = False,
) -> dict[str, Any]:
    recorder.hold(4, gripper=1.0, phase="initial_observation")
    if direction_override is None:
        direction, angle_deg = planar_direction(repetition, rng)
    else:
        direction = demo.normalized(direction_override)
        angle_deg = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    offset = float(rng.uniform(0.120, 0.145) if short else rng.uniform(0.125, 0.160))
    approach = approach_box(recorder, direction, offset, rng, phase)
    start = np.asarray([approach["behind_xy_m"][0], approach["behind_xy_m"][1], demo.CONTACT_Z])
    distance = float(rng.uniform(0.145, 0.185) if short else rng.uniform(0.17, 0.25))
    end = start.copy()
    end[:2] += direction * distance
    steps = int(rng.integers(34, 45) if short else rng.integers(32, 53))
    max_action = float(rng.uniform(0.075, 0.105) if short else rng.uniform(0.08, 0.15))
    recorder.track_line(
        start,
        end,
        steps=steps,
        gripper=1.0,
        phase=phase,
        gain=float(rng.uniform(2.7, 3.3)),
        max_action=max_action,
    )
    finish_contact(recorder, rng, phase)
    return {
        "direction_xy": direction,
        "angle_deg": angle_deg,
        "offset_m": offset,
        "track_distance_m": distance,
        "track_steps": steps,
        "max_action": max_action,
        "approach": approach,
    }


def run_contact_sweep_press(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    variant = ["curved_cw", "curved_ccw", "top_press", "side_scrape", "two_intentional_taps"][repetition % 5]
    if variant == "curved_cw":
        return {"variant": variant, "details": source.run_curved_contact_sweep(recorder, clockwise=True, rng=rng)}
    if variant == "curved_ccw":
        return {"variant": variant, "details": source.run_curved_contact_sweep(recorder, clockwise=False, rng=rng)}
    if variant == "top_press":
        return {"variant": variant, "details": source.run_top_press(recorder, rng=rng)}
    if variant == "side_scrape":
        direction = np.asarray([0.0, 1.0 if (repetition // 5) % 2 == 0 else -1.0])
        return {
            "variant": variant,
            "details": run_position_push(
                recorder,
                repetition,
                rng,
                phase="side_contact_scrape",
                direction_override=direction,
            ),
        }

    recorder.hold(4, gripper=1.0, phase="initial_observation")
    direction, angle_deg = planar_direction(repetition, rng)
    first_approach = approach_box(recorder, direction, 0.145, rng, "first_intentional_tap")
    first_box = current_box(recorder)
    first_event = recorder.contact_triggered_pulse(
        direction,
        peak=float(rng.uniform(0.24, 0.32)),
        line_xy=first_box[:2],
        contact_z=demo.CONTACT_Z,
        max_precontact_steps=36,
        hold_after_contact=2,
        phase="first_intentional_tap",
    )
    recorder.hold(10, gripper=1.0, phase="between_intentional_taps")
    eef = current_eef(recorder)
    move_and_measure(
        recorder,
        (eef[0], eef[1], 1.08),
        gripper=1.0,
        phase="reposition_for_second_tap",
        rng=rng,
        free_space=False,
    )
    second_direction = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    second_approach = approach_box(recorder, second_direction, 0.14, rng, "second_intentional_tap")
    second_box = current_box(recorder)
    second_event = recorder.contact_triggered_pulse(
        second_direction,
        peak=float(rng.uniform(0.20, 0.29)),
        line_xy=second_box[:2],
        contact_z=demo.CONTACT_Z,
        max_precontact_steps=36,
        hold_after_contact=2,
        phase="second_intentional_tap",
    )
    finish_contact(recorder, rng, "two_intentional_taps")
    return {
        "variant": variant,
        "first_angle_deg": angle_deg,
        "first_approach": first_approach,
        "first_event": first_event,
        "second_approach": second_approach,
        "second_event": second_event,
    }


def sample_far_place(current_xy: np.ndarray, repetition: int, rng: np.random.Generator) -> np.ndarray:
    x_values = np.linspace(PLACE_X[0] + 0.02, PLACE_X[1] - 0.02, 4)
    y_values = np.linspace(PLACE_Y[0] + 0.02, PLACE_Y[1] - 0.02, 3)
    candidates = [np.asarray([x, y], dtype=np.float64) for y in y_values for x in x_values]
    candidates = candidates[repetition % len(candidates) :] + candidates[: repetition % len(candidates)]
    candidates = [candidate + rng.uniform(-0.012, 0.012, size=2) for candidate in candidates]
    valid = [candidate for candidate in candidates if float(np.linalg.norm(candidate - current_xy)) >= 0.26]
    if valid:
        return valid[0]
    return max(candidates, key=lambda candidate: float(np.linalg.norm(candidate - current_xy)))


def grasp_current_box(recorder: Any, rng: np.random.Generator, phase_prefix: str) -> dict[str, Any]:
    attempts = []
    # The object starts at z ~= 0.935 m but settles to z ~= 0.909 m after a
    # push. The proven grasp EEF height is absolute z ~= 0.900 m in this fixed
    # table scene; subtracting a fixed offset from the post-push box center
    # would incorrectly command the hand below the table.
    grasp_z_targets = [demo.CONTACT_Z - 0.015, demo.CONTACT_Z - 0.007, demo.CONTACT_Z - 0.021]
    for attempt_index, grasp_z_target in enumerate(grasp_z_targets):
        box_before = current_box(recorder)
        lift_z = float(rng.uniform(1.13, 1.21))
        approach_target = np.asarray([box_before[0], box_before[1], 1.13], dtype=np.float64)
        descend_target = np.asarray([box_before[0], box_before[1], grasp_z_target], dtype=np.float64)
        approach_move = move_and_measure(
            recorder,
            approach_target,
            gripper=-1.0,
            phase=f"{phase_prefix}_approach_{attempt_index}",
            rng=rng,
            max_action=0.18,
            free_space=False,
        )
        descend_move = move_and_measure(
            recorder,
            descend_target,
            gripper=-1.0,
            phase=f"{phase_prefix}_descend_{attempt_index}",
            rng=rng,
            max_action=0.12,
            steps=34,
            free_space=False,
        )
        recorder.hold(4, gripper=-1.0, phase=f"{phase_prefix}_open_align_{attempt_index}")
        recorder.hold(28, gripper=1.0, phase=f"{phase_prefix}_close_{attempt_index}")
        grasp_before_lift = bool(demo.grasping(recorder.env))
        lift_move = move_and_measure(
            recorder,
            (box_before[0], box_before[1], lift_z),
            gripper=1.0,
            phase=f"{phase_prefix}_lift_{attempt_index}",
            rng=rng,
            max_action=0.16,
            steps=36,
            free_space=False,
        )
        box_after = current_box(recorder)
        grasp_after_lift = bool(demo.grasping(recorder.env))
        lifted = bool(float(box_after[2] - box_before[2]) > 0.035)
        success = bool(grasp_before_lift or grasp_after_lift or lifted)
        attempts.append(
            {
                "attempt_index": attempt_index,
                "box_before_xyz_m": box_before,
                "box_after_xyz_m": box_after,
                "grasp_eef_target_z_m": grasp_z_target,
                "box_center_to_eef_z_m": float(grasp_z_target - box_before[2]),
                "approach_move": approach_move,
                "descend_move": descend_move,
                "lift_move": lift_move,
                "robosuite_grasp_before_lift": grasp_before_lift,
                "robosuite_grasp_after_lift": grasp_after_lift,
                "box_lifted": lifted,
                "success": success,
            }
        )
        if success:
            return {"success": True, "attempt_count": attempt_index + 1, "attempts": attempts, "lift_z_m": lift_z}
        recorder.hold(10, gripper=-1.0, phase=f"{phase_prefix}_failed_release_{attempt_index}")
        eef = current_eef(recorder)
        move_and_measure(
            recorder,
            (eef[0], eef[1], 1.13),
            gripper=-1.0,
            phase=f"{phase_prefix}_retry_clear_{attempt_index}",
            rng=rng,
            max_action=0.15,
            free_space=False,
        )
    return {"success": False, "attempt_count": len(attempts), "attempts": attempts, "lift_z_m": 1.15}


def place_grasped_box(
    recorder: Any,
    place_xy: np.ndarray,
    rng: np.random.Generator,
    *,
    phase_prefix: str,
    carry_waypoints: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    carry_z = float(rng.uniform(1.14, 1.22))
    moves = []
    for index, waypoint in enumerate(carry_waypoints or []):
        target = np.asarray([waypoint[0], waypoint[1], max(carry_z, float(waypoint[2]))])
        moves.append(
            move_and_measure(
                recorder,
                target,
                gripper=1.0,
                phase=f"{phase_prefix}_carry_waypoint_{index}",
                rng=rng,
                max_action=0.16,
                free_space=True,
            )
        )
    moves.append(
        move_and_measure(
            recorder,
            (place_xy[0], place_xy[1], carry_z),
            gripper=1.0,
            phase=f"{phase_prefix}_carry_to_place",
            rng=rng,
            max_action=0.16,
            free_space=False,
        )
    )
    moves.append(
        move_and_measure(
            recorder,
            (place_xy[0], place_xy[1], 0.99),
            gripper=1.0,
            phase=f"{phase_prefix}_lower",
            rng=rng,
            max_action=0.10,
            steps=30,
            free_space=False,
        )
    )
    recorder.hold(24, gripper=-1.0, phase=f"{phase_prefix}_release")
    box_after_release = current_box(recorder)
    eef = current_eef(recorder)
    moves.append(
        move_and_measure(
            recorder,
            (eef[0], eef[1], 1.14),
            gripper=-1.0,
            phase=f"{phase_prefix}_clear",
            rng=rng,
            max_action=0.15,
            free_space=False,
        )
    )
    recorder.hold(6, gripper=-1.0, phase=f"{phase_prefix}_final_settle")
    return {"place_xy_m": place_xy, "carry_z_m": carry_z, "moves": moves, "box_after_release_xyz_m": box_after_release}


def run_direct_grasp_far_place(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    grasp = grasp_current_box(recorder, rng, "direct_grasp")
    if not grasp["success"]:
        raise RuntimeError(f"Direct grasp failed after {grasp['attempt_count']} attempts")
    box_held = current_box(recorder)
    place_xy = sample_far_place(box_held[:2], repetition, rng)
    placement = place_grasped_box(recorder, place_xy, rng, phase_prefix="direct_far_place")
    return {"grasp": grasp, "placement": placement}


def run_push_then_grasp_far_place(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=1.0, phase="initial_observation")
    # Keep this composition's first push centered and gentle so the rectangular
    # box remains aligned for the subsequent top grasp. Directional and strong
    # pushes are already covered by the dedicated 100 contact episodes.
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    angle_deg = 0.0
    box_before_push = current_box(recorder)
    yaw_before_push = current_box_yaw(recorder)
    offset = float(rng.uniform(0.122, 0.132))
    approach = approach_box(recorder, direction, offset, rng, "pregrasp_short_push")
    start = np.asarray([approach["behind_xy_m"][0], approach["behind_xy_m"][1], demo.CONTACT_Z])
    end = start.copy()
    push_track_distance = float(rng.uniform(0.142, 0.156))
    end[:2] += direction * push_track_distance
    push_steps = int(rng.integers(40, 49))
    push_max_action = float(rng.uniform(0.065, 0.085))
    recorder.track_line(
        start,
        end,
        steps=push_steps,
        gripper=1.0,
        phase="pregrasp_short_push",
        gain=float(rng.uniform(2.7, 3.0)),
        max_action=push_max_action,
    )
    recorder.hold(18, gripper=1.0, phase="pregrasp_object_settle")
    eef = current_eef(recorder)
    move_and_measure(
        recorder,
        (eef[0], eef[1], 1.10),
        gripper=-1.0,
        phase="pregrasp_clear_after_push",
        rng=rng,
        max_action=0.15,
        free_space=False,
    )
    box_after_push = current_box(recorder)
    yaw_after_push = current_box_yaw(recorder)
    yaw_delta = float(math.atan2(math.sin(yaw_after_push - yaw_before_push), math.cos(yaw_after_push - yaw_before_push)))
    if abs(yaw_delta) > math.radians(7.0):
        raise RuntimeError(f"Pre-grasp push rotated box by {math.degrees(yaw_delta):.2f} deg")
    grasp = grasp_current_box(recorder, rng, "postpush_grasp")
    if not grasp["success"]:
        print(
            "POST_PUSH_GRASP_FAILURE="
            + json.dumps(
                base.to_jsonable(
                    {
                        "box_before_push_xyz_m": box_before_push,
                        "box_after_push_xyz_m": box_after_push,
                        "box_yaw_before_push_rad": yaw_before_push,
                        "box_yaw_after_push_rad": yaw_after_push,
                        "box_yaw_delta_rad": yaw_delta,
                        "grasp": grasp,
                    }
                ),
                indent=2,
            ),
            flush=True,
        )
        raise RuntimeError(f"Post-push grasp failed after {grasp['attempt_count']} attempts")
    box_held = current_box(recorder)
    place_xy = sample_far_place(box_held[:2], repetition + 3, rng)
    placement = place_grasped_box(recorder, place_xy, rng, phase_prefix="postpush_far_place")
    return {
        "push_angle_deg": angle_deg,
        "push_direction_xy": direction,
        "push_track_distance_m": push_track_distance,
        "push_steps": push_steps,
        "push_max_action": push_max_action,
        "push_approach": approach,
        "box_before_push_xyz_m": box_before_push,
        "box_after_push_xyz_m": box_after_push,
        "box_yaw_before_push_rad": yaw_before_push,
        "box_yaw_after_push_rad": yaw_after_push,
        "box_yaw_delta_rad": yaw_delta,
        "grasp": grasp,
        "placement": placement,
    }


def run_grasp_carry_waypoint_place(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    recorder.hold(4, gripper=-1.0, phase="initial_observation")
    grasp = grasp_current_box(recorder, rng, "waypoint_grasp")
    if not grasp["success"]:
        raise RuntimeError(f"Waypoint grasp failed after {grasp['attempt_count']} attempts")
    waypoint_count = int(rng.integers(2, 5))
    cells = list(range(X_CELLS * Y_CELLS))
    rng.shuffle(cells)
    waypoints = [
        sample_workspace_point(rng, cell_index=cells[index], height_band=2)
        for index in range(waypoint_count)
    ]
    box_held = current_box(recorder)
    place_xy = sample_far_place(box_held[:2], repetition + 6, rng)
    placement = place_grasped_box(
        recorder,
        place_xy,
        rng,
        phase_prefix="waypoint_far_place",
        carry_waypoints=waypoints,
    )
    return {"grasp": grasp, "carry_waypoints_xyz_m": waypoints, "placement": placement}


def run_family(recorder: Any, family: str, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    if family == "workspace_waypoint_tour":
        return run_workspace_waypoint_tour(recorder, repetition, rng)
    if family == "global_axis_polyline":
        return run_global_axis_polyline(recorder, repetition, rng)
    if family == "global_geometric_path":
        return run_global_geometric_path(recorder, repetition, rng)
    if family == "far_reach_gripper_pose":
        return run_far_reach_gripper_pose(recorder, repetition, rng)
    if family == "orientation_workspace_motion":
        return run_orientation_workspace_motion(recorder, repetition, rng)
    if family == "short_poke":
        return run_event_pulse(
            recorder,
            repetition,
            rng,
            family=family,
            peak_range=(0.12, 0.20),
            offset_range=(0.105, 0.130),
            post_contact_range=(1, 2),
        )
    if family == "impulse_tap":
        return run_event_pulse(
            recorder,
            repetition,
            rng,
            family=family,
            peak_range=(0.22, 0.35),
            offset_range=(0.130, 0.155),
            post_contact_range=(2, 3),
        )
    if family == "strong_ram":
        return run_event_pulse(
            recorder,
            repetition,
            rng,
            family=family,
            peak_range=(0.38, 0.50),
            offset_range=(0.150, 0.170),
            post_contact_range=(2, 3),
        )
    if family == "position_push":
        return run_position_push(recorder, repetition, rng)
    if family == "contact_sweep_press":
        return run_contact_sweep_press(recorder, repetition, rng)
    if family == "direct_grasp_far_place":
        return run_direct_grasp_far_place(recorder, repetition, rng)
    if family == "push_then_grasp_far_place":
        return run_push_then_grasp_far_place(recorder, repetition, rng)
    if family == "grasp_carry_waypoint_place":
        return run_grasp_carry_waypoint_place(recorder, repetition, rng)
    raise KeyError(f"Unknown family: {family}")


def make_plan() -> list[dict[str, Any]]:
    plan = []
    for family_index, (family, count) in enumerate(FAMILY_COUNTS.items()):
        for repetition in range(int(count)):
            plan.append(
                {
                    "family": family,
                    "family_index": family_index,
                    "family_repetition": repetition,
                    "behavior_group": GROUP_BY_FAMILY[family],
                    "parameter_seed": BASE_SEED + family_index * 1000 + repetition,
                }
            )
    rng = np.random.default_rng(BASE_SEED)
    rng.shuffle(plan)
    return plan


def workspace_cell(xyz: np.ndarray) -> int:
    x_edges = np.linspace(WORKSPACE_X[0], WORKSPACE_X[1], X_CELLS + 1)
    y_edges = np.linspace(WORKSPACE_Y[0], WORKSPACE_Y[1], Y_CELLS + 1)
    ix = int(np.clip(np.searchsorted(x_edges, float(xyz[0]), side="right") - 1, 0, X_CELLS - 1))
    iy = int(np.clip(np.searchsorted(y_edges, float(xyz[1]), side="right") - 1, 0, Y_CELLS - 1))
    return iy * X_CELLS + ix


def extended_summary(recorder: Any) -> dict[str, Any]:
    summary = recorder.summary()
    eef_positions = [np.asarray(row["eef_xyz_t"], dtype=np.float64) for row in recorder.rows]
    eef_positions.append(current_eef(recorder))
    eef_array = np.asarray(eef_positions, dtype=np.float64)
    box_positions = [np.asarray(row["box_xyz_tp1"], dtype=np.float64) for row in recorder.rows]
    box_array = np.asarray(box_positions, dtype=np.float64) if box_positions else np.asarray([current_box(recorder)])
    actions = np.asarray([row["action_env"] for row in recorder.rows], dtype=np.float64)
    cells = sorted({workspace_cell(xyz) for xyz in eef_array})
    summary.update(
        {
            "eef_xyz_min_m": np.min(eef_array, axis=0),
            "eef_xyz_max_m": np.max(eef_array, axis=0),
            "eef_xyz_span_m": np.ptp(eef_array, axis=0),
            "workspace_xy_cells_visited": cells,
            "workspace_xy_cell_count": len(cells),
            "max_box_z_m": float(np.max(box_array[:, 2])),
            "nonzero_rotation_action_steps": int(np.sum(np.linalg.norm(actions[:, 3:6], axis=1) > 1e-6)),
            "gripper_command_transition_count": int(np.sum(np.abs(np.diff(actions[:, -1])) > 1e-6)) if len(actions) > 1 else 0,
        }
    )
    return summary


def observable_interaction(recorder: Any) -> bool:
    return bool(
        any(bool(row["robot_box_contact"]) for row in recorder.rows)
        or any(bool(row["robosuite_grasping"]) for row in recorder.rows)
        or any(float(np.linalg.norm(row["box_xyz_tp1"][:2] - recorder.initial_box_xyz[:2])) > 0.001 for row in recorder.rows)
        or any(float(np.linalg.norm(row["box_qvel_tp1"][:2])) > 0.03 for row in recorder.rows)
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if sum(FAMILY_COUNTS.values()) != EXPECTED_EPISODES:
        raise RuntimeError(f"Family counts sum to {sum(FAMILY_COUNTS.values())}, expected {EXPECTED_EPISODES}")
    if set(FAMILY_COUNTS) != set(TASK_PROMPTS) or set(FAMILY_COUNTS) != set(GROUP_BY_FAMILY):
        raise RuntimeError("Family, behavior-group, and task-prompt definitions do not match")

    base.patch_lerobot_video_crf(demo.VIDEO_CRF)
    dataset = base.create_dataset(
        output,
        repo_id="libero_mu0100_workspace_rich_eef_300eps_hai_machine",
    )
    bddl_file = demo.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="workspace_rich_eef_mu0100_hidden",
        init_xy=demo.INIT_XY,
        target_xy=demo.TARGET_XY,
        init_half_size=0.002,
        target_radius=0.025,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )
    case = demo.build_demo_case(bddl_file)
    plan = make_plan()

    episode_rows: list[dict[str, Any]] = []
    created_at = dt.datetime.now().isoformat()
    metadata = {
        "created_at": created_at,
        "dataset_type": "libero_mu0100_workspace_rich_eef_300eps_lerobot_2026-07-16_hai-machine",
        "purpose": "broad-workspace action-conditioned world-model and instruction-following training",
        "data_relationship": "new collection from scratch; does not append, copy, or mix the prior 200 EEF or 100 joint-position episodes",
        "episode_count_expected": EXPECTED_EPISODES,
        "friction_mu": demo.FRICTION_MU,
        "target_visible": False,
        "camera_resolution": demo.CAMERA_RESOLUTION,
        "fps": demo.FPS,
        "video_crf": demo.VIDEO_CRF,
        "controller": "OSC_POSE",
        "controller_scale": demo.CONTROLLER_SCALE,
        "sim_seed_fixed_for_same_scene": SIM_SEED,
        "action_names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"],
        "action_semantics": "normalized OSC_POSE relative command for translation and axis-angle rotation; gripper converted to open=1 closed=0 in LeRobot",
        "alignment": "observation_t and action_t are recorded before env.step(action_t), followed by observation_tp1; native 20 Hz is preserved and one terminal observation is appended",
        "task_prompt_policy": "four semantically equivalent long-horizon family-specific language instructions are attached to every frame; numeric trajectory parameters remain in episode metadata",
        "long_episode_policy": "workspace-exploration episodes use two broad prelude poses, a primary path, and two broad epilogue poses; contact and grasp episodes execute the primary interaction from the stable reset configuration, then use four broad epilogue poses because broad pre-motion can enter an OSC local-IK branch that cannot return to the object",
        "composition_quality_gate": "push-then-grasp uses a centered gentle +x push and rejects box yaw changes above 7 degrees before grasp; broad directional impacts are isolated in contact families",
        "workspace_bounds_m": {"x": WORKSPACE_X, "y": WORKSPACE_Y, "free_z": FREE_Z, "safe_place_x": PLACE_X, "safe_place_y": PLACE_Y},
        "workspace_stratification": {"x_cells": X_CELLS, "y_cells": Y_CELLS, "z_bands": Z_BANDS},
        "family_counts": FAMILY_COUNTS,
        "behavior_group_counts": dict(Counter(GROUP_BY_FAMILY[family] for family, count in FAMILY_COUNTS.items() for _ in range(count))),
        "task_prompts": TASK_PROMPTS,
        "design_references": DESIGN_REFERENCES,
        "episodes": episode_rows,
    }
    manifest = {
        "created_at": created_at,
        "output": str(output),
        "expected_episodes": EXPECTED_EPISODES,
        "controller": "OSC_POSE",
        "friction_mu": demo.FRICTION_MU,
        "family_counts": FAMILY_COUNTS,
        "behavior_group_counts": metadata["behavior_group_counts"],
        "episodes": episode_rows,
    }

    def autosave() -> None:
        write_json(output / "collection_manifest.json", manifest)
        base.write_dataset_metadata(output, base.to_jsonable(metadata), base.to_jsonable(episode_rows))

    autosave()
    for collection_index, item in enumerate(plan):
        family = str(item["family"])
        repetition = int(item["family_repetition"])
        behavior_group = str(item["behavior_group"])
        parameter_seed = int(item["parameter_seed"])
        rng = np.random.default_rng(parameter_seed)
        instructions = episode_task_prompts(family)
        source.TASK_PROMPT = instructions
        env = demo.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=SIM_SEED)
        try:
            recorder = source.LeRobotEpisodeRecorder(env=env, dataset=dataset)
            if behavior_group == "workspace_exploration":
                prelude = run_workspace_context_segment(
                    recorder,
                    repetition,
                    rng,
                    prefix="episode_prelude",
                    count=2,
                )
                epilogue_count = 2
            else:
                prelude = {
                    "policy": "primary interaction starts from the stable reset configuration",
                    "movements": [],
                }
                epilogue_count = 4
            primary = run_family(recorder, family, repetition, rng)
            epilogue = run_workspace_context_segment(
                recorder,
                repetition,
                rng,
                prefix="episode_epilogue",
                count=epilogue_count,
            )
            parameters = {
                "episode_prelude": prelude,
                "primary_behavior": primary,
                "episode_epilogue": epilogue,
            }
            if behavior_group in {"object_contact", "grasp_and_transport"} and not observable_interaction(recorder):
                raise RuntimeError(f"No observable object interaction for family={family} repetition={repetition}")
            recorder.add_terminal_observation()
            episode_index = int(dataset.meta.total_episodes)
            metrics = extended_summary(recorder)
            dataset.save_episode()
        finally:
            env.close()

        row = {
            "episode_index": episode_index,
            "collection_index": collection_index,
            "family": family,
            "family_repetition": repetition,
            "behavior_group": behavior_group,
            "language_instructions": instructions,
            "parameter_seed": parameter_seed,
            "sim_seed": SIM_SEED,
            "friction_mu": demo.FRICTION_MU,
            "target_visible": False,
            "parameters": parameters,
            "metrics": metrics,
        }
        episode_rows.append(row)
        print(
            f"[{collection_index + 1:03d}/{EXPECTED_EPISODES:03d}] ep={episode_index:03d} "
            f"group={behavior_group} family={family} frames={metrics['frames_in_lerobot_episode']} "
            f"span=({metrics['eef_xyz_span_m'][0]:.2f},{metrics['eef_xyz_span_m'][1]:.2f},{metrics['eef_xyz_span_m'][2]:.2f})m "
            f"box={metrics['final_box_displacement_m'] * 100.0:.1f}cm "
            f"contact={metrics['robot_box_contact_steps']} grasp={metrics['robosuite_grasping_steps']}",
            flush=True,
        )
        autosave()

    actual_eef_min = np.min(np.asarray([row["metrics"]["eef_xyz_min_m"] for row in episode_rows]), axis=0)
    actual_eef_max = np.max(np.asarray([row["metrics"]["eef_xyz_max_m"] for row in episode_rows]), axis=0)
    all_cells = sorted({cell for row in episode_rows for cell in row["metrics"]["workspace_xy_cells_visited"]})
    summary = {
        "completed_at": dt.datetime.now().isoformat(),
        "output": str(output),
        "episode_count": len(episode_rows),
        "expected_episode_count": EXPECTED_EPISODES,
        "controller": "OSC_POSE",
        "action_dim": 7,
        "count_by_family": dict(Counter(row["family"] for row in episode_rows)),
        "count_by_behavior_group": dict(Counter(row["behavior_group"] for row in episode_rows)),
        "friction_mu": demo.FRICTION_MU,
        "target_visible": False,
        "total_lerobot_frames": int(sum(row["metrics"]["frames_in_lerobot_episode"] for row in episode_rows)),
        "actual_dataset_eef_xyz_min_m": actual_eef_min,
        "actual_dataset_eef_xyz_max_m": actual_eef_max,
        "actual_dataset_eef_xyz_span_m": actual_eef_max - actual_eef_min,
        "workspace_xy_cells_visited_union": all_cells,
        "workspace_xy_cell_coverage": f"{len(all_cells)}/{X_CELLS * Y_CELLS}",
        "episodes_with_rotation_actions": int(sum(row["metrics"]["nonzero_rotation_action_steps"] > 0 for row in episode_rows)),
        "episodes_with_grasp_observed": int(sum(row["metrics"]["robosuite_grasping_steps"] > 0 for row in episode_rows)),
    }
    write_json(output / "collection_summary.json", summary)
    autosave()
    print(json.dumps(base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

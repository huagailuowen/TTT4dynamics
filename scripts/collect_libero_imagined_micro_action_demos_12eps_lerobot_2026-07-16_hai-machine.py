#!/usr/bin/env python3
"""Collect one normal-quality LeRobot demo for each imagined-tool primitive."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
FORMAL_SCRIPT = HERE / "collect_libero_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine.py"
DEFAULT_GRAPH = (
    HERE.parent
    / "data/various_actions/calibration/eef_pose_transition_graph_2026-07-16_hai-machine.json"
)
DEFAULT_OUTPUT = (
    HERE.parent
    / "data/various_actions/demos/"
    "libero_mu0100_imagined_micro_actions_12eps_lerobot_2026-07-16_hai-machine"
)
DEMO_FAMILY = "imagined_tool_micro_action_demo"
DEMO_PROMPTS = [
    "Perform one clear imagined-tool manipulation motion at a safe pose.",
    "Move to a verified pose and demonstrate one common manipulation primitive.",
    "Execute one visible task-shaped end-effector motion without a physical tool.",
    "Demonstrate a single calibrated micro-action and safely return home.",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


formal = load_module(FORMAL_SCRIPT, "imagined_micro_demo_formal_base")
rich = formal.rich
calibration = formal.calibration


def make_plan() -> list[dict[str, Any]]:
    return [
        {
            "family": DEMO_FAMILY,
            "family_index": 0,
            "family_repetition": repetition,
            "behavior_group": "imagined_tool_micro_action_demo",
            "parameter_seed": 20260716 + repetition,
        }
        for repetition in range(len(formal.MICRO_PRIMITIVES))
    ]


def run_demo(recorder: Any, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    primitive = formal.MICRO_PRIMITIVES[repetition]
    trace = calibration.SafetyTrace()
    home_xyz = rich.current_eef(recorder)
    home_orientation = calibration.current_orientation(recorder)
    case = formal.select_micro_case(primitive, repetition, 0)
    anchor, tool_orientation = formal.pose_for_case(case, home_orientation)

    approach = formal.checked_move(
        recorder,
        anchor,
        tool_orientation,
        phase=f"demo_{primitive}_approach",
        trace=trace,
    )
    recorder.hold(steps=8, gripper=-1.0, phase=f"demo_{primitive}_ready")
    trace.sample(recorder)

    positions, orientations, grippers = formal.primitive_path(
        primitive, anchor, tool_orientation, rng
    )
    motion = formal.follow_micro_path(
        recorder,
        positions,
        orientations,
        grippers,
        phase=f"demo_{primitive}",
        trace=trace,
    )
    settle = formal.checked_move(
        recorder,
        anchor,
        tool_orientation,
        phase=f"demo_{primitive}_settle",
        trace=trace,
    )
    recorder.hold(steps=8, gripper=-1.0, phase=f"demo_{primitive}_finished")
    trace.sample(recorder)
    returned = formal.checked_move(
        recorder,
        home_xyz,
        home_orientation,
        phase=f"demo_{primitive}_return",
        trace=trace,
        returning_home=True,
    )

    if motion["actual_eef_span_norm_m"] < 0.010 and motion["actual_orientation_span_deg"] < 3.0:
        raise RuntimeError(f"Demo did not visibly move: {primitive}")
    formal.require_safe_trace(trace, DEMO_FAMILY)
    return {
        "demo_index": repetition,
        "primitive": primitive,
        "validated_case_id": case["id"],
        "tool_orientation_preset": case["orientation"],
        "imagined_tool_only": True,
        "visible_tool_asset": False,
        "approach": approach,
        "motion": motion,
        "settle": settle,
        "return": returned,
        "safety": trace.as_dict(),
        "runtime_gate_pass": True,
    }


def run_family(recorder: Any, family: str, repetition: int, rng: np.random.Generator) -> dict[str, Any]:
    if family != DEMO_FAMILY:
        raise ValueError(family)
    return run_demo(recorder, repetition, rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formal.load_validated_graph(args.graph)
    count = len(formal.MICRO_PRIMITIVES)
    rich.FAMILY_COUNTS = {DEMO_FAMILY: count}
    rich.EXPECTED_EPISODES = count
    rich.GROUP_BY_FAMILY = {DEMO_FAMILY: "imagined_tool_micro_action_demo"}
    rich.TASK_PROMPTS = {DEMO_FAMILY: DEMO_PROMPTS}
    rich.make_plan = make_plan
    rich.run_family = run_family
    rich.episode_task_prompts = lambda family: DEMO_PROMPTS
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

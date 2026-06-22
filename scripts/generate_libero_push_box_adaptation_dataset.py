#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt4dynamics.push_box_libero import LiberoPushBoxCase, LiberoPushBoxEnv  # noqa: E402


GEOMETRY_TEMPLATES = [
    ("g00", (-0.245, -0.035), 0.265),
    ("g01", (-0.255, -0.015), 0.275),
    ("g02", (-0.235, 0.010), 0.255),
    ("g03", (-0.250, 0.035), 0.280),
    ("g04", (-0.225, -0.005), 0.245),
    ("g05", (-0.260, 0.025), 0.290),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and calibrate a LIBERO push-box source/adaptation dataset."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs" / "libero_push_box_adaptation_dataset.json",
    )
    parser.add_argument(
        "--bddl-dir",
        type=Path,
        default=REPO_ROOT / "generated_bddl" / "push_box_adaptation_dataset",
    )
    parser.add_argument("--source-friction", type=float, default=0.006)
    parser.add_argument("--source-count", type=int, default=5)
    parser.add_argument("--adapt-count-per-group", type=int, default=5)
    parser.add_argument("--adapt-friction-min", type=float, default=0.0)
    parser.add_argument("--adapt-friction-max", type=float, default=0.2)
    parser.add_argument("--adapt-friction-count", type=int, default=21)
    parser.add_argument(
        "--adapt-frictions",
        type=float,
        nargs="+",
        help="Explicit adaptation friction values. Overrides min/max/count.",
    )
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--push-distance-x", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--calibration-resolution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--successes-before-stop", type=int, default=3)
    parser.add_argument(
        "--max-calibration-trials",
        type=int,
        default=80,
        help="Maximum push-speed candidates to try per case before keeping the nearest result.",
    )
    parser.add_argument("--allow-unsolved", action="store_true")
    return parser.parse_args()


def format_mu_tag(friction_mu: float) -> str:
    return f"mu{int(round(float(friction_mu) * 10000)):04d}"


def adapt_friction_groups(args: argparse.Namespace) -> list[tuple[str, float]]:
    if args.adapt_frictions:
        values = [float(mu) for mu in args.adapt_frictions]
    else:
        if int(args.adapt_friction_count) <= 0:
            raise ValueError("--adapt-friction-count must be positive.")
        lo = float(args.adapt_friction_min)
        hi = float(args.adapt_friction_max)
        if hi < lo:
            raise ValueError("--adapt-friction-max must be >= --adapt-friction-min.")
        values = np.linspace(lo, hi, int(args.adapt_friction_count), dtype=np.float64).tolist()

    groups: list[tuple[str, float]] = []
    seen: set[str] = set()
    for mu in values:
        group = f"adapt_{format_mu_tag(mu)}"
        if group in seen:
            raise ValueError(f"Duplicate adaptation friction group after rounding: {group}")
        seen.add(group)
        groups.append((group, float(mu)))
    return groups


def _region_bounds(center: tuple[float, float], half_size: float) -> tuple[float, float, float, float]:
    x, y = center
    return x - half_size, y - half_size, x + half_size, y + half_size


def _bddl_text(
    *,
    init_xy: tuple[float, float],
    target_xy: tuple[float, float],
    init_half_size: float,
    target_radius: float,
) -> str:
    ix0, iy0, ix1, iy1 = _region_bounds(init_xy, init_half_size)
    tx0, ty0, tx1, ty1 = _region_bounds(target_xy, target_radius)
    return f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language push the cream cheese box across the smooth table into the green target region)
    (:regions
      (box_init_region
          (:target main_table)
          (:ranges (
              ({ix0:.4f} {iy0:.4f} {ix1:.4f} {iy1:.4f})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (target_region
          (:target main_table)
          (:ranges (
              ({tx0:.4f} {ty0:.4f} {tx1:.4f} {ty1:.4f})
            )
          )
          (:rgba
              (0.0 0.8 0.2 0.45)
          )
      )
    )

  (:fixtures
    main_table - table
  )

  (:objects
    cream_cheese_1 - cream_cheese
  )

  (:obj_of_interest
    cream_cheese_1
    main_table_target_region
  )

  (:init
    (On cream_cheese_1 main_table_box_init_region)
  )

  (:goal
    (And (On cream_cheese_1 main_table_target_region))
  )

)
"""


def write_geometry_bddl(
    *,
    repo_root: Path,
    bddl_dir: Path,
    geometry_id: str,
    init_xy: tuple[float, float],
    target_xy: tuple[float, float],
    init_half_size: float,
    target_radius: float,
) -> str:
    bddl_dir.mkdir(parents=True, exist_ok=True)
    path = bddl_dir / f"push_box_{geometry_id}.bddl"
    path.write_text(
        _bddl_text(
            init_xy=init_xy,
            target_xy=target_xy,
            init_half_size=init_half_size,
            target_radius=target_radius,
        ),
        encoding="utf-8",
    )
    return str(path.resolve().relative_to(repo_root.resolve()))


def candidate_push_settings(friction_mu: float) -> list[tuple[int, float, float]]:
    common_action_ends = [0.5, 0.6, 0.8, 1.0]
    fast_action_ends = list(reversed(common_action_ends))
    if friction_mu <= 0.005:
        seeds = [(steps, action_end, 1.0) for steps in (20, 22, 24, 26, 30) for action_end in common_action_ends]
    elif friction_mu <= 0.02:
        seeds = [(steps, action_end, 1.0) for steps in (16, 18, 20, 22, 24) for action_end in common_action_ends]
    elif friction_mu <= 0.08:
        seeds = [
            (steps, action_end, scale)
            for steps in (6, 8, 10, 12, 14, 16, 18)
            for scale in (14.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
            for action_end in fast_action_ends
        ]
    else:
        seeds = [
            (steps, action_end, scale)
            for steps in (4, 5, 6, 8, 10, 12, 14, 16)
            for scale in (20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0)
            for action_end in fast_action_ends
        ]

    expanded = seeds[:]
    fallback_action_ends = fast_action_ends if friction_mu > 0.02 else common_action_ends
    fallback_scales = (
        20.0,
        18.0,
        16.0,
        14.0,
        12.0,
        10.0,
        8.0,
        6.0,
        5.0,
        4.0,
        3.0,
        2.0,
        1.0,
    ) if friction_mu > 0.02 else (1.0,)
    for push_steps in (4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30):
        expanded.extend((push_steps, action_end, scale) for scale in fallback_scales for action_end in fallback_action_ends)

    seen: set[tuple[int, float, float]] = set()
    ordered = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def rollout_case(case: LiberoPushBoxCase, *, repo_root: Path, seed: int) -> dict[str, Any]:
    env = LiberoPushBoxEnv(case, repo_root=repo_root, seed=seed)
    try:
        env.reset()
        records = [env.step_info().as_dict()]
        for _ in range(int(case.max_steps)):
            _, _, _, info = env.step()
            records.append(info["push_box"])
        final = records[-1]
        return {
            "success": bool(final["success"]),
            "final": final,
            "min_distance_to_target": float(min(record["distance_to_target"] for record in records)),
        }
    finally:
        env.close()


def calibrate_case(
    case: LiberoPushBoxCase,
    *,
    repo_root: Path,
    seed: int,
    calibration_resolution: int,
    successes_before_stop: int,
    max_calibration_trials: int,
) -> LiberoPushBoxCase | None:
    best_success: tuple[float, LiberoPushBoxCase, dict[str, Any]] | None = None
    best_any: tuple[float, LiberoPushBoxCase, dict[str, Any]] | None = None
    successes = 0
    candidates = candidate_push_settings(case.friction_mu)
    if int(max_calibration_trials) > 0:
        candidates = candidates[: int(max_calibration_trials)]
    for push_steps, action_end, push_controller_scale in candidates:
        trial = replace(
            case,
            camera_resolution=int(calibration_resolution),
            pusher_push_steps=int(push_steps),
            pusher_push_action_end=float(action_end),
            pusher_max_pos_action=1.0,
            pusher_push_controller_scale=float(push_controller_scale),
            pusher_push_controller_scale_ramp_steps=2,
            controller_output_scale=1.0,
            enable_controller_output_scaling=False,
        )
        result = rollout_case(trial, repo_root=repo_root, seed=seed)
        final_distance = float(result["final"]["distance_to_target"])
        if best_any is None or final_distance < best_any[0]:
            best_any = (final_distance, trial, result)
        if result["success"]:
            successes += 1
            if best_success is None or final_distance < best_success[0]:
                best_success = (final_distance, trial, result)
            if successes >= max(1, int(successes_before_stop)):
                break

    selected = best_success if best_success is not None else best_any
    if selected is None:
        return None

    _, trial, result = selected
    final_case = replace(
        trial,
        camera_resolution=int(case.camera_resolution),
        calibration={
            "success": bool(result["success"]),
            "selected_nearest_without_success": bool(not result["success"]),
            "final_distance_m": float(result["final"]["distance_to_target"]),
            "min_distance_m": float(result["min_distance_to_target"]),
            "final_xy": [float(result["final"]["box_xyz"][0]), float(result["final"]["box_xyz"][1])],
            "push_distance_x_m": float(trial.pusher_push_distance_x),
            "pusher_push_steps": int(trial.pusher_push_steps),
            "pusher_push_action_end": float(trial.pusher_push_action_end),
            "pusher_push_controller_scale": float(trial.pusher_push_controller_scale),
            "pusher_push_controller_scale_ramp_steps": int(trial.pusher_push_controller_scale_ramp_steps),
            "controller_output_scale": float(trial.controller_output_scale),
            "enable_controller_output_scaling": bool(trial.enable_controller_output_scaling),
        },
    )
    return final_case


def build_case(
    *,
    case_id: str,
    domain: str,
    friction_group: str,
    friction_mu: float,
    geometry_id: str,
    init_xy: tuple[float, float],
    target_distance: float,
    bddl_file: str,
    target_radius: float,
    push_distance_x: float,
    max_steps: int,
    camera_resolution: int,
) -> LiberoPushBoxCase:
    target_xy = (init_xy[0] + target_distance, init_xy[1])
    return LiberoPushBoxCase(
        case_id=case_id,
        friction_mu=float(friction_mu),
        domain=domain,
        friction_group=friction_group,
        geometry_id=geometry_id,
        init_xy=init_xy,
        target_xy=target_xy,
        target_distance=float(target_distance),
        bddl_file=bddl_file,
        target_radius=float(target_radius),
        pusher_push_distance_x=float(push_distance_x),
        max_steps=int(max_steps),
        camera_resolution=int(camera_resolution),
        pusher_approach_steps=25,
        pusher_descend_steps=35,
        pusher_retreat_steps=60,
        pusher_settle_steps=120,
        pusher_push_accel_steps=6,
        pusher_push_profile="smootherstep",
        pusher_push_mode="impulse",
        pusher_push_action_end=1.0,
        pusher_max_pos_action=1.0,
        pusher_push_controller_scale=1.0,
        pusher_push_controller_scale_ramp_steps=2,
        controller_output_scale=1.0,
        enable_controller_output_scaling=False,
    )


def select_templates(count: int) -> list[tuple[str, tuple[float, float], float]]:
    if count > len(GEOMETRY_TEMPLATES):
        raise ValueError(f"Requested {count} geometries, but only {len(GEOMETRY_TEMPLATES)} are defined.")
    return GEOMETRY_TEMPLATES[:count]


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    bddl_dir = args.bddl_dir if args.bddl_dir.is_absolute() else repo_root / args.bddl_dir
    friction_groups = adapt_friction_groups(args)
    source_templates = select_templates(int(args.source_count))
    adapt_templates = select_templates(int(args.adapt_count_per_group))
    all_templates = {item[0]: item for item in source_templates + adapt_templates}

    bddl_by_geometry: dict[str, str] = {}
    for geometry_id, init_xy, target_distance in all_templates.values():
        target_xy = (init_xy[0] + target_distance, init_xy[1])
        bddl_by_geometry[geometry_id] = write_geometry_bddl(
            repo_root=repo_root,
            bddl_dir=bddl_dir,
            geometry_id=geometry_id,
            init_xy=init_xy,
            target_xy=target_xy,
            init_half_size=float(args.init_half_size),
            target_radius=float(args.target_radius),
        )

    raw_cases: list[LiberoPushBoxCase] = []
    for idx, (geometry_id, init_xy, target_distance) in enumerate(source_templates):
        raw_cases.append(
            build_case(
                case_id=f"source_{format_mu_tag(args.source_friction)}_{geometry_id}",
                domain="source",
                friction_group=f"source_mu{args.source_friction:.4f}",
                friction_mu=float(args.source_friction),
                geometry_id=geometry_id,
                init_xy=init_xy,
                target_distance=target_distance,
                bddl_file=bddl_by_geometry[geometry_id],
                target_radius=float(args.target_radius),
                push_distance_x=float(args.push_distance_x),
                max_steps=int(args.max_steps),
                camera_resolution=int(args.camera_resolution),
            )
        )

    for group_name, friction_mu in friction_groups:
        for idx, (geometry_id, init_xy, target_distance) in enumerate(adapt_templates):
            raw_cases.append(
                build_case(
                    case_id=f"{group_name}_{geometry_id}",
                    domain="adapt",
                    friction_group=group_name,
                    friction_mu=float(friction_mu),
                    geometry_id=geometry_id,
                    init_xy=init_xy,
                    target_distance=target_distance,
                    bddl_file=bddl_by_geometry[geometry_id],
                    target_radius=float(args.target_radius),
                    push_distance_x=float(args.push_distance_x),
                    max_steps=int(args.max_steps),
                    camera_resolution=int(args.camera_resolution),
                )
            )

    calibrated: list[LiberoPushBoxCase] = []
    failures: list[dict[str, Any]] = []
    for idx, case in enumerate(raw_cases):
        print(
            f"[{idx + 1}/{len(raw_cases)}] calibrating {case.case_id} "
            f"mu={case.friction_mu:.4f} geom={case.geometry_id} "
            f"dist={case.target_distance:.3f} stroke={case.pusher_push_distance_x:.3f}",
            flush=True,
        )
        solved = calibrate_case(
            case,
            repo_root=repo_root,
            seed=int(args.seed),
            calibration_resolution=int(args.calibration_resolution),
            successes_before_stop=int(args.successes_before_stop),
            max_calibration_trials=int(args.max_calibration_trials),
        )
        if solved is None:
            failures.append(case.as_dict())
            print(f"  FAILED {case.case_id}", flush=True)
            if not args.allow_unsolved:
                raise RuntimeError(f"Could not calibrate {case.case_id}; rerun with a narrower geometry/friction range.")
            continue
        calibrated.append(solved)
        cal = solved.calibration or {}
        status = "nearest" if cal.get("selected_nearest_without_success") else "solved"
        print(
            f"  {status} dist={float(cal['final_distance_m']) * 100:.2f}cm "
            f"push_action_end={solved.pusher_push_action_end:.2f} "
            f"push_scale={solved.pusher_push_controller_scale:.1f} "
            f"push_steps={solved.pusher_push_steps}",
            flush=True,
        )

    source_ids = [case.case_id for case in calibrated if case.domain == "source"]
    adapt_ids_by_group: dict[str, list[str]] = {}
    for group_name, _ in friction_groups:
        adapt_ids_by_group[group_name] = [
            case.case_id for case in calibrated if case.domain == "adapt" and case.friction_group == group_name
        ]

    payload = {
        "description": "LIBERO short-stroke push-box dataset with one fixed source friction and broad adaptation friction groups.",
        "fixed_constraints": {
            "pusher_push_distance_x_m": float(args.push_distance_x),
            "short_push_then_free_slide": True,
            "target_radius_m": float(args.target_radius),
            "source_friction_mu": float(args.source_friction),
            "adapt_friction_range": [
                float(min(mu for _, mu in friction_groups)),
                float(max(mu for _, mu in friction_groups)),
            ],
            "max_calibration_trials": int(args.max_calibration_trials),
            "record_nearest_when_unsolved": True,
            "adapt_friction_groups": [
                {"group": group_name, "friction_mu": friction_mu} for group_name, friction_mu in friction_groups
            ],
        },
        "splits": {
            "source": source_ids,
            "adapt": adapt_ids_by_group,
        },
        "failures": failures,
        "near_misses": [
            case.case_id
            for case in calibrated
            if case.calibration and case.calibration.get("selected_nearest_without_success")
        ],
        "cases": [case.as_dict() for case in calibrated],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(calibrated)} cases to {args.output}")
    if failures:
        print(f"unsolved cases kept out of dataset: {len(failures)}")


if __name__ == "__main__":
    main()

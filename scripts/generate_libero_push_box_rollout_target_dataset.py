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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_libero_push_box_adaptation_dataset import write_geometry_bddl  # noqa: E402
from ttt4dynamics.push_box_libero import LiberoPushBoxCase, LiberoPushBoxEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LIBERO push-box cases by rolling out pushes first and using the final resting point as target."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs" / "libero_push_box_rollout_target_dataset.json",
    )
    parser.add_argument(
        "--bddl-dir",
        type=Path,
        default=REPO_ROOT / "generated_bddl" / "push_box_rollout_target_dataset",
    )
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--straight-angles", type=float, nargs="+", default=[0.0])
    parser.add_argument("--angled-angles", type=float, nargs="+", default=[-30.0, -20.0, -10.0, 10.0, 20.0, 30.0])
    parser.add_argument("--push-steps", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28])
    parser.add_argument("--push-distances", type=float, nargs="+", default=[0.12, 0.14, 0.16, 0.18])
    parser.add_argument("--init-xy", type=float, nargs=2, default=(-0.245, -0.035))
    parser.add_argument("--dummy-target-distance", type=float, default=0.265)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--probe-resolution", type=int, default=24)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0, help="Stop after this many accepted rollout pairs. 0 keeps all.")
    parser.add_argument("--min-displacement", type=float, default=0.06)
    parser.add_argument("--max-displacement", type=float, default=0.34)
    parser.add_argument("--max-final-speed", type=float, default=0.02)
    parser.add_argument("--x-bounds", type=float, nargs=2, default=(-0.20, 0.09))
    parser.add_argument("--y-bounds", type=float, nargs=2, default=(-0.13, 0.09))
    return parser.parse_args()


def mu_tag(friction_mu: float) -> str:
    return f"mu{int(round(float(friction_mu) * 10000)):04d}"


def angle_tag(angle_deg: float) -> str:
    signed = int(round(float(angle_deg)))
    prefix = "p" if signed >= 0 else "m"
    return f"{prefix}{abs(signed):02d}"


def direction_xy(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)


def build_probe_case(
    *,
    case_id: str,
    friction_mu: float,
    split: str,
    angle_deg: float,
    push_steps: int,
    push_distance: float,
    init_xy: tuple[float, float],
    target_xy: tuple[float, float],
    bddl_file: str,
    max_steps: int,
    camera_resolution: int,
    target_radius: float,
) -> LiberoPushBoxCase:
    return LiberoPushBoxCase(
        case_id=case_id,
        friction_mu=float(friction_mu),
        domain="rollout_probe",
        friction_group=mu_tag(friction_mu),
        geometry_id=f"g00_{split}_{angle_tag(angle_deg)}",
        init_xy=init_xy,
        target_xy=target_xy,
        target_distance=float(np.linalg.norm(np.asarray(target_xy) - np.asarray(init_xy))),
        bddl_file=bddl_file,
        target_radius=float(target_radius),
        max_steps=int(max_steps),
        camera_resolution=int(camera_resolution),
        pusher_approach_steps=30,
        pusher_descend_steps=40,
        pusher_retreat_steps=60,
        pusher_settle_steps=max(120, int(max_steps) - 30 - 40 - int(push_steps) - 60),
        pusher_push_steps=int(push_steps),
        pusher_push_distance_x=float(push_distance),
        pusher_push_angle_deg=float(angle_deg),
        pusher_push_mode="position",
        pusher_push_profile="smootherstep",
        pusher_push_action_end=1.0,
        pusher_max_pos_action=1.0,
        pusher_push_action_delta=10.0,
        pusher_push_controller_scale=2.0,
        pusher_push_controller_scale_ramp_steps=2,
        controller_output_scale=1.0,
        enable_controller_output_scaling=False,
    )


def rollout(case: LiberoPushBoxCase, *, repo_root: Path, seed: int) -> dict[str, Any]:
    env = LiberoPushBoxEnv(case, repo_root=repo_root, seed=seed)
    records: list[dict[str, Any]] = []
    try:
        env.reset()
        initial = env.step_info().as_dict()
        initial["phase"] = "reset"
        initial["action"] = None
        records.append(initial)
        for _ in range(int(case.max_steps)):
            _, _, _, info = env.step()
            records.append(info["push_box"])
    finally:
        env.close()

    final = records[-1]
    push_actions = [float(r["action"][0]) for r in records if r.get("phase") == "push" and r.get("action")]
    eef = np.asarray([r["eef_xyz"] for r in records if r.get("phase") == "push" and r.get("eef_xyz")], dtype=np.float64)
    eef_dx = np.diff(eef[:, 0]) if len(eef) > 1 else np.zeros(0, dtype=np.float64)
    all_eef = np.asarray([r["eef_xyz"] for r in records if r.get("eef_xyz")], dtype=np.float64)
    all_eef_step = np.linalg.norm(np.diff(all_eef, axis=0), axis=1) if len(all_eef) > 1 else np.zeros(0)
    return {
        "records": records,
        "final_xy": [float(final["box_xyz"][0]), float(final["box_xyz"][1])],
        "final_speed": float(np.linalg.norm(np.asarray(final["box_vxy"], dtype=np.float64))),
        "push_backward_action_count": int(sum(1 for x in push_actions if x < -1e-6)),
        "push_eef_backward_x_steps": int(np.sum(eef_dx < -1e-4)),
        "max_eef_step_m": float(np.max(all_eef_step)) if all_eef_step.size else 0.0,
    }


def accept_rollout(
    *,
    init_xy: tuple[float, float],
    angle_deg: float,
    final_xy: list[float],
    final_speed: float,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    init = np.asarray(init_xy, dtype=np.float64)
    final = np.asarray(final_xy, dtype=np.float64)
    delta = final - init
    direction = direction_xy(angle_deg)
    forward = float(np.dot(delta, direction))
    lateral = float(direction[0] * delta[1] - direction[1] * delta[0])
    displacement = float(np.linalg.norm(delta))
    x_ok = float(args.x_bounds[0]) <= float(final[0]) <= float(args.x_bounds[1])
    y_ok = float(args.y_bounds[0]) <= float(final[1]) <= float(args.y_bounds[1])
    ok = (
        x_ok
        and y_ok
        and forward >= float(args.min_displacement)
        and displacement <= float(args.max_displacement)
        and float(final_speed) <= float(args.max_final_speed)
    )
    return ok, {
        "forward_m": forward,
        "lateral_m": lateral,
        "displacement_m": displacement,
        "final_speed_mps": float(final_speed),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    bddl_dir = args.bddl_dir if args.bddl_dir.is_absolute() else repo_root / args.bddl_dir
    init_xy = (float(args.init_xy[0]), float(args.init_xy[1]))
    dummy_target_xy = (init_xy[0] + float(args.dummy_target_distance), init_xy[1])
    probe_bddl = write_geometry_bddl(
        repo_root=repo_root,
        bddl_dir=bddl_dir / "probe",
        geometry_id="g00_probe_invisible",
        init_xy=init_xy,
        target_xy=dummy_target_xy,
        init_half_size=float(args.init_half_size),
        target_radius=float(args.target_radius),
        target_rgba=(0.0, 0.0, 0.0, 0.0),
    )

    cases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    angle_splits = [("straight", float(a)) for a in args.straight_angles] + [
        ("angled", float(a)) for a in args.angled_angles
    ]
    for friction_mu in [float(mu) for mu in args.frictions]:
        for split, angle_deg in angle_splits:
            for push_distance in [float(d) for d in args.push_distances]:
                for push_steps in [int(n) for n in args.push_steps]:
                    base_id = (
                        f"{split}_{mu_tag(friction_mu)}_{angle_tag(angle_deg)}_"
                        f"d{int(round(push_distance * 100)):02d}_n{push_steps:02d}"
                    )
                    probe = build_probe_case(
                        case_id=f"probe_{base_id}",
                        friction_mu=friction_mu,
                        split=split,
                        angle_deg=angle_deg,
                        push_steps=push_steps,
                        push_distance=push_distance,
                        init_xy=init_xy,
                        target_xy=dummy_target_xy,
                        bddl_file=probe_bddl,
                        max_steps=int(args.max_steps),
                        camera_resolution=int(args.probe_resolution),
                        target_radius=float(args.target_radius),
                    )
                    result = rollout(probe, repo_root=repo_root, seed=int(args.seed))
                    accepted, metrics = accept_rollout(
                        init_xy=init_xy,
                        angle_deg=angle_deg,
                        final_xy=result["final_xy"],
                        final_speed=float(result["final_speed"]),
                        args=args,
                    )
                    metrics.update(
                        {
                            "final_xy": result["final_xy"],
                            "push_backward_action_count": int(result["push_backward_action_count"]),
                            "push_eef_backward_x_steps": int(result["push_eef_backward_x_steps"]),
                            "max_eef_step_m": float(result["max_eef_step_m"]),
                        }
                    )
                    if not accepted:
                        rejected.append({"case_id": probe.case_id, "friction_mu": friction_mu, **metrics})
                        continue

                    visible_bddl = write_geometry_bddl(
                        repo_root=repo_root,
                        bddl_dir=bddl_dir / "task",
                        geometry_id=base_id,
                        init_xy=init_xy,
                        target_xy=(float(result["final_xy"][0]), float(result["final_xy"][1])),
                        init_half_size=float(args.init_half_size),
                        target_radius=float(args.target_radius),
                        target_rgba=(0.0, 0.8, 0.2, 0.45),
                    )
                    invisible_bddl = write_geometry_bddl(
                        repo_root=repo_root,
                        bddl_dir=bddl_dir / "observation",
                        geometry_id=base_id,
                        init_xy=init_xy,
                        target_xy=(float(result["final_xy"][0]), float(result["final_xy"][1])),
                        init_half_size=float(args.init_half_size),
                        target_radius=float(args.target_radius),
                        target_rgba=(0.0, 0.0, 0.0, 0.0),
                    )
                    task_case = replace(
                        probe,
                        case_id=f"task_{base_id}",
                        domain="task",
                        bddl_file=visible_bddl,
                        target_xy=(float(result["final_xy"][0]), float(result["final_xy"][1])),
                        target_distance=float(metrics["displacement_m"]),
                        camera_resolution=int(args.camera_resolution),
                    )
                    observation_case = replace(
                        task_case,
                        case_id=f"observation_{base_id}",
                        domain="observation",
                        bddl_file=invisible_bddl,
                    )
                    cases.extend([observation_case.as_dict(), task_case.as_dict()])
                    pairs.append(
                        {
                            "pair_id": base_id,
                            "split": split,
                            "friction_mu": friction_mu,
                            "angle_deg": angle_deg,
                            "push_distance_x": push_distance,
                            "pusher_push_steps": push_steps,
                            "observation_case_id": observation_case.case_id,
                            "task_case_id": task_case.case_id,
                            "target_xy": result["final_xy"],
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"accepted {base_id} target=({result['final_xy'][0]:.3f},{result['final_xy'][1]:.3f}) "
                        f"disp={metrics['displacement_m'] * 100:.1f}cm speed={metrics['final_speed_mps']:.4f}",
                        flush=True,
                    )
                    if int(args.max_cases) > 0 and len(pairs) >= int(args.max_cases):
                        break
                if int(args.max_cases) > 0 and len(pairs) >= int(args.max_cases):
                    break
            if int(args.max_cases) > 0 and len(pairs) >= int(args.max_cases):
                break
        if int(args.max_cases) > 0 and len(pairs) >= int(args.max_cases):
            break

    payload = {
        "description": "Rollout-first LIBERO push-box dataset. Each accepted rollout creates an observation case with an invisible target and a task case with a green target at the observed resting point.",
        "generation": {
            "init_xy": list(init_xy),
            "target_radius_m": float(args.target_radius),
            "frictions": [float(mu) for mu in args.frictions],
            "straight_angles_deg": [float(a) for a in args.straight_angles],
            "angled_angles_deg": [float(a) for a in args.angled_angles],
            "push_steps": [int(n) for n in args.push_steps],
            "push_distances_m": [float(d) for d in args.push_distances],
            "acceptance": {
                "min_displacement_m": float(args.min_displacement),
                "max_displacement_m": float(args.max_displacement),
                "max_final_speed_mps": float(args.max_final_speed),
                "x_bounds": [float(v) for v in args.x_bounds],
                "y_bounds": [float(v) for v in args.y_bounds],
            },
        },
        "pairs": pairs,
        "cases": cases,
        "rejected_count": len(rejected),
        "rejected_preview": rejected[:50],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(pairs)} pairs / {len(cases)} cases to {output}")
    print(f"rejected {len(rejected)} candidates")


if __name__ == "__main__":
    main()

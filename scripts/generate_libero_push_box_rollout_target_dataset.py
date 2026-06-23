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
    parser.add_argument("--push-scales", type=float, nargs="+", default=[2.0, 4.0, 8.0, 12.0])
    parser.add_argument("--init-xy", type=float, nargs=2, default=(-0.245, -0.035))
    parser.add_argument("--dummy-target-distance", type=float, default=0.265)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--probe-resolution", type=int, default=24)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0, help="Stop after this many accepted rollout pairs. 0 keeps all.")
    parser.add_argument(
        "--pairs-per-friction-split-speed-bin",
        type=int,
        default=0,
        help="Balanced mode: accepted rollout pairs per (friction, straight/angled split, speed bin).",
    )
    parser.add_argument(
        "--speed-bin-edges",
        type=float,
        nargs="+",
        default=[0.007, 0.012],
        help="Edges for push speed bins, measured as push_distance_x / pusher_push_steps.",
    )
    parser.add_argument("--no-shuffle-candidates", action="store_true")
    parser.add_argument("--autosave-every", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--max-trials-per-bucket",
        type=int,
        default=0,
        help="Balanced mode guardrail. 0 tries all candidates in each bucket.",
    )
    parser.add_argument("--min-displacement", type=float, default=0.06)
    parser.add_argument("--max-displacement", type=float, default=0.34)
    parser.add_argument("--max-final-speed", type=float, default=0.02)
    parser.add_argument("--max-eef-step", type=float, default=0.06)
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


def speed_bin(speed_m_per_step: float, edges: list[float]) -> str:
    sorted_edges = sorted(float(edge) for edge in edges)
    for idx, edge in enumerate(sorted_edges):
        if float(speed_m_per_step) < edge:
            return f"speed_{idx:02d}"
    return f"speed_{len(sorted_edges):02d}"


def build_probe_case(
    *,
    case_id: str,
    friction_mu: float,
    split: str,
    angle_deg: float,
    push_steps: int,
    push_distance: float,
    push_scale: float,
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
        pusher_push_controller_scale=float(push_scale),
        pusher_max_push_controller_scale=max(20.0, float(push_scale)),
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
    direction = direction_xy(float(case.pusher_push_angle_deg))
    push_actions = [
        float(np.dot(np.asarray(r["action"][:2], dtype=np.float64), direction))
        for r in records
        if r.get("phase") == "push" and r.get("action")
    ]
    eef = np.asarray([r["eef_xyz"] for r in records if r.get("phase") == "push" and r.get("eef_xyz")], dtype=np.float64)
    eef_forward = eef[:, :2] @ direction if len(eef) > 0 else np.zeros(0, dtype=np.float64)
    eef_dforward = np.diff(eef_forward) if len(eef_forward) > 1 else np.zeros(0, dtype=np.float64)
    all_eef = np.asarray([r["eef_xyz"] for r in records if r.get("eef_xyz")], dtype=np.float64)
    all_eef_step = np.linalg.norm(np.diff(all_eef, axis=0), axis=1) if len(all_eef) > 1 else np.zeros(0)
    return {
        "records": records,
        "final_xy": [float(final["box_xyz"][0]), float(final["box_xyz"][1])],
        "final_speed": float(np.linalg.norm(np.asarray(final["box_vxy"], dtype=np.float64))),
        "push_backward_action_count": int(sum(1 for x in push_actions if x < -1e-6)),
        "push_eef_backward_steps": int(np.sum(eef_dforward < -1e-4)),
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


def build_payload(
    *,
    args: argparse.Namespace,
    init_xy: tuple[float, float],
    pairs: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    target_buckets: dict[tuple[str, str, str], int],
    accepted_buckets: dict[tuple[str, str, str], int],
    trial_buckets: dict[tuple[str, str, str], int],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "description": "Rollout-first LIBERO push-box dataset. Each accepted rollout creates an observation case with an invisible target and a task case with a green target at the observed resting point.",
        "generation": {
            "init_xy": list(init_xy),
            "target_radius_m": float(args.target_radius),
            "frictions": [float(mu) for mu in args.frictions],
            "straight_angles_deg": [float(a) for a in args.straight_angles],
            "angled_angles_deg": [float(a) for a in args.angled_angles],
            "push_steps": [int(n) for n in args.push_steps],
            "push_distances_m": [float(d) for d in args.push_distances],
            "push_scales": [float(s) for s in args.push_scales],
            "speed_bin_edges_m_per_step": [float(v) for v in args.speed_bin_edges],
            "pairs_per_friction_split_speed_bin": int(args.pairs_per_friction_split_speed_bin),
            "max_trials_per_bucket": int(args.max_trials_per_bucket),
            "acceptance": {
                "min_displacement_m": float(args.min_displacement),
                "max_displacement_m": float(args.max_displacement),
                "max_final_speed_mps": float(args.max_final_speed),
                "max_eef_step_m": float(args.max_eef_step),
                "x_bounds": [float(v) for v in args.x_bounds],
                "y_bounds": [float(v) for v in args.y_bounds],
            },
        },
        "pairs": pairs,
        "cases": cases,
        "balance": {
            "target_buckets": {
                "|".join(key): value for key, value in sorted(target_buckets.items())
            },
            "accepted_buckets": {
                "|".join(key): value for key, value in sorted(accepted_buckets.items())
            },
            "trial_buckets": {
                "|".join(key): value for key, value in sorted(trial_buckets.items())
            },
            "missing_buckets": {
                "|".join(key): target_buckets[key] - accepted_buckets.get(key, 0)
                for key in sorted(target_buckets)
                if accepted_buckets.get(key, 0) < target_buckets[key]
            },
        },
        "rejected_count": len(rejected),
        "rejected_preview": rejected[:50],
    }


def write_payload(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    existing_pair_ids: set[str] = set()
    if args.resume_existing and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        pairs = list(existing.get("pairs", []))
        cases = list(existing.get("cases", []))
        existing_pair_ids = {str(pair.get("pair_id")) for pair in pairs}
        print(f"resuming {len(pairs)} pairs / {len(cases)} cases from {output}", flush=True)

    angle_splits = [("straight", float(a)) for a in args.straight_angles] + [
        ("angled", float(a)) for a in args.angled_angles
    ]
    candidates: list[dict[str, Any]] = []
    for friction_mu in [float(mu) for mu in args.frictions]:
        for split, angle_deg in angle_splits:
            for push_distance in [float(d) for d in args.push_distances]:
                for push_steps in [int(n) for n in args.push_steps]:
                    for push_scale in [float(s) for s in args.push_scales]:
                        speed = float(push_distance) / float(push_steps)
                        candidates.append(
                            {
                                "friction_mu": friction_mu,
                                "split": split,
                                "angle_deg": angle_deg,
                                "push_distance": float(push_distance),
                                "push_steps": int(push_steps),
                                "push_scale": float(push_scale),
                                "speed_m_per_step": speed,
                                "speed_bin": speed_bin(speed, [float(v) for v in args.speed_bin_edges]),
                            }
                        )

    if not args.no_shuffle_candidates:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(candidates)

    quota = int(args.pairs_per_friction_split_speed_bin)
    target_buckets: dict[tuple[str, str, str], int] = {}
    accepted_buckets: dict[tuple[str, str, str], int] = {}
    trial_buckets: dict[tuple[str, str, str], int] = {}
    if quota > 0:
        split_names = sorted({split for split, _ in angle_splits})
        bins = sorted({candidate["speed_bin"] for candidate in candidates})
        for friction_mu in [float(mu) for mu in args.frictions]:
            for split in split_names:
                for bin_name in bins:
                    key = (mu_tag(friction_mu), split, bin_name)
                    target_buckets[key] = quota
                    accepted_buckets[key] = 0
                    trial_buckets[key] = 0
        for pair in pairs:
            key = (mu_tag(float(pair["friction_mu"])), str(pair["split"]), str(pair["speed_bin"]))
            if key in accepted_buckets:
                accepted_buckets[key] = accepted_buckets.get(key, 0) + 1

    interrupted = False
    try:
        for candidate in candidates:
            friction_mu = float(candidate["friction_mu"])
            split = str(candidate["split"])
            angle_deg = float(candidate["angle_deg"])
            push_distance = float(candidate["push_distance"])
            push_steps = int(candidate["push_steps"])
            push_scale = float(candidate["push_scale"])
            bin_name = str(candidate["speed_bin"])
            bucket_key = (mu_tag(friction_mu), split, bin_name)
            if quota > 0 and accepted_buckets.get(bucket_key, 0) >= quota:
                continue
            if (
                quota > 0
                and int(args.max_trials_per_bucket) > 0
                and trial_buckets.get(bucket_key, 0) >= int(args.max_trials_per_bucket)
            ):
                continue
            if quota > 0:
                trial_buckets[bucket_key] = trial_buckets.get(bucket_key, 0) + 1

            base_id = (
                f"{split}_{mu_tag(friction_mu)}_{bin_name}_{angle_tag(angle_deg)}_"
                f"d{int(round(push_distance * 100)):02d}_n{push_steps:02d}_s{push_scale:g}"
            )
            if base_id in existing_pair_ids:
                continue
            probe = build_probe_case(
                case_id=f"probe_{base_id}",
                friction_mu=friction_mu,
                split=split,
                angle_deg=angle_deg,
                push_steps=push_steps,
                push_distance=push_distance,
                push_scale=push_scale,
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
                    "push_eef_backward_steps": int(result["push_eef_backward_steps"]),
                    "max_eef_step_m": float(result["max_eef_step_m"]),
                    "speed_m_per_step": float(candidate["speed_m_per_step"]),
                    "speed_bin": bin_name,
                    "push_scale": push_scale,
                }
            )
            if not accepted:
                rejected.append({"case_id": probe.case_id, "friction_mu": friction_mu, **metrics})
                continue
            if (
                int(result["push_backward_action_count"]) > 0
                or int(result["push_eef_backward_steps"]) > 0
                or float(result["max_eef_step_m"]) > float(args.max_eef_step)
            ):
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
                    "speed_bin": bin_name,
                    "speed_m_per_step": float(candidate["speed_m_per_step"]),
                    "angle_deg": angle_deg,
                    "push_distance_x": push_distance,
                    "pusher_push_steps": push_steps,
                    "pusher_push_controller_scale": push_scale,
                    "observation_case_id": observation_case.case_id,
                    "task_case_id": task_case.case_id,
                    "target_xy": result["final_xy"],
                    "metrics": metrics,
                }
            )
            if quota > 0:
                accepted_buckets[bucket_key] = accepted_buckets.get(bucket_key, 0) + 1
            print(
                f"accepted {base_id} target=({result['final_xy'][0]:.3f},{result['final_xy'][1]:.3f}) "
                f"disp={metrics['displacement_m'] * 100:.1f}cm final_speed={metrics['final_speed_mps']:.4f} "
                f"push_speed={candidate['speed_m_per_step']:.4f}",
                flush=True,
            )
            if int(args.autosave_every) > 0 and len(pairs) % int(args.autosave_every) == 0:
                write_payload(
                    output,
                    build_payload(
                        args=args,
                        init_xy=init_xy,
                        pairs=pairs,
                        cases=cases,
                        target_buckets=target_buckets,
                        accepted_buckets=accepted_buckets,
                        trial_buckets=trial_buckets,
                        rejected=rejected,
                    ),
                )
            if int(args.max_cases) > 0 and len(pairs) >= int(args.max_cases):
                break
            if quota > 0 and all(accepted_buckets[key] >= target for key, target in target_buckets.items()):
                break
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; writing partial dataset", flush=True)

    if interrupted:
        payload = build_payload(
            args=args,
            init_xy=init_xy,
            pairs=pairs,
            cases=cases,
            target_buckets=target_buckets,
            accepted_buckets=accepted_buckets,
            trial_buckets=trial_buckets,
            rejected=rejected,
        )
        payload["interrupted"] = True
        write_payload(output, payload)
        print(f"wrote partial {len(pairs)} pairs / {len(cases)} cases to {output}")
        return

    # Unbalanced legacy mode and balanced mode share the same final payload path.
    # The loop above handles both; target bucket maps stay empty when quota is 0.
    payload = build_payload(
        args=args,
        init_xy=init_xy,
        pairs=pairs,
        cases=cases,
        target_buckets=target_buckets,
        accepted_buckets=accepted_buckets,
        trial_buckets=trial_buckets,
        rejected=rejected,
    )
    write_payload(output, payload)
    print(f"wrote {len(pairs)} pairs / {len(cases)} cases to {output}")
    print(f"rejected {len(rejected)} candidates")
    if payload["balance"]["missing_buckets"]:
        print(f"missing buckets: {len(payload['balance']['missing_buckets'])}")
    return


if __name__ == "__main__":
    main()

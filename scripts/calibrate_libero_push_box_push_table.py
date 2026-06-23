#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(FASTWAM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FASTWAM_ROOT / "src"))

from generate_libero_push_box_adaptation_dataset import write_geometry_bddl  # noqa: E402
from generate_libero_push_box_rollout_target_dataset import (  # noqa: E402
    build_probe_case,
    direction_xy,
    mu_tag,
    rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a no-video physics calibration table for LIBERO push-box. "
            "The output maps friction and desired slide distance to push stroke/speed/scale settings."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / "libero_push_box_push_calibration")
    parser.add_argument(
        "--seed-config",
        type=Path,
        default=REPO_ROOT / "configs" / "libero_push_box_all_friction_compare_selected.json",
        help="Prior successful push settings used to center the local sweep.",
    )
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.005, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--angles", type=float, nargs="+", default=[0.0])
    parser.add_argument("--init-xy", type=float, nargs=2, default=[-0.245, -0.035])
    parser.add_argument("--target-distances", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.265, 0.30, 0.40, 0.50])
    parser.add_argument("--target-tolerance", type=float, default=0.04)
    parser.add_argument("--camera-resolution", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--max-final-speed", type=float, default=0.035)
    parser.add_argument("--max-eef-step", type=float, default=0.06)
    parser.add_argument("--x-bounds", type=float, nargs=2, default=[-0.24, 0.38])
    parser.add_argument("--y-bounds", type=float, nargs=2, default=[-0.26, 0.26])
    parser.add_argument("--max-candidates-per-mu", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026062301)
    parser.add_argument("--write-config-copy", type=Path, default=REPO_ROOT / "configs" / "libero_push_box_push_calibration_table.json")
    return parser.parse_args()


def load_seed_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    return [case for case in cases if isinstance(case, dict)]


def nearest_seed(cases: list[dict[str, Any]], friction_mu: float) -> dict[str, Any] | None:
    if not cases:
        return None
    return min(cases, key=lambda case: abs(float(case.get("friction_mu", 0.0)) - float(friction_mu)))


def unique_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out = []
    for item in candidates:
        key = (
            item["mode"],
            int(item["push_steps"]),
            round(float(item["push_distance"]), 4),
            round(float(item["push_scale"]), 4),
            round(float(item["action_end"]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def local_candidates_for_mu(friction_mu: float, seed: dict[str, Any] | None, max_count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if seed is not None:
        mode = str(seed.get("pusher_push_mode", "position"))
        steps = int(seed.get("pusher_push_steps", 8))
        stroke = float(seed.get("pusher_push_distance_x", 0.14))
        scale = float(seed.get("pusher_push_controller_scale", 8.0))
        action_end = float(seed.get("pusher_push_action_end", 1.0 if mode == "position" else 0.4))
        step_values = sorted({max(1, steps + delta) for delta in (-4, -2, -1, 0, 1, 2, 4, 8)})
        stroke_values = sorted({round(max(0.10, stroke + delta), 3) for delta in (-0.02, 0.0, 0.02, 0.04)})
        scale_values = sorted({max(1.0, scale + delta) for delta in (-6, -3, 0, 3, 6, 10)})
        action_values = sorted({max(0.2, action_end + delta) for delta in (-0.1, -0.05, 0.0, 0.05, 0.1)})
        for push_steps in step_values:
            for push_distance in stroke_values:
                for push_scale in scale_values:
                    for action in (action_values if mode == "impulse" else [1.0]):
                        candidates.append(
                            {
                                "mode": mode,
                                "push_steps": int(push_steps),
                                "push_distance": float(push_distance),
                                "push_scale": float(push_scale),
                                "action_end": float(action),
                                "source": "seed_local",
                            }
                        )

    if float(friction_mu) <= 0.02:
        for push_steps in (8, 12, 16, 24, 32, 40):
            for push_distance in (0.12, 0.14, 0.16, 0.18, 0.20, 0.22):
                for push_scale in (1.5, 2.0, 3.0, 4.0, 6.0):
                    candidates.append(
                        {
                            "mode": "position",
                            "push_steps": push_steps,
                            "push_distance": push_distance,
                            "push_scale": push_scale,
                            "action_end": 1.0,
                            "source": "low_mu_grid",
                        }
                    )
    else:
        for push_steps in (2, 3, 4, 5, 6, 8):
            for push_distance in (0.12, 0.14, 0.16):
                for push_scale in (8.0, 10.0, 12.0, 14.0, 16.0, 20.0):
                    for action_end in (0.35, 0.4, 0.45, 0.5, 0.6):
                        candidates.append(
                            {
                                "mode": "impulse",
                                "push_steps": push_steps,
                                "push_distance": push_distance,
                                "push_scale": push_scale,
                                "action_end": action_end,
                                "source": "high_mu_impulse_grid",
                            }
                        )
        for push_steps in (8, 12, 16, 24, 36):
            for push_distance in (0.16, 0.18, 0.20, 0.22):
                for push_scale in (12.0, 20.0, 32.0, 48.0, 64.0):
                    candidates.append(
                        {
                            "mode": "position",
                            "push_steps": push_steps,
                            "push_distance": push_distance,
                            "push_scale": push_scale,
                            "action_end": 1.0,
                            "source": "high_mu_position_grid",
                        }
                    )

    ordered = unique_candidates(candidates)
    if max_count > 0 and len(ordered) > max_count:
        # Keep a deterministic spread across the ordered local/grid candidates instead of truncating only one region.
        idxs = np.linspace(0, len(ordered) - 1, num=max_count, dtype=int).tolist()
        ordered = [ordered[idx] for idx in idxs]
    return ordered


def probe_candidate(
    *,
    args: argparse.Namespace,
    friction_mu: float,
    angle_deg: float,
    candidate: dict[str, Any],
    trial_index: int,
) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    init_xy = (float(args.init_xy[0]), float(args.init_xy[1]))
    direction = direction_xy(float(angle_deg))
    target_xy = (
        float(init_xy[0] + direction[0] * 0.30),
        float(init_xy[1] + direction[1] * 0.30),
    )
    case_id = (
        f"calib_{trial_index:05d}_{mu_tag(float(friction_mu))}_"
        f"a{int(round(float(angle_deg))):+03d}_{candidate['mode']}_"
        f"d{int(round(float(candidate['push_distance']) * 100)):02d}_"
        f"n{int(candidate['push_steps']):02d}_s{float(candidate['push_scale']):g}_"
        f"ae{float(candidate['action_end']):g}"
    ).replace("+", "p").replace("-", "m")
    bddl_file = write_geometry_bddl(
        repo_root=repo_root,
        bddl_dir=args.output_dir / "bddl",
        geometry_id=case_id,
        init_xy=init_xy,
        target_xy=target_xy,
        init_half_size=float(args.init_half_size),
        target_radius=float(args.target_radius),
        target_rgba=(0.0, 0.0, 0.0, 0.0),
    )
    probe = build_probe_case(
        case_id=case_id,
        friction_mu=float(friction_mu),
        split="straight" if abs(float(angle_deg)) < 1e-6 else "angled",
        angle_deg=float(angle_deg),
        push_steps=int(candidate["push_steps"]),
        push_distance=float(candidate["push_distance"]),
        push_scale=float(candidate["push_scale"]),
        init_xy=init_xy,
        target_xy=target_xy,
        bddl_file=bddl_file,
        max_steps=int(args.max_steps),
        camera_resolution=int(args.camera_resolution),
        target_radius=float(args.target_radius),
    )
    probe = replace(
        probe,
        pusher_push_mode=str(candidate["mode"]),
        pusher_push_action_end=float(candidate["action_end"]),
        pusher_push_controller_scale=float(candidate["push_scale"]),
        pusher_max_push_controller_scale=max(20.0, float(candidate["push_scale"])),
        pusher_push_distance_x=float(candidate["push_distance"]),
        pusher_push_steps=int(candidate["push_steps"]),
    )
    result = rollout(probe, repo_root=repo_root, seed=int(args.seed))
    final = np.asarray(result["final_xy"], dtype=np.float64)
    init = np.asarray(init_xy, dtype=np.float64)
    delta = final - init
    forward = float(np.dot(delta, direction))
    lateral = float(direction[0] * delta[1] - direction[1] * delta[0])
    displacement = float(np.linalg.norm(delta))
    x_ok = float(args.x_bounds[0]) <= float(final[0]) <= float(args.x_bounds[1])
    y_ok = float(args.y_bounds[0]) <= float(final[1]) <= float(args.y_bounds[1])
    smooth_ok = (
        int(result["push_backward_action_count"]) == 0
        and int(result["push_eef_backward_steps"]) == 0
        and float(result["max_eef_step_m"]) <= float(args.max_eef_step)
    )
    valid = bool(x_ok and y_ok and smooth_ok and float(result["final_speed"]) <= float(args.max_final_speed))
    return {
        "friction_mu": float(friction_mu),
        "angle_deg": float(angle_deg),
        "mode": str(candidate["mode"]),
        "push_steps": int(candidate["push_steps"]),
        "push_distance_m": float(candidate["push_distance"]),
        "push_scale": float(candidate["push_scale"]),
        "action_end": float(candidate["action_end"]),
        "speed_m_per_step": float(candidate["push_distance"]) / float(max(1, int(candidate["push_steps"]))),
        "source": str(candidate.get("source", "")),
        "forward_m": forward,
        "lateral_m": lateral,
        "displacement_m": displacement,
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_speed_mps": float(result["final_speed"]),
        "max_eef_step_m": float(result["max_eef_step_m"]),
        "push_backward_action_count": int(result["push_backward_action_count"]),
        "push_eef_backward_steps": int(result["push_eef_backward_steps"]),
        "x_ok": bool(x_ok),
        "y_ok": bool(y_ok),
        "smooth_ok": bool(smooth_ok),
        "valid": valid,
    }


def choose_recommendations(rows: list[dict[str, Any]], targets: list[float], tolerance: float) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(float(row["friction_mu"]), float(row["angle_deg"])) for row in rows})
    for friction_mu, angle_deg in groups:
        group_rows = [row for row in rows if float(row["friction_mu"]) == friction_mu and float(row["angle_deg"]) == angle_deg]
        valid_rows = [row for row in group_rows if row["valid"]]
        for target in targets:
            candidates = valid_rows or group_rows
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda row: (
                    abs(float(row["displacement_m"]) - float(target)),
                    0 if row["valid"] else 1,
                    float(row["max_eef_step_m"]),
                    int(row["push_steps"]),
                ),
            )
            item = dict(best)
            item["target_displacement_m"] = float(target)
            item["abs_error_m"] = abs(float(best["displacement_m"]) - float(target))
            item["within_tolerance"] = bool(item["abs_error_m"] <= float(tolerance))
            out.append(item)
    return out


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "probes.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    csv_path = args.output_dir / "probes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_at": dt.datetime.now().isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "recommendations": recommendations,
    }
    rec_path = args.output_dir / "recommendations.json"
    rec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.write_config_copy:
        args.write_config_copy.parent.mkdir(parents=True, exist_ok=True)
        args.write_config_copy.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = args.output_dir / "recommendations.md"
    lines = [
        "# LIBERO Push Box Calibration",
        "",
        "| mu | angle | target cm | actual cm | mode | stroke cm | steps | speed cm/step | scale | action_end | valid | err cm |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in recommendations:
        lines.append(
            "| "
            f"{row['friction_mu']:.3f} | {row['angle_deg']:.0f} | {row['target_displacement_m'] * 100:.1f} | "
            f"{row['displacement_m'] * 100:.1f} | {row['mode']} | {row['push_distance_m'] * 100:.1f} | "
            f"{row['push_steps']} | {row['speed_m_per_step'] * 100:.2f} | {row['push_scale']:.1f} | "
            f"{row['action_end']:.2f} | {row['valid']} | {row['abs_error_m'] * 100:.1f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rows={len(rows)} recommendations={len(recommendations)}")
    print(f"probes={rows_path}")
    print(f"recommendations={rec_path}")
    print(f"markdown={md_path}")
    if args.write_config_copy:
        print(f"config_copy={args.write_config_copy}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_cases = load_seed_cases(args.seed_config)
    rows: list[dict[str, Any]] = []
    trial_index = 0
    for friction_mu in [float(mu) for mu in args.frictions]:
        seed = nearest_seed(seed_cases, friction_mu)
        candidates = local_candidates_for_mu(friction_mu, seed, int(args.max_candidates_per_mu))
        print(f"mu={friction_mu:.3f} candidates={len(candidates)} seed={seed.get('case_id') if seed else None}", flush=True)
        for angle_deg in [float(angle) for angle in args.angles]:
            for candidate in candidates:
                row = probe_candidate(args=args, friction_mu=friction_mu, angle_deg=angle_deg, candidate=candidate, trial_index=trial_index)
                rows.append(row)
                trial_index += 1
                if len(rows) % 20 == 0:
                    print(
                        f"probed={len(rows)} latest mu={friction_mu:.3f} disp={row['displacement_m'] * 100:.1f}cm "
                        f"mode={row['mode']} steps={row['push_steps']} scale={row['push_scale']} valid={row['valid']}",
                        flush=True,
                    )
    recommendations = choose_recommendations(rows, [float(v) for v in args.target_distances], float(args.target_tolerance))
    write_outputs(args, rows, recommendations)


if __name__ == "__main__":
    main()

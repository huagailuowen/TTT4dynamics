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

from generate_libero_push_box_adaptation_dataset import build_case, write_geometry_bddl  # noqa: E402
from ttt4dynamics.push_box_libero import LiberoPushBoxCase, LiberoPushBoxEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep one LIBERO push-box impulse controller case.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bddl-dir", type=Path, default=REPO_ROOT / "generated_bddl" / "push_box_impulse_sweep")
    parser.add_argument("--friction", type=float, default=0.2)
    parser.add_argument("--init-xy", type=float, nargs=2, default=(-0.245, -0.035))
    parser.add_argument("--target-distance", type=float, default=0.265)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--push-distance-x", type=float, default=0.14)
    parser.add_argument("--push-steps", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    parser.add_argument("--push-scales", type=float, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--action-ends", type=float, nargs="+", default=[0.5, 0.6, 0.8, 1.0])
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--camera-resolution", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def rollout(case: LiberoPushBoxCase, *, repo_root: Path, seed: int) -> dict[str, Any]:
    env = LiberoPushBoxEnv(case, repo_root=repo_root, seed=seed)
    records: list[dict[str, Any]] = []
    try:
        env.reset()
        initial = env.step_info().as_dict()
        initial["phase"] = "reset"
        records.append(initial)
        for _ in range(int(case.max_steps)):
            _, _, _, info = env.step()
            records.append(info["push_box"])
    finally:
        env.close()

    final = records[-1]
    push_actions = [float(r["action"][0]) for r in records if r.get("phase") == "push" and r.get("action")]
    eef_x = [float(r["eef_xyz"][0]) for r in records if r.get("phase") == "push" and r.get("eef_xyz")]
    eef_dx = np.diff(np.asarray(eef_x, dtype=np.float64)) if len(eef_x) > 1 else np.zeros(0)
    all_eef = [np.asarray(r["eef_xyz"], dtype=np.float64) for r in records if r.get("eef_xyz")]
    all_eef_step = np.linalg.norm(np.diff(np.stack(all_eef), axis=0), axis=1) if len(all_eef) > 1 else np.zeros(0)
    return {
        "success": bool(final["success"]),
        "final_distance_m": float(final["distance_to_target"]),
        "min_distance_m": float(min(r["distance_to_target"] for r in records)),
        "final_xy": [float(final["box_xyz"][0]), float(final["box_xyz"][1])],
        "push_action_min": float(min(push_actions)) if push_actions else 0.0,
        "push_action_max": float(max(push_actions)) if push_actions else 0.0,
        "push_backward_action_count": int(sum(1 for x in push_actions if x < -1e-6)),
        "push_eef_backward_steps": int(np.sum(eef_dx < -1e-4)),
        "max_eef_step_m": float(np.max(all_eef_step)) if all_eef_step.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    bddl_dir = args.bddl_dir if args.bddl_dir.is_absolute() else repo_root / args.bddl_dir
    init_xy = (float(args.init_xy[0]), float(args.init_xy[1]))
    target_xy = (init_xy[0] + float(args.target_distance), init_xy[1])
    bddl_file = write_geometry_bddl(
        repo_root=repo_root,
        bddl_dir=bddl_dir,
        geometry_id="sweep_g00",
        init_xy=init_xy,
        target_xy=target_xy,
        init_half_size=float(args.init_half_size),
        target_radius=float(args.target_radius),
    )
    base = build_case(
        case_id=f"sweep_mu{int(round(float(args.friction) * 10000)):04d}",
        domain="sweep",
        friction_group="sweep",
        friction_mu=float(args.friction),
        geometry_id="sweep_g00",
        init_xy=init_xy,
        target_distance=float(args.target_distance),
        bddl_file=bddl_file,
        target_radius=float(args.target_radius),
        push_distance_x=float(args.push_distance_x),
        max_steps=int(args.max_steps),
        camera_resolution=int(args.camera_resolution),
    )

    results = []
    for push_steps in args.push_steps:
        for push_scale in args.push_scales:
            for action_end in args.action_ends:
                case = replace(
                    base,
                    case_id=f"{base.case_id}_s{push_scale:g}_n{push_steps}_a{action_end:g}",
                    pusher_push_steps=int(push_steps),
                    pusher_push_action_end=float(action_end),
                    pusher_max_pos_action=1.0,
                    pusher_push_controller_scale=float(push_scale),
                    pusher_push_controller_scale_ramp_steps=2,
                    pusher_push_mode="impulse",
                    controller_output_scale=1.0,
                    enable_controller_output_scaling=False,
                )
                result = {"case": case.as_dict(), "metrics": rollout(case, repo_root=repo_root, seed=int(args.seed))}
                results.append(result)
                m = result["metrics"]
                print(
                    f"{case.case_id}: success={m['success']} final={m['final_distance_m'] * 100:.2f}cm "
                    f"min={m['min_distance_m'] * 100:.2f}cm max_eef={m['max_eef_step_m'] * 100:.2f}cm "
                    f"back_action={m['push_backward_action_count']} back_eef={m['push_eef_backward_steps']}",
                    flush=True,
                )
                if m["success"] and m["push_backward_action_count"] == 0 and m["push_eef_backward_steps"] == 0:
                    payload = {"results": results, "selected": result}
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    print(f"selected={case.case_id}")
                    print(f"wrote {args.output}")
                    return

    best = min(results, key=lambda item: item["metrics"]["final_distance_m"]) if results else None
    payload = {"results": results, "selected": best}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best is not None:
        print(f"selected={best['case']['case_id']} nearest={best['metrics']['final_distance_m'] * 100:.2f}cm")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

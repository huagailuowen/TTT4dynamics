#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt4dynamics.push_box_libero import (  # noqa: E402
    LiberoPushBoxCase,
    LiberoPushBoxEnv,
    load_libero_push_box_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LIBERO push-box friction cases.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cases", type=Path, default=REPO_ROOT / "configs" / "libero_push_box_friction_cases.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "libero_push_box_friction_cases")
    parser.add_argument("--frictions", type=float, nargs="+", help="Override config with a friction sweep.")
    parser.add_argument("--friction-min", type=float, default=None)
    parser.add_argument("--friction-max", type=float, default=None)
    parser.add_argument("--friction-count", type=int, default=21)
    parser.add_argument("--push-action-end", type=float, default=None)
    parser.add_argument("--controller-output-scale", type=float, default=None)
    parser.add_argument("--push-steps", type=int, default=None)
    parser.add_argument("--push-accel-steps", type=int, default=None)
    parser.add_argument("--push-distance-x", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--camera-resolution", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", choices=["agent", "wrist", "both"], default="agent")
    parser.add_argument("--comparison-cols", type=int, default=3)
    return parser.parse_args()


def _format_mu_tag(friction_mu: float) -> str:
    return f"mu{int(round(float(friction_mu) * 10000)):04d}"


def _case_with_overrides(
    case: LiberoPushBoxCase,
    *,
    push_action_end: float | None,
    controller_output_scale: float | None,
    push_steps: int | None,
    push_accel_steps: int | None,
    push_distance_x: float | None,
    max_steps: int | None,
    camera_resolution: int | None,
) -> LiberoPushBoxCase:
    data = case.as_dict()
    if push_action_end is not None:
        data["pusher_push_action_end"] = float(push_action_end)
    if controller_output_scale is not None:
        data["controller_output_scale"] = float(controller_output_scale)
    if push_steps is not None:
        data["pusher_push_steps"] = int(push_steps)
    if push_accel_steps is not None:
        data["pusher_push_accel_steps"] = int(push_accel_steps)
    if push_distance_x is not None:
        data["pusher_push_distance_x"] = float(push_distance_x)
    if max_steps is not None:
        data["max_steps"] = int(max_steps)
    if camera_resolution is not None:
        data["camera_resolution"] = int(camera_resolution)
    return LiberoPushBoxCase.from_dict(data)


def _load_cases(args: argparse.Namespace) -> list[LiberoPushBoxCase]:
    if args.frictions:
        friction_values = [float(mu) for mu in args.frictions]
    elif args.friction_min is not None or args.friction_max is not None:
        lo = 0.0 if args.friction_min is None else float(args.friction_min)
        hi = 0.2 if args.friction_max is None else float(args.friction_max)
        if hi < lo:
            raise ValueError("--friction-max must be >= --friction-min.")
        if int(args.friction_count) <= 0:
            raise ValueError("--friction-count must be positive.")
        friction_values = np.linspace(lo, hi, int(args.friction_count), dtype=np.float64).tolist()
    else:
        friction_values = []

    if friction_values:
        base = LiberoPushBoxCase(case_id="libero_push_box_base", friction_mu=float(friction_values[0]))
        cases = [base.with_friction(mu, f"libero_push_box_{_format_mu_tag(mu)}") for mu in friction_values]
    else:
        cases = load_libero_push_box_cases(args.cases)
    return [
        _case_with_overrides(
            case,
            push_action_end=args.push_action_end,
            controller_output_scale=args.controller_output_scale,
            push_steps=args.push_steps,
            push_accel_steps=args.push_accel_steps,
            push_distance_x=args.push_distance_x,
            max_steps=args.max_steps,
            camera_resolution=args.camera_resolution,
        )
        for case in cases
    ]


def _label_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    pil = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    line_height = 18
    width = max(draw.textbbox((0, 0), line)[2] for line in lines) + 14
    height = line_height * len(lines) + 8
    draw.rectangle((4, 4, min(pil.width - 2, width), min(pil.height - 2, height)), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((8, 7 + idx * line_height), line, fill=(255, 255, 255))
    return np.asarray(pil)


def _obs_to_frame(obs: dict, camera: str, lines: list[str]) -> np.ndarray:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if camera == "agent":
        return _label_frame(agent, lines)
    if camera == "wrist":
        return _label_frame(wrist, lines)
    return np.concatenate(
        [
            _label_frame(agent, ["agent"] + lines),
            _label_frame(wrist, ["wrist"] + lines),
        ],
        axis=1,
    )


def render_case(case: LiberoPushBoxCase, *, repo_root: Path, output: Path, fps: int, seed: int, camera: str) -> dict:
    env = LiberoPushBoxEnv(case, repo_root=repo_root, seed=seed)
    frames: list[np.ndarray] = []
    step_records: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        obs = env.reset()
        info = env.step_info()
        initial_record = info.as_dict()
        initial_record["phase"] = "reset"
        initial_record["action"] = None
        step_records.append(initial_record)
        frames.append(
            _obs_to_frame(
                obs,
                camera,
                [
                    f"{case.case_id} mu={case.friction_mu:.3f} ax={case.pusher_push_action_end:.2f}",
                    f"scale={case.controller_output_scale:.1f} stroke={case.pusher_push_distance_x * 100:.1f}cm push={case.pusher_push_steps}",
                    f"step={info.step:03d} dist={info.distance_to_target * 100:.1f}cm success={info.success}",
                ],
            )
        )
        with imageio.get_writer(output, fps=fps, codec="libx264") as writer:
            writer.append_data(frames[-1])
            for _ in range(int(case.max_steps)):
                obs, _, _, step_raw = env.step()
                record = step_raw["push_box"]
                step_records.append(record)
                frame = _obs_to_frame(
                    obs,
                    camera,
                    [
                        f"{case.case_id} mu={case.friction_mu:.3f} ax={case.pusher_push_action_end:.2f}",
                        f"scale={case.controller_output_scale:.1f} stroke={case.pusher_push_distance_x * 100:.1f}cm push={case.pusher_push_steps}",
                        f"step={record['step']:03d} dist={record['distance_to_target'] * 100:.1f}cm success={record['success']}",
                    ],
                )
                frames.append(frame)
                writer.append_data(frame)
    finally:
        env.close()

    final = step_records[-1]
    return {
        "case": case.as_dict(),
        "video": str(output),
        "final": final,
        "success": bool(final["success"]),
        "min_distance_to_target": float(min(record["distance_to_target"] for record in step_records)),
        "steps": step_records,
        "frames": frames,
    }


def write_comparison(results: list[dict], *, output: Path, fps: int, cols: int) -> None:
    if not results:
        return
    rows = int(np.ceil(len(results) / float(cols)))
    max_len = max(len(result["frames"]) for result in results)
    sample = results[0]["frames"][0]
    frame_h, frame_w = sample.shape[:2]
    blank = np.zeros_like(sample)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264") as writer:
        for frame_idx in range(max_len):
            row_frames = []
            for row in range(rows):
                col_frames = []
                for col in range(cols):
                    result_idx = row * cols + col
                    if result_idx >= len(results):
                        col_frames.append(blank)
                        continue
                    frames = results[result_idx]["frames"]
                    col_frames.append(frames[min(frame_idx, len(frames) - 1)])
                row_frames.append(np.concatenate(col_frames, axis=1))
            writer.append_data(np.concatenate(row_frames, axis=0))


def strip_frames(result: dict) -> dict:
    payload = dict(result)
    payload.pop("frames", None)
    return payload


def main() -> None:
    args = parse_args()
    cases = _load_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case_idx, case in enumerate(cases):
        video = args.output_dir / f"{case_idx:02d}_{case.case_id}.mp4"
        result = render_case(
            case,
            repo_root=args.repo_root,
            output=video,
            fps=args.fps,
            seed=args.seed,
            camera=args.camera,
        )
        results.append(result)
        final = result["final"]
        print(
            f"{case.case_id}: mu={case.friction_mu:.3f} "
            f"final_xy=({final['box_xyz'][0]:.3f},{final['box_xyz'][1]:.3f}) "
            f"dist={final['distance_to_target'] * 100:.2f}cm "
            f"min_dist={result['min_distance_to_target'] * 100:.2f}cm "
            f"success={result['success']} video={video}"
        )

    comparison = args.output_dir / "comparison.mp4"
    write_comparison(results, output=comparison, fps=args.fps, cols=int(args.comparison_cols))
    metadata = {
        "comparison_video": str(comparison),
        "results": [strip_frames(result) for result in results],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"comparison_video={comparison}")
    print(f"metadata={args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

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

from ttt4dynamics.cases import DynamicCarrierCase
from ttt4dynamics.dynamic_env import create_libero_env_for_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an MP4 from a collected dynamic-carrier demo.")
    parser.add_argument("--demo-npz", type=Path, required=True, help="Path to demo_XXXXXX.npz.")
    parser.add_argument("--demo-json", type=Path, help="Path to matching demo_XXXXXX.json.")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 path.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="TTT4dynamics repo root.")
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", choices=["agent", "wrist", "both"], default="both")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap; 0 renders all states.")
    return parser.parse_args()


def _resolve_demo_json(demo_npz: Path, demo_json: Path | None) -> Path:
    if demo_json is not None:
        return demo_json
    candidate = demo_npz.with_suffix(".json")
    if not candidate.exists():
        raise FileNotFoundError(f"Could not infer demo JSON path: {candidate}")
    return candidate


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    pil = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    x0, y0 = 6, 6
    x1 = min(pil.width - 2, x0 + 8 * len(label) + 8)
    y1 = y0 + 20
    draw.rectangle((x0 - 2, y0 - 2, x1, y1), fill=(0, 0, 0))
    draw.text((x0, y0), label, fill=(255, 255, 255))
    return np.asarray(pil)


def _obs_to_frame(obs: dict, camera: str, label: str) -> np.ndarray:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if camera == "agent":
        return _label_frame(agent, label)
    if camera == "wrist":
        return _label_frame(wrist, label)
    return np.concatenate(
        [
            _label_frame(agent, f"agent | {label}"),
            _label_frame(wrist, "wrist"),
        ],
        axis=1,
    )


def render_demo(
    *,
    demo_npz: Path,
    demo_json: Path,
    output: Path,
    repo_root: Path,
    camera_resolution: int,
    fps: int,
    camera: str,
    max_frames: int,
) -> None:
    meta = json.loads(demo_json.read_text(encoding="utf-8"))
    case = DynamicCarrierCase.from_dict(meta["case"])
    states = np.load(demo_npz)["states"]
    if max_frames > 0:
        states = states[:max_frames]

    output.parent.mkdir(parents=True, exist_ok=True)
    env, _, _ = create_libero_env_for_case(
        case,
        repo_root=repo_root,
        camera_resolution=camera_resolution,
        seed=int(meta.get("seed", 0)),
    )

    try:
        env.reset()
        with imageio.get_writer(output, fps=fps) as writer:
            for step_idx, state in enumerate(states):
                obs = env.set_init_state(state)
                label = f"{meta['case_id']} step={step_idx:03d} success={meta['success']}"
                writer.append_data(_obs_to_frame(obs, camera, label))
    finally:
        env.close()

    print(f"Wrote video: {output}")


def main() -> None:
    args = parse_args()
    demo_json = _resolve_demo_json(args.demo_npz, args.demo_json)
    render_demo(
        demo_npz=args.demo_npz,
        demo_json=demo_json,
        output=args.output,
        repo_root=args.repo_root,
        camera_resolution=args.camera_resolution,
        fps=args.fps,
        camera=args.camera,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

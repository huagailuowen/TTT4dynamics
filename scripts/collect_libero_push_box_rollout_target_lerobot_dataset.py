#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(FASTWAM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FASTWAM_ROOT / "src"))

from fastwam.datasets.lerobot.lerobot import lerobot_dataset as lerobot_dataset_module  # noqa: E402
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset  # noqa: E402

from generate_libero_push_box_adaptation_dataset import write_geometry_bddl  # noqa: E402
from generate_libero_push_box_rollout_target_dataset import (  # noqa: E402
    accept_rollout,
    angle_tag,
    build_probe_case,
    direction_xy,
    mu_tag,
    rollout,
    speed_bin,
)
from ttt4dynamics.push_box_libero import LiberoPushBoxCase, LiberoPushBoxEnv  # noqa: E402


VISIBLE_TASK_PROMPT = "push the cream cheese box across the table into the green target region"
HIDDEN_OBSERVATION_PROMPT = (
    "observe how the cream cheese box slides after a short robot push on the table; no target is shown"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a balanced LIBERO push-box rollout-target dataset in LeRobot format. "
            "Accepted rollouts are written twice: hidden target for observation/play and visible target for task."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=FASTWAM_ROOT / "data" / "libero_push_box_rollout_target_v1",
        help="Prefix for four LeRobot roots: *_hidden_straight_lerobot, *_hidden_angled_lerobot, etc.",
    )
    parser.add_argument(
        "--bddl-dir",
        type=Path,
        default=REPO_ROOT / "generated_bddl" / "push_box_rollout_target_v1",
    )
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.005, 0.02, 0.1, 0.2])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["straight", "angled"],
        choices=["straight", "angled"],
        help="Which push direction families to collect. Use separate runs for clean parallel straight/angled output.",
    )
    parser.add_argument("--straight-angles", type=float, nargs="+", default=[0.0])
    parser.add_argument("--angled-angles", type=float, nargs="+", default=[-30.0, -20.0, -10.0, 10.0, 20.0, 30.0])
    parser.add_argument(
        "--init-xys",
        nargs="+",
        action="append",
        default=None,
        help="Initial box xy values as 'x,y'. Can be repeated to pass negative coordinates safely.",
    )
    parser.add_argument("--push-steps", type=int, nargs="+", default=[8, 12, 16, 28, 40])
    parser.add_argument("--push-distances", type=float, nargs="+", default=[0.12, 0.16, 0.20, 0.24])
    parser.add_argument("--push-scales", type=float, nargs="+", default=[4.0])
    parser.add_argument("--dummy-target-distance", type=float, default=0.30)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--probe-resolution", type=int, default=24)
    parser.add_argument("--camera-resolution", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--repo-id-prefix", default="libero_push_box_rollout_target_v1")
    parser.add_argument("--video-codec", default="h264", choices=["h264", "hevc", "libsvtav1", "h264_nvenc"])
    parser.add_argument("--video-crf", type=int, default=18, help="Lower is higher quality. The quick smoke used 30.")
    parser.add_argument("--jpeg-quality", type=int, default=98)
    parser.add_argument("--pairs-per-bucket", type=int, default=1)
    parser.add_argument("--max-trials-per-bucket", type=int, default=80)
    parser.add_argument("--max-pairs", type=int, default=0, help="Debug cap on accepted rollout pairs. 0 means all buckets.")
    parser.add_argument(
        "--max-pairs-per-friction",
        type=int,
        default=0,
        help="Optional marginal cap per friction value. Useful when balancing split/distance but keeping friction roughly even.",
    )
    parser.add_argument("--autosave-every", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit successfully after saving manifest even when some balanced buckets remain physically unacceptable.",
    )
    parser.add_argument("--no-shuffle-candidates", action="store_true")
    parser.add_argument("--speed-bin-edges", type=float, nargs="+", default=[0.006, 0.012])
    parser.add_argument("--distance-bin-edges", type=float, nargs="+", default=[0.18])
    parser.add_argument(
        "--displacement-bin-edges",
        type=float,
        nargs="+",
        default=[0.20, 0.35],
        help="Actual rollout displacement bin edges in meters, computed after probing.",
    )
    parser.add_argument(
        "--displacement-bin-quotas",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Per-bucket quotas for actual displacement bins. For edges [0.20,0.35], "
            "three quotas map to short/mid/long bins."
        ),
    )
    parser.add_argument("--scale-bin-edges", type=float, nargs="+", default=[2.0])
    parser.add_argument(
        "--calibration-table",
        type=Path,
        default=None,
        help=(
            "Optional recommendations JSON from calibrate_libero_push_box_push_table.py. "
            "When provided, candidate pushes are generated around the calibrated mode/stroke/speed/scale settings."
        ),
    )
    parser.add_argument(
        "--balance-dimensions",
        nargs="+",
        default=["friction", "split", "speed_bin", "distance_bin"],
        choices=["friction", "split", "speed_bin", "distance_bin", "displacement_bin", "scale_bin", "init_bin"],
    )
    parser.add_argument("--min-displacement", type=float, default=0.05)
    parser.add_argument("--max-displacement", type=float, default=0.34)
    parser.add_argument("--max-final-speed", type=float, default=0.02)
    parser.add_argument("--max-eef-step", type=float, default=0.06)
    parser.add_argument("--x-bounds", type=float, nargs=2, default=(-0.20, 0.10))
    parser.add_argument("--y-bounds", type=float, nargs=2, default=(-0.14, 0.11))
    return parser.parse_args()


def parse_init_xys(values: list[list[str]] | None) -> list[tuple[str, tuple[float, float]]]:
    if values is None:
        flat_values = ["-0.255,-0.055", "-0.245,-0.035", "-0.235,-0.015", "-0.225,0.005"]
    else:
        flat_values = [item for group in values for item in group]
    parsed = []
    for idx, value in enumerate(flat_values):
        parts = [part.strip() for part in str(value).split(",")]
        if len(parts) != 2:
            raise ValueError(f"--init-xys values must be 'x,y', got {value!r}")
        parsed.append((f"i{idx:02d}", (float(parts[0]), float(parts[1]))))
    return parsed


def value_bin(value: float, edges: list[float], prefix: str) -> str:
    sorted_edges = sorted(float(edge) for edge in edges)
    for idx, edge in enumerate(sorted_edges):
        if float(value) < edge:
            return f"{prefix}_{idx:02d}"
    return f"{prefix}_{len(sorted_edges):02d}"


def bin_values(edges: list[float], prefix: str) -> list[str]:
    return [f"{prefix}_{idx:02d}" for idx in range(len(edges) + 1)]


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with shape (4,), got {quat.shape}")
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sin_half, w)
    axis = xyz / sin_half
    return (axis * angle).astype(np.float32)


def _obs_to_images(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
    return agent, wrist


def _obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _env_action_to_fastwam_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[-1] = (1.0 - out[-1]) / 2.0
    return out


def build_features(camera_resolution: int) -> dict[str, dict[str, Any]]:
    image_shape = (3, int(camera_resolution), int(camera_resolution))
    return {
        "observation.images.image": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["channel", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["channel", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_axis_x",
                "eef_axis_y",
                "eef_axis_z",
                "gripper_qpos_0",
                "gripper_qpos_1",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"],
        },
    }


def dataset_root(output_prefix: Path, domain: str, split: str) -> Path:
    domain_name = "visible" if domain == "task" else "hidden"
    return output_prefix.parent / f"{output_prefix.name}_{domain_name}_{split}_lerobot"


def prompt_for_case(domain: str, split: str) -> list[str]:
    split_prompt = "straight push" if split == "straight" else "angled push"
    if domain == "task":
        return [
            f"push-box physical-property adaptation {split_prompt}",
            VISIBLE_TASK_PROMPT,
            "successful scripted demonstration",
            "success",
        ]
    return [
        f"push-box physical-property observation {split_prompt}",
        HIDDEN_OBSERVATION_PROMPT,
        "successful scripted observation rollout",
        "success",
    ]


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.calibration_table is not None:
        return build_calibrated_candidates(args)

    init_xys = parse_init_xys(args.init_xys)
    selected_splits = set(str(split) for split in args.splits)
    angle_splits = []
    if "straight" in selected_splits:
        angle_splits.extend(("straight", float(a)) for a in args.straight_angles)
    if "angled" in selected_splits:
        angle_splits.extend(("angled", float(a)) for a in args.angled_angles)
    candidates: list[dict[str, Any]] = []
    for init_id, init_xy in init_xys:
        for friction_mu in [float(mu) for mu in args.frictions]:
            for split, angle_deg in angle_splits:
                for push_distance in [float(d) for d in args.push_distances]:
                    for push_steps in [int(n) for n in args.push_steps]:
                        for push_scale in [float(s) for s in args.push_scales]:
                            speed = float(push_distance) / float(push_steps)
                            candidates.append(
                                {
                                    "init_id": init_id,
                                    "init_xy": init_xy,
                                    "friction_mu": friction_mu,
                                    "split": split,
                                    "angle_deg": angle_deg,
                                    "push_distance": push_distance,
                                    "push_steps": push_steps,
                                    "push_scale": push_scale,
                                    "push_mode": "position",
                                    "action_end": 1.0,
                                    "speed_m_per_step": speed,
                                    "speed_bin": speed_bin(speed, [float(v) for v in args.speed_bin_edges]),
                                    "distance_bin": value_bin(push_distance, [float(v) for v in args.distance_bin_edges], "dist"),
                                    "scale_bin": value_bin(push_scale, [float(v) for v in args.scale_bin_edges], "scale"),
                                    "init_bin": init_id,
                                }
                            )
    if not args.no_shuffle_candidates:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(candidates)
    return candidates


def _load_calibration_recommendations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("recommendations", payload if isinstance(payload, list) else [])
    return [row for row in rows if isinstance(row, dict)]


def _nearest_calibration_rows(rows: list[dict[str, Any]], friction_mu: float, angle_deg: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    min_mu_delta = min(abs(float(row.get("friction_mu", 0.0)) - float(friction_mu)) for row in rows)
    mu_rows = [row for row in rows if abs(abs(float(row.get("friction_mu", 0.0)) - float(friction_mu)) - min_mu_delta) < 1e-9]
    min_angle_delta = min(abs(float(row.get("angle_deg", 0.0)) - float(angle_deg)) for row in mu_rows)
    return [row for row in mu_rows if abs(abs(float(row.get("angle_deg", 0.0)) - float(angle_deg)) - min_angle_delta) < 1e-9]


def _calibrated_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(row.get("mode", "position"))
    base_steps = int(row.get("push_steps", 8))
    base_distance = float(row.get("push_distance_m", row.get("push_distance", 0.14)))
    base_scale = float(row.get("push_scale", 1.0))
    base_action_end = float(row.get("action_end", 1.0 if mode == "position" else 0.4))
    variants = [
        (base_distance, base_steps, base_scale, base_action_end, "calibrated"),
        (base_distance, max(1, base_steps - 1), base_scale, base_action_end, "step_minus"),
        (base_distance, base_steps + 1, base_scale, base_action_end, "step_plus"),
        (base_distance, base_steps, max(1.0, base_scale - 2.0), base_action_end, "scale_minus"),
        (base_distance, base_steps, base_scale + 2.0, base_action_end, "scale_plus"),
        (max(0.10, base_distance - 0.02), base_steps, base_scale, base_action_end, "stroke_minus"),
        (base_distance + 0.02, base_steps, base_scale, base_action_end, "stroke_plus"),
    ]
    if mode == "impulse":
        variants.extend(
            [
                (base_distance, base_steps, base_scale, max(0.2, base_action_end - 0.05), "action_minus"),
                (base_distance, base_steps, base_scale, base_action_end + 0.05, "action_plus"),
            ]
        )
    out = []
    seen: set[tuple[float, int, float, float]] = set()
    for push_distance, push_steps, push_scale, action_end, source in variants:
        key = (round(float(push_distance), 4), int(push_steps), round(float(push_scale), 4), round(float(action_end), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "push_mode": mode,
                "push_distance": float(push_distance),
                "push_steps": int(push_steps),
                "push_scale": float(push_scale),
                "action_end": float(action_end),
                "calibration_source": source,
                "target_hint_m": float(row.get("target_displacement_m", row.get("displacement_m", 0.0))),
                "calibrated_displacement_m": float(row.get("displacement_m", 0.0)),
                "calibrated_valid": bool(row.get("valid", False)),
            }
        )
    return out


def build_calibrated_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    init_xys = parse_init_xys(args.init_xys)
    selected_splits = set(str(split) for split in args.splits)
    angle_splits = []
    if "straight" in selected_splits:
        angle_splits.extend(("straight", float(a)) for a in args.straight_angles)
    if "angled" in selected_splits:
        angle_splits.extend(("angled", float(a)) for a in args.angled_angles)

    calibration_rows = _load_calibration_recommendations(args.calibration_table)
    candidates: list[dict[str, Any]] = []
    for init_id, init_xy in init_xys:
        for friction_mu in [float(mu) for mu in args.frictions]:
            for split, angle_deg in angle_splits:
                rows = _nearest_calibration_rows(calibration_rows, friction_mu, angle_deg)
                for row in sorted(rows, key=lambda item: float(item.get("target_displacement_m", 0.0))):
                    for variant in _calibrated_variants(row):
                        push_distance = float(variant["push_distance"])
                        push_steps = int(variant["push_steps"])
                        push_scale = float(variant["push_scale"])
                        speed = push_distance / float(max(1, push_steps))
                        candidates.append(
                            {
                                "init_id": init_id,
                                "init_xy": init_xy,
                                "friction_mu": friction_mu,
                                "split": split,
                                "angle_deg": angle_deg,
                                "push_distance": push_distance,
                                "push_steps": push_steps,
                                "push_scale": push_scale,
                                "push_mode": str(variant["push_mode"]),
                                "action_end": float(variant["action_end"]),
                                "calibration_source": str(variant["calibration_source"]),
                                "target_hint_m": float(variant["target_hint_m"]),
                                "calibrated_displacement_m": float(variant["calibrated_displacement_m"]),
                                "calibrated_valid": bool(variant["calibrated_valid"]),
                                "speed_m_per_step": speed,
                                "speed_bin": speed_bin(speed, [float(v) for v in args.speed_bin_edges]),
                                "distance_bin": value_bin(push_distance, [float(v) for v in args.distance_bin_edges], "dist"),
                                "scale_bin": value_bin(push_scale, [float(v) for v in args.scale_bin_edges], "scale"),
                                "init_bin": init_id,
                            }
                        )
    if not args.no_shuffle_candidates:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(candidates)
    return candidates


def bucket_key(
    candidate: dict[str, Any],
    dimensions: list[str],
    *,
    displacement_bin: str | None = None,
) -> tuple[str, ...]:
    values = {
        "friction": mu_tag(float(candidate["friction_mu"])),
        "split": str(candidate["split"]),
        "speed_bin": str(candidate["speed_bin"]),
        "distance_bin": str(candidate["distance_bin"]),
        "displacement_bin": displacement_bin,
        "scale_bin": str(candidate["scale_bin"]),
        "init_bin": str(candidate["init_bin"]),
    }
    if "displacement_bin" in dimensions and displacement_bin is None:
        raise ValueError("displacement_bin is required when balancing on actual displacement")
    return tuple(values[dim] for dim in dimensions)


def build_target_buckets(
    candidates: list[dict[str, Any]],
    dimensions: list[str],
    quota: int,
    *,
    displacement_edges: list[float],
    displacement_quotas: list[int] | None,
) -> dict[tuple[str, ...], int]:
    if "displacement_bin" not in dimensions:
        keys = sorted({bucket_key(candidate, dimensions) for candidate in candidates})
        return {key: int(quota) for key in keys}

    displacement_bins = bin_values(displacement_edges, "disp")
    if displacement_quotas is not None and len(displacement_quotas) != len(displacement_bins):
        raise ValueError(
            f"--displacement-bin-quotas must have {len(displacement_bins)} values for "
            f"{len(displacement_edges)} edges, got {len(displacement_quotas)}"
        )

    keys: set[tuple[str, ...]] = set()
    for candidate in candidates:
        for disp_bin in displacement_bins:
            keys.add(bucket_key(candidate, dimensions, displacement_bin=disp_bin))

    quotas = {}
    disp_index = {name: idx for idx, name in enumerate(displacement_bins)}
    disp_dim_index = dimensions.index("displacement_bin")
    for key in sorted(keys):
        if displacement_quotas is None:
            quotas[key] = int(quota)
        else:
            quotas[key] = int(displacement_quotas[disp_index[str(key[disp_dim_index])]])
    return quotas


def possible_bucket_keys(
    candidate: dict[str, Any],
    dimensions: list[str],
    *,
    displacement_edges: list[float],
) -> list[tuple[str, ...]]:
    if "displacement_bin" not in dimensions:
        return [bucket_key(candidate, dimensions)]
    return [
        bucket_key(candidate, dimensions, displacement_bin=disp_bin)
        for disp_bin in bin_values(displacement_edges, "disp")
    ]


def patch_lerobot_video_crf(crf: int) -> None:
    original = lerobot_dataset_module.encode_video_frames

    def encode_with_crf(*args: Any, **kwargs: Any) -> None:
        kwargs["crf"] = int(crf)
        return original(*args, **kwargs)

    lerobot_dataset_module.encode_video_frames = encode_with_crf


def _write_image_for_last_frame(
    dataset: LeRobotDataset,
    key: str,
    frame_index: int,
    image: np.ndarray,
    *,
    jpeg_quality: int,
) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=frame_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=int(jpeg_quality))


def _remove_current_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    episode_index = dataset.episode_buffer["episode_index"]
    for key in dataset.meta.video_keys:
        image_dir = dataset._get_image_file_path(
            episode_index=episode_index,
            image_key=key,
            frame_index=0,
        ).parent
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def collect_case_frames(case: LiberoPushBoxCase, *, repo_root: Path, seed: int) -> dict[str, Any]:
    env = LiberoPushBoxEnv(case, repo_root=repo_root, seed=seed)
    frames: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    phase_counts: dict[str, int] = {}
    success = False
    try:
        obs = env.reset()
        for frame_idx in range(int(case.max_steps)):
            obs_for_frame = obs
            obs, _, _, info = env.step()
            record = info["push_box"]
            action = np.asarray(record["action"], dtype=np.float32)
            agent, wrist = _obs_to_images(obs_for_frame)
            frames.append(
                {
                    "observation.images.image": agent,
                    "observation.images.wrist_image": wrist,
                    "observation.state": _obs_to_state(obs_for_frame),
                    "action": _env_action_to_fastwam_action(action),
                }
            )
            records.append(record)
            phase = str(record.get("phase", "unknown"))
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            success = bool(record["success"])
            if success:
                break
    finally:
        env.close()

    final = records[-1] if records else None
    return {
        "success": bool(success),
        "frames": frames,
        "records": records,
        "phase_counts": phase_counts,
        "steps": len(frames),
        "final": final,
    }


def write_frames_to_dataset(
    dataset: LeRobotDataset,
    *,
    rollout_result: dict[str, Any],
    task: list[str],
    fps: int,
    jpeg_quality: int,
) -> int:
    _remove_current_episode_images(dataset)
    episode_index = int(dataset.meta.total_episodes)
    for frame_idx, frame in enumerate(rollout_result["frames"]):
        dataset.add_frame(frame, task=task, timestamp=float(frame_idx) / float(fps))
        _write_image_for_last_frame(
            dataset,
            "observation.images.image",
            frame_idx,
            frame["observation.images.image"],
            jpeg_quality=jpeg_quality,
        )
        _write_image_for_last_frame(
            dataset,
            "observation.images.wrist_image",
            frame_idx,
            frame["observation.images.wrist_image"],
            jpeg_quality=jpeg_quality,
        )
    dataset.save_episode()
    return episode_index


def make_case_id(candidate: dict[str, Any]) -> str:
    mode = str(candidate.get("push_mode", "position"))[:3]
    action_end = int(round(float(candidate.get("action_end", 1.0)) * 100))
    return (
        f"{candidate['init_id']}_{candidate['split']}_{mu_tag(float(candidate['friction_mu']))}_"
        f"{candidate['speed_bin']}_{candidate['distance_bin']}_{candidate['scale_bin']}_"
        f"{angle_tag(float(candidate['angle_deg']))}_d{int(round(float(candidate['push_distance']) * 100)):02d}_"
        f"n{int(candidate['push_steps']):02d}_s{float(candidate['push_scale']):g}_{mode}_ae{action_end:03d}"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_dataset_metadata(root: Path, metadata: dict[str, Any], episode_rows: list[dict[str, Any]]) -> None:
    (root / "push_box_generation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_jsonl(root / "meta" / "push_box_episode_metadata.jsonl", episode_rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def create_datasets(args: argparse.Namespace) -> dict[tuple[str, str], LeRobotDataset]:
    output_prefix = args.output_prefix.resolve()
    selected_splits = set(str(split) for split in args.splits)
    roots = {
        (domain, split): dataset_root(output_prefix, domain, split)
        for domain in ("observation", "task")
        for split in ("straight", "angled")
        if split in selected_splits
    }
    existing = [root for root in roots.values() if root.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output dataset roots already exist; pass --overwrite: " + ", ".join(str(p) for p in existing))
    if args.overwrite:
        for root in roots.values():
            if root.exists():
                shutil.rmtree(root)

    return {
        key: LeRobotDataset.create(
            repo_id=f"{args.repo_id_prefix}_{'hidden' if key[0] == 'observation' else 'visible'}_{key[1]}",
            root=root,
            fps=int(args.fps),
            features=build_features(int(args.camera_resolution)),
            use_videos=True,
            video_codec=args.video_codec,
            is_compute_episode_stats_image=False,
        )
        for key, root in roots.items()
    }


def main() -> None:
    args = parse_args()
    patch_lerobot_video_crf(int(args.video_crf))
    repo_root = args.repo_root.resolve()
    bddl_dir = args.bddl_dir if args.bddl_dir.is_absolute() else repo_root / args.bddl_dir
    output_prefix = args.output_prefix.resolve()
    datasets = create_datasets(args)
    candidates = build_candidates(args)
    dimensions = list(args.balance_dimensions)
    displacement_edges = [float(v) for v in args.displacement_bin_edges]
    target_buckets = build_target_buckets(
        candidates,
        dimensions,
        int(args.pairs_per_bucket),
        displacement_edges=displacement_edges,
        displacement_quotas=[int(v) for v in args.displacement_bin_quotas]
        if args.displacement_bin_quotas is not None
        else None,
    )
    target_pair_count = int(sum(target_buckets.values()))
    accepted_buckets = {key: 0 for key in target_buckets}
    trial_buckets = {key: 0 for key in target_buckets}
    accepted_frictions = {mu_tag(float(mu)): 0 for mu in args.frictions}

    subset_rows: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in datasets}
    subset_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for key in datasets:
        subset_metadata[key] = {
            "created_at": dt.datetime.now().isoformat(),
            "dataset_type": "libero_push_box_rollout_target_lerobot",
            "domain": key[0],
            "target_visible": bool(key[0] == "task"),
            "split": key[1],
            "camera_resolution": int(args.camera_resolution),
            "fps": int(args.fps),
            "video_codec": str(args.video_codec),
            "video_crf": int(args.video_crf),
            "jpeg_quality": int(args.jpeg_quality),
            "seed": int(args.seed),
            "output_root": str(dataset_root(output_prefix, key[0], key[1])),
            "episodes": [],
        }

    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_rollout_target_lerobot_collection",
        "output_prefix": str(output_prefix),
        "bddl_dir": str(bddl_dir),
        "balance_dimensions": dimensions,
        "target_buckets": {"|".join(key): value for key, value in target_buckets.items()},
        "accepted_buckets": {},
        "accepted_frictions": {},
        "trial_buckets": {},
        "pairs": [],
        "rejected": [],
        "subset_roots": {
            f"{domain}_{split}": str(dataset_root(output_prefix, domain, split))
            for domain, split in datasets
        },
        "generation_args": to_jsonable(vars(args)),
    }
    manifest_path = output_prefix.parent / f"{output_prefix.name}_manifest.json"

    def autosave() -> None:
        manifest["accepted_buckets"] = {"|".join(key): value for key, value in accepted_buckets.items()}
        manifest["accepted_frictions"] = dict(accepted_frictions)
        manifest["trial_buckets"] = {"|".join(key): value for key, value in trial_buckets.items()}
        manifest["missing_buckets"] = {
            "|".join(key): target_buckets[key] - accepted_buckets.get(key, 0)
            for key in sorted(target_buckets)
            if accepted_buckets.get(key, 0) < target_buckets[key]
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for key, metadata in subset_metadata.items():
            write_dataset_metadata(dataset_root(output_prefix, key[0], key[1]), metadata, subset_rows[key])

    try:
        for candidate in candidates:
            candidate_possible_keys = possible_bucket_keys(
                candidate,
                dimensions,
                displacement_edges=displacement_edges,
            )
            if all(
                accepted_buckets.get(possible_key, 0) >= target_buckets.get(possible_key, 0)
                for possible_key in candidate_possible_keys
            ):
                continue

            base_id = make_case_id(candidate)
            init_xy = tuple(float(v) for v in candidate["init_xy"])
            direction = direction_xy(float(candidate["angle_deg"]))
            dummy_target_xy = (
                float(init_xy[0] + direction[0] * float(args.dummy_target_distance)),
                float(init_xy[1] + direction[1] * float(args.dummy_target_distance)),
            )
            probe_bddl = write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / "probe",
                geometry_id=f"{base_id}_probe_invisible",
                init_xy=init_xy,
                target_xy=dummy_target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            probe = build_probe_case(
                case_id=f"probe_{base_id}",
                friction_mu=float(candidate["friction_mu"]),
                split=str(candidate["split"]),
                angle_deg=float(candidate["angle_deg"]),
                push_steps=int(candidate["push_steps"]),
                push_distance=float(candidate["push_distance"]),
                push_scale=float(candidate["push_scale"]),
                init_xy=init_xy,
                target_xy=dummy_target_xy,
                bddl_file=probe_bddl,
                max_steps=int(args.max_steps),
                camera_resolution=int(args.probe_resolution),
                target_radius=float(args.target_radius),
            )
            probe = replace(
                probe,
                pusher_push_mode=str(candidate.get("push_mode", "position")),
                pusher_push_action_end=float(candidate.get("action_end", 1.0)),
                pusher_push_controller_scale=float(candidate["push_scale"]),
                pusher_max_push_controller_scale=max(20.0, float(candidate["push_scale"])),
            )
            result = rollout(probe, repo_root=repo_root, seed=int(args.seed))
            accepted, metrics = accept_rollout(
                init_xy=init_xy,
                angle_deg=float(candidate["angle_deg"]),
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
                    "speed_bin": str(candidate["speed_bin"]),
                    "distance_bin": str(candidate["distance_bin"]),
                    "displacement_bin": value_bin(
                        float(metrics["displacement_m"]),
                        displacement_edges,
                        "disp",
                    ),
                    "scale_bin": str(candidate["scale_bin"]),
                }
            )
            key = bucket_key(candidate, dimensions, displacement_bin=str(metrics["displacement_bin"]))
            if int(args.max_trials_per_bucket) > 0 and trial_buckets.get(key, 0) >= int(args.max_trials_per_bucket):
                manifest["rejected"].append(
                    {"case_id": base_id, "reason": "bucket_trial_limit", "candidate": candidate, "metrics": metrics}
                )
                continue
            trial_buckets[key] = trial_buckets.get(key, 0) + 1

            rejected_reason = ""
            if not accepted:
                rejected_reason = "acceptance"
            elif key not in target_buckets:
                rejected_reason = "bucket_not_requested"
            elif accepted_buckets.get(key, 0) >= target_buckets.get(key, 0):
                rejected_reason = "bucket_full"
            elif (
                int(args.max_pairs_per_friction) > 0
                and accepted_frictions.get(mu_tag(float(candidate["friction_mu"])), 0)
                >= int(args.max_pairs_per_friction)
            ):
                rejected_reason = "friction_cap"
            elif int(result["push_backward_action_count"]) > 0:
                rejected_reason = "backward_action"
            elif int(result["push_eef_backward_steps"]) > 0:
                rejected_reason = "backward_eef"
            elif float(result["max_eef_step_m"]) > float(args.max_eef_step):
                rejected_reason = "eef_step"

            if (
                not accepted
                or key not in target_buckets
                or accepted_buckets.get(key, 0) >= target_buckets.get(key, 0)
                or (
                    int(args.max_pairs_per_friction) > 0
                    and accepted_frictions.get(mu_tag(float(candidate["friction_mu"])), 0)
                    >= int(args.max_pairs_per_friction)
                )
                or int(result["push_backward_action_count"]) > 0
                or int(result["push_eef_backward_steps"]) > 0
                or float(result["max_eef_step_m"]) > float(args.max_eef_step)
            ):
                manifest["rejected"].append(
                    {"case_id": base_id, "reason": rejected_reason, "candidate": candidate, "metrics": metrics}
                )
                if int(args.progress_every) > 0 and len(manifest["rejected"]) % int(args.progress_every) == 0:
                    print(
                        f"rejected={len(manifest['rejected'])} accepted={len(manifest['pairs'])} "
                        f"latest={base_id} reason={rejected_reason} "
                        f"disp={metrics['displacement_m'] * 100:.1f}cm "
                        f"speed={metrics['final_speed_mps']:.4f}",
                        flush=True,
                    )
                continue

            target_xy = (float(result["final_xy"][0]), float(result["final_xy"][1]))
            visible_bddl = write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / "task",
                geometry_id=base_id,
                init_xy=init_xy,
                target_xy=target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                target_rgba=(0.0, 0.8, 0.2, 0.45),
            )
            invisible_bddl = write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / "observation",
                geometry_id=base_id,
                init_xy=init_xy,
                target_xy=target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            task_case = replace(
                probe,
                case_id=f"task_{base_id}",
                domain="task",
                geometry_id=base_id,
                bddl_file=visible_bddl,
                target_xy=target_xy,
                target_distance=float(metrics["displacement_m"]),
                camera_resolution=int(args.camera_resolution),
            )
            observation_case = replace(
                task_case,
                case_id=f"observation_{base_id}",
                domain="observation",
                bddl_file=invisible_bddl,
            )

            obs_rollout = collect_case_frames(observation_case, repo_root=repo_root, seed=int(args.seed))
            task_rollout = collect_case_frames(task_case, repo_root=repo_root, seed=int(args.seed))
            if not obs_rollout["success"] or not task_rollout["success"]:
                manifest["rejected"].append(
                    {
                        "case_id": base_id,
                        "candidate": candidate,
                        "metrics": metrics,
                        "observation_success": bool(obs_rollout["success"]),
                        "task_success": bool(task_rollout["success"]),
                    }
                )
                continue

            split = str(candidate["split"])
            episode_indices = {
                "observation": write_frames_to_dataset(
                    datasets[("observation", split)],
                    rollout_result=obs_rollout,
                    task=prompt_for_case("observation", split),
                    fps=int(args.fps),
                    jpeg_quality=int(args.jpeg_quality),
                ),
                "task": write_frames_to_dataset(
                    datasets[("task", split)],
                    rollout_result=task_rollout,
                    task=prompt_for_case("task", split),
                    fps=int(args.fps),
                    jpeg_quality=int(args.jpeg_quality),
                ),
            }
            pair_record = {
                "pair_id": base_id,
                "bucket": "|".join(key),
                "candidate": candidate,
                "target_xy": list(target_xy),
                "metrics": metrics,
                "observation_case": observation_case.as_dict(),
                "task_case": task_case.as_dict(),
                "episode_indices": episode_indices,
                "steps": {
                    "observation": int(obs_rollout["steps"]),
                    "task": int(task_rollout["steps"]),
                },
                "phase_counts": {
                    "observation": obs_rollout["phase_counts"],
                    "task": task_rollout["phase_counts"],
                },
            }
            manifest["pairs"].append(pair_record)
            accepted_buckets[key] = accepted_buckets.get(key, 0) + 1
            accepted_frictions[mu_tag(float(candidate["friction_mu"]))] = (
                accepted_frictions.get(mu_tag(float(candidate["friction_mu"])), 0) + 1
            )

            for domain, case, rollout_result in (
                ("observation", observation_case, obs_rollout),
                ("task", task_case, task_rollout),
            ):
                subset_key = (domain, split)
                row = {
                    "episode_index": int(episode_indices[domain]),
                    "pair_id": base_id,
                    "case_id": case.case_id,
                    "domain": domain,
                    "target_visible": bool(domain == "task"),
                    "split": split,
                    "friction_mu": float(candidate["friction_mu"]),
                    "init_xy": list(init_xy),
                    "target_xy": list(target_xy),
                    "angle_deg": float(candidate["angle_deg"]),
                    "push_distance_x": float(candidate["push_distance"]),
                    "pusher_push_steps": int(candidate["push_steps"]),
                    "pusher_push_mode": str(candidate.get("push_mode", "position")),
                    "pusher_push_action_end": float(candidate.get("action_end", 1.0)),
                    "pusher_push_controller_scale": float(candidate["push_scale"]),
                    "calibration_source": candidate.get("calibration_source"),
                    "target_hint_m": candidate.get("target_hint_m"),
                    "calibrated_displacement_m": candidate.get("calibrated_displacement_m"),
                    "speed_m_per_step": float(candidate["speed_m_per_step"]),
                    "speed_bin": str(candidate["speed_bin"]),
                    "distance_bin": str(candidate["distance_bin"]),
                    "scale_bin": str(candidate["scale_bin"]),
                    "metrics": metrics,
                    "phase_counts": rollout_result["phase_counts"],
                    "steps": int(rollout_result["steps"]),
                    "bddl_file": case.bddl_file,
                }
                subset_rows[subset_key].append(row)
                subset_metadata[subset_key]["episodes"].append(row)

            print(
                f"accepted {len(manifest['pairs']):04d}/{target_pair_count:04d} "
                f"{base_id} bucket={'|'.join(key)} disp={metrics['displacement_m'] * 100:.1f}cm "
                f"episodes={episode_indices}",
                flush=True,
            )
            if int(args.autosave_every) > 0 and len(manifest["pairs"]) % int(args.autosave_every) == 0:
                autosave()
            if int(args.max_pairs) > 0 and len(manifest["pairs"]) >= int(args.max_pairs):
                break
            if all(accepted_buckets[bucket] >= target for bucket, target in target_buckets.items()):
                break
    finally:
        autosave()

    missing = {
        key: target_buckets[key] - accepted_buckets.get(key, 0)
        for key in sorted(target_buckets)
        if accepted_buckets.get(key, 0) < target_buckets[key]
    }
    if missing and not bool(args.allow_incomplete) and int(args.max_pairs) <= 0:
        raise RuntimeError(f"Missing {len(missing)} balanced buckets. Manifest: {manifest_path}")
    if missing:
        print(f"warning: missing {len(missing)} balanced buckets; manifest={manifest_path}", flush=True)
    print(f"manifest={manifest_path}")
    for key in sorted(datasets):
        print(f"{key[0]} {key[1]} root={dataset_root(output_prefix, key[0], key[1])}")


if __name__ == "__main__":
    main()

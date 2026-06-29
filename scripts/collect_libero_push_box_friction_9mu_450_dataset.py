#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


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

from collect_libero_push_box_rollout_target_lerobot_dataset import (  # noqa: E402
    build_features,
    collect_case_frames,
    patch_lerobot_video_crf,
    prompt_for_case,
    to_jsonable,
    value_bin,
    write_dataset_metadata,
    write_frames_to_dataset,
)
from generate_libero_push_box_rollout_target_dataset import (  # noqa: E402
    accept_rollout,
    angle_tag,
    build_probe_case,
    direction_xy,
    mu_tag,
    rollout,
    speed_bin,
)


DISPLACEMENT_BUCKET_LABELS = {
    "disp_00": "short",
    "disp_01": "mid",
    "disp_02": "long",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a 9-friction hidden LIBERO push-box train/test LeRobot dataset."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FASTWAM_ROOT / "data" / "libero_push_box_friction_9mu_450",
    )
    parser.add_argument(
        "--bddl-dir",
        type=Path,
        default=REPO_ROOT / "generated_bddl" / "libero_push_box_friction_9mu_450",
    )
    parser.add_argument(
        "--calibration-table",
        type=Path,
        default=REPO_ROOT / "configs" / "libero_push_box_push_calibration_table.json",
    )
    parser.add_argument(
        "--frictions",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.15, 0.2],
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
    parser.add_argument("--contact-offsets-y", type=float, nargs="+", default=[-0.025, -0.012, 0.0, 0.012, 0.025])
    parser.add_argument("--yaw-degs", type=float, nargs="+", default=[-12.0, -6.0, 0.0, 6.0, 12.0])
    parser.add_argument("--train-displacement-quotas", type=int, nargs="+", default=[4, 5, 4])
    parser.add_argument("--test-displacement-quotas", type=int, nargs="+", default=[3, 6, 3])
    parser.add_argument("--displacement-bin-edges", type=float, nargs="+", default=[0.20, 0.35])
    parser.add_argument("--dummy-target-distance", type=float, default=0.42)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--camera-resolution", type=int, default=224)
    parser.add_argument("--probe-resolution", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--video-codec", default="h264", choices=["h264", "hevc", "libsvtav1", "h264_nvenc"])
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--jpeg-quality", type=int, default=98)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--autosave-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-trials", type=int, default=30000)
    parser.add_argument("--min-displacement", type=float, default=0.08)
    parser.add_argument("--max-displacement", type=float, default=0.56)
    parser.add_argument("--max-final-speed", type=float, default=0.045)
    parser.add_argument("--max-eef-step", type=float, default=0.070)
    parser.add_argument("--x-bounds", type=float, nargs=2, default=(-0.30, 0.40))
    parser.add_argument("--y-bounds", type=float, nargs=2, default=(-0.32, 0.32))
    return parser.parse_args()


def parse_init_xys(values: list[list[str]] | None) -> list[tuple[str, tuple[float, float]]]:
    if values is None:
        flat_values = [
            "-0.255,-0.055",
            "-0.245,-0.035",
            "-0.235,-0.015",
            "-0.225,0.005",
            "-0.255,0.020",
            "-0.245,0.040",
            "-0.235,-0.075",
            "-0.225,-0.055",
        ]
    else:
        flat_values = [item for group in values for item in group]
    parsed = []
    for idx, value in enumerate(flat_values):
        parts = [part.strip() for part in str(value).split(",")]
        if len(parts) != 2:
            raise ValueError(f"--init-xys values must be 'x,y', got {value!r}")
        parsed.append((f"i{idx:02d}", (float(parts[0]), float(parts[1]))))
    return parsed


def _region_bounds(center: tuple[float, float], half_size: float) -> tuple[float, float, float, float]:
    x, y = center
    return x - half_size, y - half_size, x + half_size, y + half_size


def _bddl_text(
    *,
    init_xy: tuple[float, float],
    target_xy: tuple[float, float],
    init_half_size: float,
    target_radius: float,
    yaw_deg: float,
    target_rgba: tuple[float, float, float, float],
) -> str:
    ix0, iy0, ix1, iy1 = _region_bounds(init_xy, init_half_size)
    tx0, ty0, tx1, ty1 = _region_bounds(target_xy, target_radius)
    yaw_rad = math.radians(float(yaw_deg))
    rgba = " ".join(f"{value:.3f}" for value in target_rgba)
    return f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language observe how the cream cheese box slides after a robot push on the table)
    (:regions
      (box_init_region
          (:target main_table)
          (:ranges (
              ({ix0:.4f} {iy0:.4f} {ix1:.4f} {iy1:.4f})
            )
          )
          (:yaw_rotation (
              ({yaw_rad:.6f} {yaw_rad:.6f})
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
              ({rgba})
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
    yaw_deg: float,
    target_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> str:
    bddl_dir.mkdir(parents=True, exist_ok=True)
    path = bddl_dir / f"push_box_{geometry_id}.bddl"
    path.write_text(
        _bddl_text(
            init_xy=init_xy,
            target_xy=target_xy,
            init_half_size=init_half_size,
            target_radius=target_radius,
            yaw_deg=yaw_deg,
            target_rgba=target_rgba,
        ),
        encoding="utf-8",
    )
    return str(path.resolve().relative_to(repo_root.resolve()))


def _load_calibration_recommendations(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("recommendations", payload if isinstance(payload, list) else [])
    return [row for row in rows if isinstance(row, dict) and bool(row.get("valid", True))]


def _nearest_calibration_rows(rows: list[dict[str, Any]], friction_mu: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    min_mu_delta = min(abs(float(row.get("friction_mu", 0.0)) - float(friction_mu)) for row in rows)
    return [
        row
        for row in rows
        if abs(abs(float(row.get("friction_mu", 0.0)) - float(friction_mu)) - min_mu_delta) < 1e-9
    ]


def _calibrated_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(row.get("mode", "position"))
    base_steps = int(row.get("push_steps", 8))
    base_distance = float(row.get("push_distance_m", row.get("push_distance", 0.16)))
    base_scale = float(row.get("push_scale", 4.0))
    base_action_end = float(row.get("action_end", 1.0 if mode == "position" else 0.4))
    variants = [
        (base_distance, base_steps, base_scale, base_action_end, "calibrated"),
        (base_distance, max(1, base_steps - 1), base_scale, base_action_end, "step_minus"),
        (base_distance, base_steps + 1, base_scale, base_action_end, "step_plus"),
        (max(0.10, base_distance - 0.02), base_steps, base_scale, base_action_end, "stroke_minus"),
        (base_distance + 0.02, base_steps, base_scale, base_action_end, "stroke_plus"),
        (base_distance, base_steps, max(1.0, base_scale - 2.0), base_action_end, "scale_minus"),
        (base_distance, base_steps, base_scale + 2.0, base_action_end, "scale_plus"),
    ]
    out = []
    seen: set[tuple[float, int, float, float]] = set()
    for push_distance, push_steps, push_scale, action_end, source in variants:
        key = (
            round(float(push_distance), 4),
            int(push_steps),
            round(float(push_scale), 4),
            round(float(action_end), 4),
        )
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
                "calibrated_friction_mu": float(row.get("friction_mu", 0.0)),
            }
        )
    return out


def fallback_variants() -> list[dict[str, Any]]:
    presets = [
        ("short_a", 0.14, 32, 4.0),
        ("short_b", 0.16, 28, 4.0),
        ("short_c", 0.18, 24, 6.0),
        ("mid_a", 0.20, 16, 6.0),
        ("mid_b", 0.22, 12, 8.0),
        ("mid_c", 0.24, 10, 10.0),
        ("long_a", 0.26, 8, 12.0),
        ("long_b", 0.30, 7, 12.0),
        ("long_c", 0.34, 6, 14.0),
    ]
    return [
        {
            "push_mode": "position",
            "push_distance": dist,
            "push_steps": steps,
            "push_scale": scale,
            "action_end": 1.0,
            "calibration_source": name,
            "target_hint_m": 0.0,
            "calibrated_displacement_m": 0.0,
            "calibrated_friction_mu": 0.0,
        }
        for name, dist, steps, scale in presets
    ]


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    init_xys = parse_init_xys(args.init_xys)
    calibration_rows = _load_calibration_recommendations(args.calibration_table)
    candidates: list[dict[str, Any]] = []
    angle_splits = [("straight", float(angle)) for angle in args.straight_angles]
    angle_splits.extend(("angled", float(angle)) for angle in args.angled_angles)
    for init_id, init_xy in init_xys:
        for friction_mu in [float(mu) for mu in args.frictions]:
            rows = _nearest_calibration_rows(calibration_rows, friction_mu)
            variants = [variant for row in rows for variant in _calibrated_variants(row)]
            if not variants:
                variants = fallback_variants()
            for split, angle_deg in angle_splits:
                for contact_offset_y in [float(v) for v in args.contact_offsets_y]:
                    for yaw_deg in [float(v) for v in args.yaw_degs]:
                        for variant in variants:
                            push_distance = float(variant["push_distance"])
                            push_steps = int(variant["push_steps"])
                            push_scale = float(variant["push_scale"])
                            speed = push_distance / float(max(1, push_steps))
                            sign = "zero"
                            if split == "angled":
                                sign = "pos" if angle_deg > 0.0 else "neg"
                            candidates.append(
                                {
                                    "init_id": init_id,
                                    "init_xy": init_xy,
                                    "friction_mu": friction_mu,
                                    "split": split,
                                    "angle_deg": angle_deg,
                                    "angle_sign": sign,
                                    "yaw_deg": yaw_deg,
                                    "contact_offset_y": contact_offset_y,
                                    "approach_offset_xy": (-0.130, contact_offset_y),
                                    "contact_offset_xy": (-0.105, contact_offset_y),
                                    "push_distance": push_distance,
                                    "push_steps": push_steps,
                                    "push_scale": push_scale,
                                    "push_mode": str(variant["push_mode"]),
                                    "action_end": float(variant["action_end"]),
                                    "calibration_source": str(variant["calibration_source"]),
                                    "target_hint_m": float(variant["target_hint_m"]),
                                    "calibrated_displacement_m": float(variant["calibrated_displacement_m"]),
                                    "calibrated_friction_mu": float(variant["calibrated_friction_mu"]),
                                    "speed_m_per_step": speed,
                                    "speed_bin": speed_bin(speed, [0.006, 0.012, 0.020, 0.040]),
                                    "distance_bin": value_bin(push_distance, [0.18, 0.26, 0.32], "dist"),
                                    "scale_bin": value_bin(push_scale, [4.0, 8.0, 12.0], "scale"),
                                }
                            )
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(candidates)
    return candidates


def make_case_id(candidate: dict[str, Any]) -> str:
    mode = str(candidate.get("push_mode", "position"))[:3]
    action_end = int(round(float(candidate.get("action_end", 1.0)) * 100))
    cy = int(round((float(candidate["contact_offset_y"]) + 0.1) * 1000.0))
    yaw = int(round(float(candidate["yaw_deg"])))
    return (
        f"{candidate['init_id']}_{candidate['split']}_{mu_tag(float(candidate['friction_mu']))}_"
        f"{candidate['angle_sign']}_{angle_tag(float(candidate['angle_deg']))}_"
        f"{candidate['speed_bin']}_{candidate['distance_bin']}_{candidate['scale_bin']}_"
        f"cy{cy:03d}_yaw{yaw:+03d}_d{int(round(float(candidate['push_distance']) * 100)):02d}_"
        f"n{int(candidate['push_steps']):02d}_s{float(candidate['push_scale']):g}_{mode}_ae{action_end:03d}"
    )


def displacement_bins(args: argparse.Namespace) -> list[str]:
    return [f"disp_{idx:02d}" for idx in range(len(args.displacement_bin_edges) + 1)]


def split_sign_counts(total: int, *, extra_sign: str) -> dict[str, int]:
    if total <= 0:
        return {"neg": 0, "pos": 0}
    base = total // 2
    out = {"neg": base, "pos": base}
    if total % 2:
        out[extra_sign] += 1
    return out


def extra_sign_for(friction_index: int, subset: str, disp_bin: str) -> str:
    major = "pos" if friction_index % 2 == 0 else "neg"
    minor = "neg" if major == "pos" else "pos"
    if subset == "train" and disp_bin == "disp_01":
        return major
    if subset == "test" and disp_bin == "disp_00":
        return minor
    if subset == "test" and disp_bin == "disp_02":
        return major
    return major


def build_target_quotas(args: argparse.Namespace) -> dict[tuple[str, str, str, str, str], int]:
    bins = displacement_bins(args)
    if len(args.train_displacement_quotas) != len(bins) or len(args.test_displacement_quotas) != len(bins):
        raise ValueError(
            f"Train/test displacement quotas must each have {len(bins)} values for edges "
            f"{list(args.displacement_bin_edges)}."
        )
    high_split_quotas = [int(value) for value in args.train_displacement_quotas]
    low_split_quotas = [int(value) for value in args.test_displacement_quotas]
    quotas: dict[tuple[str, str, str, str, str], int] = {}
    for fidx, friction_mu in enumerate([float(mu) for mu in args.frictions]):
        friction = mu_tag(friction_mu)
        for split_idx, split in enumerate(("straight", "angled")):
            train_gets_extra = (fidx + split_idx) % 2 == 0
            train_quotas = high_split_quotas if train_gets_extra else low_split_quotas
            test_quotas = low_split_quotas if train_gets_extra else high_split_quotas
            for disp_bin, train_q, test_q in zip(bins, train_quotas, test_quotas):
                for subset, total_q in (("train", int(train_q)), ("test", int(test_q))):
                    if total_q <= 0:
                        continue
                    if split == "straight":
                        quotas[(friction, split, "zero", disp_bin, subset)] = total_q
                        continue
                    sign_counts = split_sign_counts(
                        total_q,
                        extra_sign=extra_sign_for(fidx, subset, disp_bin),
                    )
                    for sign, count in sign_counts.items():
                        if count > 0:
                            quotas[(friction, split, sign, disp_bin, subset)] = count
    return quotas


def choose_bucket_subset(
    candidate: dict[str, Any],
    displacement_bin: str,
    counts: dict[tuple[str, str, str, str, str], int],
    quotas: dict[tuple[str, str, str, str, str], int],
    rng: np.random.Generator,
) -> tuple[str, tuple[str, str, str, str, str]] | tuple[None, None]:
    base = (
        mu_tag(float(candidate["friction_mu"])),
        str(candidate["split"]),
        str(candidate["angle_sign"]),
        str(displacement_bin),
    )
    options = []
    weights = []
    for subset in ("train", "test"):
        key = (*base, subset)
        remaining = int(quotas.get(key, 0)) - int(counts.get(key, 0))
        if remaining > 0:
            options.append((subset, key))
            weights.append(remaining)
    if not options:
        return None, None
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    idx = int(rng.choice(len(options), p=weights_arr))
    return options[idx]


def candidate_base_has_open_quota(
    candidate: dict[str, Any],
    counts: dict[tuple[str, str, str, str, str], int],
    quotas: dict[tuple[str, str, str, str, str], int],
) -> bool:
    base = (
        mu_tag(float(candidate["friction_mu"])),
        str(candidate["split"]),
        str(candidate["angle_sign"]),
    )
    for key, target in quotas.items():
        if key[:3] == base and int(counts.get(key, 0)) < int(target):
            return True
    return False


def create_dataset(root: Path, *, repo_id: str, args: argparse.Namespace) -> LeRobotDataset:
    if root.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"{root} already exists; pass --overwrite")
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=int(args.fps),
        features=build_features(int(args.camera_resolution)),
        use_videos=True,
        video_codec=str(args.video_codec),
        is_compute_episode_stats_image=False,
    )


def phase_boundaries(candidate: dict[str, Any]) -> dict[str, int]:
    approach = 30
    descend = 40
    push = int(candidate["push_steps"])
    retreat = 60
    return {
        "approach_start": 0,
        "descend_start": approach,
        "push_start": approach + descend,
        "push_end": approach + descend + push,
        "settle_start": approach + descend + push + retreat,
    }


def capture_contact_sample(
    contact_samples: dict[tuple[str, str, str], Image.Image],
    *,
    row: dict[str, Any],
    rollout_result: dict[str, Any],
) -> None:
    key = (
        mu_tag(float(row["friction_mu"])),
        str(row["split"]),
        str(row["metrics"]["displacement_bin"]),
    )
    if key in contact_samples or not rollout_result["frames"]:
        return
    push_start = int(row["push_start"])
    push_end = int(row["push_end"])
    frame_idx = min(max(0, (push_start + push_end) // 2), len(rollout_result["frames"]) - 1)
    frame = rollout_result["frames"][frame_idx]["observation.images.image"]
    contact_samples[key] = Image.fromarray(frame).resize((160, 160))


def write_contact_sheet(
    path: Path,
    *,
    contact_samples: dict[tuple[str, str, str], Image.Image],
    frictions: list[float],
) -> None:
    cols = [
        ("straight", "disp_00"),
        ("straight", "disp_01"),
        ("straight", "disp_02"),
        ("angled", "disp_00"),
        ("angled", "disp_01"),
        ("angled", "disp_02"),
    ]
    tile_w, tile_h = 160, 186
    label_h = 26
    header_h = 28
    sheet = Image.new("RGB", (tile_w * (len(cols) + 1), header_h + tile_h * len(frictions)), "white")
    draw = ImageDraw.Draw(sheet)
    for cidx, (split, disp_bin) in enumerate(cols, start=1):
        draw.text((cidx * tile_w + 4, 6), f"{split} {DISPLACEMENT_BUCKET_LABELS.get(disp_bin, disp_bin)}", fill=(0, 0, 0))
    for ridx, friction_mu in enumerate(frictions):
        y0 = header_h + ridx * tile_h
        draw.text((4, y0 + 8), f"mu={friction_mu:g}", fill=(0, 0, 0))
        for cidx, (split, disp_bin) in enumerate(cols, start=1):
            x0 = cidx * tile_w
            key = (mu_tag(float(friction_mu)), split, disp_bin)
            image = contact_samples.get(key)
            if image is None:
                image = Image.new("RGB", (160, 160), (235, 235, 235))
                ImageDraw.Draw(image).text((42, 70), "missing", fill=(80, 80, 80))
            sheet.paste(image, (x0, y0 + label_h))
            draw.rectangle((x0, y0 + label_h, x0 + 159, y0 + label_h + 159), outline=(180, 180, 180))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def summarize(
    *,
    output_root: Path,
    manifest: dict[str, Any],
    subset_rows: dict[str, list[dict[str, Any]]],
    target_quotas: dict[tuple[str, str, str, str, str], int],
    counts: dict[tuple[str, str, str, str, str], int],
) -> dict[str, Any]:
    rows = [row for subset in ("train", "test") for row in subset_rows[subset]]
    count_by_friction_split_subset = Counter(
        (mu_tag(float(row["friction_mu"])), row["split"], row["subset"]) for row in rows
    )
    count_by_disp_split_subset = Counter(
        (row["metrics"]["displacement_bin"], row["split"], row["subset"]) for row in rows
    )
    count_by_angle_sign = Counter(
        (mu_tag(float(row["friction_mu"])), row["split"], row["angle_sign"]) for row in rows
    )
    lengths = [int(row["steps"]) for row in rows]
    phase_values: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for phase, value in row.get("phase_counts", {}).items():
            phase_values[str(phase)].append(int(value))
    rejected_by_reason = Counter(item["reason"] for item in manifest.get("rejected", []))
    missing = {
        "|".join(key): int(target_quotas[key]) - int(counts.get(key, 0))
        for key in sorted(target_quotas)
        if int(counts.get(key, 0)) < int(target_quotas[key])
    }
    summary = {
        "episodes": len(rows),
        "train_count": len(subset_rows["train"]),
        "test_count": len(subset_rows["test"]),
        "target_count": int(sum(target_quotas.values())),
        "rejected_count": len(manifest.get("rejected", [])),
        "rejected_by_reason": dict(rejected_by_reason),
        "missing_buckets": missing,
        "count_by_friction_split_subset": {"|".join(k): v for k, v in sorted(count_by_friction_split_subset.items())},
        "count_by_displacement_split_subset": {"|".join(k): v for k, v in sorted(count_by_disp_split_subset.items())},
        "count_by_angle_sign": {"|".join(k): v for k, v in sorted(count_by_angle_sign.items())},
        "episode_length": {
            "min": int(min(lengths)) if lengths else None,
            "mean": float(np.mean(lengths)) if lengths else None,
            "max": int(max(lengths)) if lengths else None,
        },
        "phase_count_stats": {
            phase: {
                "min": int(min(values)),
                "mean": float(np.mean(values)),
                "max": int(max(values)),
            }
            for phase, values in sorted(phase_values.items())
            if values
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# libero_push_box_friction_9mu_450 Summary",
        "",
        f"- episodes: {summary['episodes']} / {summary['target_count']}",
        f"- train: {summary['train_count']}",
        f"- test: {summary['test_count']}",
        f"- rejected: {summary['rejected_count']}",
        f"- missing buckets: {len(missing)}",
        "",
        "## Count by friction, split, subset",
        "",
        "| friction | split | subset | count |",
        "|---|---|---:|---:|",
    ]
    for key, value in sorted(count_by_friction_split_subset.items()):
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {value} |")
    lines.extend(["", "## Count by displacement bucket, split, subset", "", "| bucket | split | subset | count |", "|---|---|---:|---:|"])
    for key, value in sorted(count_by_disp_split_subset.items()):
        lines.append(f"| {DISPLACEMENT_BUCKET_LABELS.get(key[0], key[0])} | {key[1]} | {key[2]} | {value} |")
    lines.extend(["", "## Episode length", ""])
    lines.append(json.dumps(summary["episode_length"], indent=2))
    lines.extend(["", "## Phase count stats", ""])
    lines.append(json.dumps(summary["phase_count_stats"], indent=2))
    lines.extend(["", "## Rejections", ""])
    lines.append(json.dumps(summary["rejected_by_reason"], indent=2))
    if missing:
        lines.extend(["", "## Missing buckets", ""])
        lines.append(json.dumps(missing, indent=2))
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    patch_lerobot_video_crf(int(args.video_crf))
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    bddl_dir = args.bddl_dir if args.bddl_dir.is_absolute() else repo_root / args.bddl_dir
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed) + 1)

    target_quotas = build_target_quotas(args)
    counts = {key: 0 for key in target_quotas}
    target_total = int(sum(target_quotas.values()))
    train_target = sum(value for key, value in target_quotas.items() if key[-1] == "train")
    test_target = sum(value for key, value in target_quotas.items() if key[-1] == "test")

    datasets = {
        "train": create_dataset(
            output_root / "train_lerobot",
            repo_id="libero_push_box_friction_9mu_450_train",
            args=args,
        ),
        "test": create_dataset(
            output_root / "test_lerobot",
            repo_id="libero_push_box_friction_9mu_450_test",
            args=args,
        ),
    }
    subset_rows: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    subset_metadata = {
        subset: {
            "created_at": dt.datetime.now().isoformat(),
            "dataset_type": "libero_push_box_hidden_friction_9mu_lerobot",
            "subset": subset,
            "camera_resolution": int(args.camera_resolution),
            "fps": int(args.fps),
            "video_codec": str(args.video_codec),
            "video_crf": int(args.video_crf),
            "jpeg_quality": int(args.jpeg_quality),
            "episodes": [],
        }
        for subset in ("train", "test")
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_hidden_friction_9mu_450",
        "output_root": str(output_root),
        "train_count_target": int(train_target),
        "test_count_target": int(test_target),
        "total_count_target": int(target_total),
        "frictions": [float(mu) for mu in args.frictions],
        "straight_episodes_per_friction": int(sum(args.train_displacement_quotas) + sum(args.test_displacement_quotas)),
        "angled_episodes_per_friction": int(sum(args.train_displacement_quotas) + sum(args.test_displacement_quotas)),
        "displacement_bin_edges": [float(v) for v in args.displacement_bin_edges],
        "displacement_bucket_labels": DISPLACEMENT_BUCKET_LABELS,
        "target_quotas": {"|".join(key): value for key, value in sorted(target_quotas.items())},
        "accepted_counts": {},
        "episodes": [],
        "rejected": [],
        "generation_args": to_jsonable(vars(args)),
    }
    manifest_path = output_root / "manifest.json"
    contact_samples: dict[tuple[str, str, str], Image.Image] = {}

    def autosave() -> None:
        manifest["accepted_counts"] = {"|".join(key): int(value) for key, value in sorted(counts.items())}
        manifest["train_count"] = int(len(subset_rows["train"]))
        manifest["test_count"] = int(len(subset_rows["test"]))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for subset in ("train", "test"):
            write_dataset_metadata(output_root / f"{subset}_lerobot", subset_metadata[subset], subset_rows[subset])

    candidates = build_candidates(args)
    if not candidates:
        raise RuntimeError("No push-box candidates were generated.")

    trial_count = 0
    try:
        while len(manifest["episodes"]) < target_total:
            if trial_count >= int(args.max_trials):
                raise RuntimeError(f"Reached --max-trials={args.max_trials} before filling all buckets")
            candidate = candidates[trial_count % len(candidates)]
            if trial_count > 0 and trial_count % len(candidates) == 0:
                rng_shuffle = np.random.default_rng(int(args.seed) + trial_count // len(candidates) + 17)
                rng_shuffle.shuffle(candidates)
            trial_count += 1
            if not candidate_base_has_open_quota(candidate, counts, target_quotas):
                continue

            base_id = make_case_id(candidate)
            pass_id = trial_count // len(candidates)
            case_id = f"{base_id}_r{pass_id:02d}"
            init_xy = tuple(float(v) for v in candidate["init_xy"])
            direction = direction_xy(float(candidate["angle_deg"]))
            dummy_target_xy = (
                float(init_xy[0] + direction[0] * float(args.dummy_target_distance)),
                float(init_xy[1] + direction[1] * float(args.dummy_target_distance)),
            )
            probe_bddl = write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / "probe",
                geometry_id=f"{case_id}_probe_hidden",
                init_xy=init_xy,
                target_xy=dummy_target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                yaw_deg=float(candidate["yaw_deg"]),
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            probe = build_probe_case(
                case_id=f"probe_{case_id}",
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
                pusher_push_mode=str(candidate["push_mode"]),
                pusher_push_action_end=float(candidate["action_end"]),
                pusher_push_controller_scale=float(candidate["push_scale"]),
                pusher_max_push_controller_scale=max(20.0, float(candidate["push_scale"])),
                pusher_push_action_delta=10.0,
                pusher_max_pos_action=1.0,
                pusher_approach_offset_xy=tuple(float(v) for v in candidate["approach_offset_xy"]),
                pusher_contact_offset_xy=tuple(float(v) for v in candidate["contact_offset_xy"]),
            )
            result = rollout(probe, repo_root=repo_root, seed=int(args.seed) + trial_count)
            accepted, metrics = accept_rollout(
                init_xy=init_xy,
                angle_deg=float(candidate["angle_deg"]),
                final_xy=result["final_xy"],
                final_speed=float(result["final_speed"]),
                args=args,
            )
            displacement_bin = value_bin(
                float(metrics["displacement_m"]),
                [float(v) for v in args.displacement_bin_edges],
                "disp",
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
                    "scale_bin": str(candidate["scale_bin"]),
                    "displacement_bin": displacement_bin,
                    "displacement_bucket_label": DISPLACEMENT_BUCKET_LABELS.get(displacement_bin, displacement_bin),
                }
            )
            subset, quota_key = choose_bucket_subset(candidate, displacement_bin, counts, target_quotas, rng)
            rejected_reason = ""
            if not accepted:
                rejected_reason = "acceptance"
            elif int(result["push_backward_action_count"]) > 0:
                rejected_reason = "backward_action"
            elif int(result["push_eef_backward_steps"]) > 0:
                rejected_reason = "backward_eef"
            elif float(result["max_eef_step_m"]) > float(args.max_eef_step):
                rejected_reason = "eef_step"
            elif subset is None or quota_key is None:
                rejected_reason = "bucket_full"

            if rejected_reason:
                manifest["rejected"].append(
                    {
                        "case_id": case_id,
                        "reason": rejected_reason,
                        "candidate": candidate,
                        "metrics": metrics,
                    }
                )
                if int(args.progress_every) > 0 and len(manifest["rejected"]) % int(args.progress_every) == 0:
                    print(
                        f"rejected={len(manifest['rejected'])} accepted={len(manifest['episodes'])}/"
                        f"{target_total} latest={case_id} reason={rejected_reason} "
                        f"disp={metrics['displacement_m'] * 100:.1f}cm",
                        flush=True,
                    )
                continue

            target_xy = (float(result["final_xy"][0]), float(result["final_xy"][1]))
            task_bddl = write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / str(subset),
                geometry_id=case_id,
                init_xy=init_xy,
                target_xy=target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                yaw_deg=float(candidate["yaw_deg"]),
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            task_case = replace(
                probe,
                case_id=f"{subset}_{case_id}",
                domain="observation",
                geometry_id=case_id,
                bddl_file=task_bddl,
                target_xy=target_xy,
                target_distance=float(metrics["displacement_m"]),
                camera_resolution=int(args.camera_resolution),
            )
            rollout_result = collect_case_frames(task_case, repo_root=repo_root, seed=int(args.seed) + trial_count)
            if not rollout_result["success"]:
                manifest["rejected"].append(
                    {
                        "case_id": case_id,
                        "reason": "task_rollout_failed",
                        "candidate": candidate,
                        "metrics": metrics,
                    }
                )
                continue

            episode_index = write_frames_to_dataset(
                datasets[str(subset)],
                rollout_result=rollout_result,
                task=prompt_for_case("observation", str(candidate["split"])),
                fps=int(args.fps),
                jpeg_quality=int(args.jpeg_quality),
            )
            counts[quota_key] = counts.get(quota_key, 0) + 1
            boundaries = phase_boundaries(candidate)
            row = {
                "episode_index": int(episode_index),
                "subset": str(subset),
                "case_id": task_case.case_id,
                "pair_id": case_id,
                "base_id": case_id,
                "split": str(candidate["split"]),
                "angle_sign": str(candidate["angle_sign"]),
                "friction_mu": float(candidate["friction_mu"]),
                "quota_bucket": "|".join(quota_key),
                "init_xy": list(init_xy),
                "target_xy": list(target_xy),
                "angle_deg": float(candidate["angle_deg"]),
                "yaw_deg": float(candidate["yaw_deg"]),
                "contact_offset_xy": list(candidate["contact_offset_xy"]),
                "approach_offset_xy": list(candidate["approach_offset_xy"]),
                "robot_start_pose": {
                    "approach_offset_xy": list(candidate["approach_offset_xy"]),
                    "contact_offset_xy": list(candidate["contact_offset_xy"]),
                    "approach_z": float(task_case.pusher_approach_z),
                    "contact_z": float(task_case.pusher_contact_z),
                    "retreat_z": float(task_case.pusher_retreat_z),
                },
                "push_distance_bucket": metrics["displacement_bucket_label"],
                "push_distance_x": float(candidate["push_distance"]),
                "commanded_push_distance": float(candidate["push_distance"]),
                "pusher_push_steps": int(candidate["push_steps"]),
                "push_steps": int(candidate["push_steps"]),
                "push_speed_m_per_step": float(candidate["speed_m_per_step"]),
                "push_impulse_scale": float(candidate["push_scale"]),
                "pusher_push_mode": str(candidate["push_mode"]),
                "pusher_push_action_end": float(candidate["action_end"]),
                "calibration_source": str(candidate["calibration_source"]),
                "target_hint_m": float(candidate["target_hint_m"]),
                "calibrated_displacement_m": float(candidate["calibrated_displacement_m"]),
                "calibrated_friction_mu": float(candidate["calibrated_friction_mu"]),
                "metrics": metrics,
                "phase_counts": rollout_result["phase_counts"],
                "steps": int(rollout_result["steps"]),
                "push_start": int(boundaries["push_start"]),
                "push_end": int(boundaries["push_end"]),
                "settle_start": int(boundaries["settle_start"]),
                "bddl_file": task_case.bddl_file,
            }
            subset_rows[str(subset)].append(row)
            subset_metadata[str(subset)]["episodes"].append(row)
            manifest["episodes"].append(row)
            capture_contact_sample(contact_samples, row=row, rollout_result=rollout_result)
            print(
                f"accepted {subset} {len(manifest['episodes']):03d}/{target_total:03d} "
                f"{case_id} bucket={'|'.join(quota_key)} disp={metrics['displacement_m'] * 100:.1f}cm",
                flush=True,
            )
            if int(args.autosave_every) > 0 and len(manifest["episodes"]) % int(args.autosave_every) == 0:
                autosave()
    finally:
        autosave()

    missing = {
        key: int(target_quotas[key]) - int(counts.get(key, 0))
        for key in target_quotas
        if int(counts.get(key, 0)) < int(target_quotas[key])
    }
    summary = summarize(
        output_root=output_root,
        manifest=manifest,
        subset_rows=subset_rows,
        target_quotas=target_quotas,
        counts=counts,
    )
    write_contact_sheet(
        output_root / "contact_sheet.jpg",
        contact_samples=contact_samples,
        frictions=[float(mu) for mu in args.frictions],
    )
    autosave()
    if missing:
        raise RuntimeError(f"Missing buckets: {missing}; manifest={manifest_path}")

    print(f"manifest={manifest_path}")
    print(f"summary={output_root / 'summary.md'}")
    print(f"contact_sheet={output_root / 'contact_sheet.jpg'}")
    print(f"train_root={output_root / 'train_lerobot'}")
    print(f"test_root={output_root / 'test_lerobot'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

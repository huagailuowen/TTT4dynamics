#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
import json
from pathlib import Path
import shutil
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fixed-friction LIBERO push-box train/test LeRobot datasets.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FASTWAM_ROOT / "data" / "libero_push_box_fixed_friction_200",
    )
    parser.add_argument(
        "--bddl-dir",
        type=Path,
        default=REPO_ROOT / "generated_bddl" / "libero_push_box_fixed_friction_200",
    )
    parser.add_argument("--friction", type=float, default=0.05)
    parser.add_argument("--train-count", type=int, default=150)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--camera-resolution", type=int, default=224)
    parser.add_argument("--probe-resolution", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--video-codec", default="h264", choices=["h264", "hevc", "libsvtav1", "h264_nvenc"])
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--jpeg-quality", type=int, default=98)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--autosave-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--max-trials", type=int, default=4000)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--init-half-size", type=float, default=0.002)
    parser.add_argument("--dummy-target-distance", type=float, default=0.42)
    parser.add_argument("--displacement-bin-edges", type=float, nargs="+", default=[0.18, 0.28])
    parser.add_argument("--min-displacement", type=float, default=0.08)
    parser.add_argument("--max-displacement", type=float, default=0.45)
    parser.add_argument("--max-final-speed", type=float, default=0.025)
    parser.add_argument("--max-eef-step", type=float, default=0.06)
    parser.add_argument("--x-bounds", type=float, nargs=2, default=(-0.21, 0.16))
    parser.add_argument("--y-bounds", type=float, nargs=2, default=(-0.24, 0.20))
    return parser.parse_args()


def train_test_bucket_quotas(train_count: int, test_count: int) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    if int(train_count) != 150 or int(test_count) != 50:
        raise ValueError("This balanced collector currently expects --train-count 150 and --test-count 50.")
    train = {}
    test = {}
    # Per split: short/mid/long train = 24/24/27, test = 8/8/9.
    # This keeps straight/angled exactly half in each subset and keeps each bucket at a 3:1 train:test ratio.
    for split in ("straight", "angled"):
        train[(split, "disp_00")] = 24
        train[(split, "disp_01")] = 24
        train[(split, "disp_02")] = 27
        test[(split, "disp_00")] = 8
        test[(split, "disp_01")] = 8
        test[(split, "disp_02")] = 9
    return train, test


def init_xys() -> list[tuple[str, tuple[float, float]]]:
    return [
        ("i00", (-0.255, -0.055)),
        ("i01", (-0.245, -0.035)),
        ("i02", (-0.235, -0.015)),
        ("i03", (-0.225, 0.005)),
        ("i04", (-0.255, 0.020)),
        ("i05", (-0.245, 0.040)),
        ("i06", (-0.235, -0.075)),
        ("i07", (-0.225, -0.055)),
    ]


def push_presets() -> list[dict[str, Any]]:
    return [
        {"preset": "short_a", "push_distance": 0.16, "push_steps": 24, "push_scale": 32.0},
        {"preset": "short_b", "push_distance": 0.18, "push_steps": 12, "push_scale": 20.0},
        {"preset": "short_c", "push_distance": 0.22, "push_steps": 36, "push_scale": 64.0},
        {"preset": "mid_a", "push_distance": 0.24, "push_steps": 8, "push_scale": 64.0},
        {"preset": "mid_b", "push_distance": 0.28, "push_steps": 6, "push_scale": 64.0},
        {"preset": "mid_c", "push_distance": 0.30, "push_steps": 8, "push_scale": 80.0},
        {"preset": "long_a", "push_distance": 0.30, "push_steps": 6, "push_scale": 80.0},
        {"preset": "long_b", "push_distance": 0.30, "push_steps": 4, "push_scale": 80.0},
        {"preset": "long_c", "push_distance": 0.34, "push_steps": 5, "push_scale": 80.0},
        {"preset": "long_d", "push_distance": 0.36, "push_steps": 4, "push_scale": 80.0},
    ]


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    directions = [
        ("straight", "s00", 0.0),
        ("straight", "s01", 0.0),
        ("straight", "s02", 0.0),
        ("straight", "s03", 0.0),
        ("angled", "m45", -45.0),
        ("angled", "m30", -30.0),
        ("angled", "p30", 30.0),
        ("angled", "p45", 45.0),
    ]
    candidates = []
    for init_id, init_xy in init_xys():
        for split, direction_id, angle_deg in directions:
            for preset in push_presets():
                speed = float(preset["push_distance"]) / float(preset["push_steps"])
                candidates.append(
                    {
                        "init_id": init_id,
                        "init_xy": init_xy,
                        "friction_mu": float(args.friction),
                        "split": split,
                        "direction_id": direction_id,
                        "angle_deg": float(angle_deg),
                        "push_distance": float(preset["push_distance"]),
                        "push_steps": int(preset["push_steps"]),
                        "push_scale": float(preset["push_scale"]),
                        "push_mode": "position",
                        "action_end": 1.0,
                        "preset": str(preset["preset"]),
                        "speed_m_per_step": speed,
                        "speed_bin": speed_bin(speed, [0.006, 0.012, 0.020, 0.040]),
                        "distance_bin": value_bin(float(preset["push_distance"]), [0.18, 0.26, 0.32], "dist"),
                        "scale_bin": value_bin(float(preset["push_scale"]), [32.0, 64.0], "scale"),
                    }
                )
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(candidates)
    return candidates


def make_case_id(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['init_id']}_{candidate['split']}_{mu_tag(float(candidate['friction_mu']))}_"
        f"{candidate['preset']}_{candidate['direction_id']}_{angle_tag(float(candidate['angle_deg']))}_"
        f"d{int(round(float(candidate['push_distance']) * 100)):02d}_"
        f"n{int(candidate['push_steps']):02d}_s{float(candidate['push_scale']):g}"
    )


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


def assign_subset(
    bucket: tuple[str, str],
    train_counts: dict[tuple[str, str], int],
    test_counts: dict[tuple[str, str], int],
    train_quota: dict[tuple[str, str], int],
    test_quota: dict[tuple[str, str], int],
) -> str | None:
    if train_counts.get(bucket, 0) < train_quota.get(bucket, 0):
        return "train"
    if test_counts.get(bucket, 0) < test_quota.get(bucket, 0):
        return "test"
    return None


def main() -> None:
    args = parse_args()
    patch_lerobot_video_crf(int(args.video_crf))
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    bddl_dir = args.bddl_dir if args.bddl_dir.is_absolute() else repo_root / args.bddl_dir
    output_root.mkdir(parents=True, exist_ok=True)

    train_quota, test_quota = train_test_bucket_quotas(int(args.train_count), int(args.test_count))
    train_counts = {key: 0 for key in train_quota}
    test_counts = {key: 0 for key in test_quota}
    datasets = {
        "train": create_dataset(output_root / "train_lerobot", repo_id="libero_push_box_fixed_friction_200_train", args=args),
        "test": create_dataset(output_root / "test_lerobot", repo_id="libero_push_box_fixed_friction_200_test", args=args),
    }
    subset_rows: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    subset_metadata = {
        subset: {
            "created_at": dt.datetime.now().isoformat(),
            "dataset_type": "libero_push_box_fixed_friction_lerobot",
            "subset": subset,
            "friction_mu": float(args.friction),
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
        "dataset_type": "libero_push_box_fixed_friction_200",
        "output_root": str(output_root),
        "friction_mu": float(args.friction),
        "train_count_target": int(args.train_count),
        "test_count_target": int(args.test_count),
        "bucket_definition": "split x actual displacement bin; displacement edges are meters",
        "displacement_bin_edges": [float(v) for v in args.displacement_bin_edges],
        "train_bucket_quotas": {"|".join(k): v for k, v in train_quota.items()},
        "test_bucket_quotas": {"|".join(k): v for k, v in test_quota.items()},
        "train_bucket_counts": {},
        "test_bucket_counts": {},
        "episodes": [],
        "rejected": [],
        "generation_args": to_jsonable(vars(args)),
    }
    manifest_path = output_root / "manifest.json"

    def autosave() -> None:
        manifest["train_bucket_counts"] = {"|".join(k): v for k, v in train_counts.items()}
        manifest["test_bucket_counts"] = {"|".join(k): v for k, v in test_counts.items()}
        manifest["train_count"] = int(len(subset_rows["train"]))
        manifest["test_count"] = int(len(subset_rows["test"]))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for subset in ("train", "test"):
            write_dataset_metadata(output_root / f"{subset}_lerobot", subset_metadata[subset], subset_rows[subset])

    candidates = build_candidates(args)
    trial_count = 0
    try:
        while (
            len(subset_rows["train"]) < int(args.train_count)
            or len(subset_rows["test"]) < int(args.test_count)
        ):
            if trial_count >= int(args.max_trials):
                raise RuntimeError(f"Reached --max-trials={args.max_trials} before filling all buckets")
            candidate = candidates[trial_count % len(candidates)]
            # Re-shuffle order on every full pass so later passes do not repeat the same subset order.
            if trial_count > 0 and trial_count % len(candidates) == 0:
                rng = np.random.default_rng(int(args.seed) + trial_count // len(candidates))
                rng.shuffle(candidates)
            trial_count += 1

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
                geometry_id=f"{case_id}_probe_invisible",
                init_xy=init_xy,
                target_xy=dummy_target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            probe = build_probe_case(
                case_id=f"probe_{case_id}",
                friction_mu=float(args.friction),
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
                pusher_push_mode="position",
                pusher_push_action_end=1.0,
                pusher_push_controller_scale=float(candidate["push_scale"]),
                pusher_max_push_controller_scale=max(80.0, float(candidate["push_scale"])),
                pusher_push_action_delta=10.0,
                pusher_max_pos_action=1.0,
            )
            result = rollout(probe, repo_root=repo_root, seed=int(args.seed) + trial_count)
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
                    "scale_bin": str(candidate["scale_bin"]),
                    "displacement_bin": value_bin(
                        float(metrics["displacement_m"]),
                        [float(v) for v in args.displacement_bin_edges],
                        "disp",
                    ),
                }
            )
            bucket = (str(candidate["split"]), str(metrics["displacement_bin"]))
            subset = assign_subset(bucket, train_counts, test_counts, train_quota, test_quota)
            rejected_reason = ""
            if not accepted:
                rejected_reason = "acceptance"
            elif int(result["push_backward_action_count"]) > 0:
                rejected_reason = "backward_action"
            elif int(result["push_eef_backward_steps"]) > 0:
                rejected_reason = "backward_eef"
            elif float(result["max_eef_step_m"]) > float(args.max_eef_step):
                rejected_reason = "eef_step"
            elif subset is None:
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
                        f"rejected={len(manifest['rejected'])} train={len(subset_rows['train'])}/"
                        f"{args.train_count} test={len(subset_rows['test'])}/{args.test_count} "
                        f"latest={case_id} reason={rejected_reason} disp={metrics['displacement_m'] * 100:.1f}cm",
                        flush=True,
                    )
                continue

            target_xy = (float(result["final_xy"][0]), float(result["final_xy"][1]))
            task_bddl = write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / subset,
                geometry_id=case_id,
                init_xy=init_xy,
                target_xy=target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                target_rgba=(0.0, 0.8, 0.2, 0.45),
            )
            task_case = replace(
                probe,
                case_id=f"{subset}_{case_id}",
                domain="task",
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
                datasets[subset],
                rollout_result=rollout_result,
                task=prompt_for_case("task", str(candidate["split"])),
                fps=int(args.fps),
                jpeg_quality=int(args.jpeg_quality),
            )
            if subset == "train":
                train_counts[bucket] = train_counts.get(bucket, 0) + 1
            else:
                test_counts[bucket] = test_counts.get(bucket, 0) + 1

            row = {
                "episode_index": int(episode_index),
                "subset": subset,
                "case_id": task_case.case_id,
                "base_id": case_id,
                "split": str(candidate["split"]),
                "friction_mu": float(args.friction),
                "bucket": "|".join(bucket),
                "init_xy": list(init_xy),
                "target_xy": list(target_xy),
                "angle_deg": float(candidate["angle_deg"]),
                "push_distance_x": float(candidate["push_distance"]),
                "pusher_push_steps": int(candidate["push_steps"]),
                "pusher_push_mode": "position",
                "pusher_push_controller_scale": float(candidate["push_scale"]),
                "preset": str(candidate["preset"]),
                "metrics": metrics,
                "phase_counts": rollout_result["phase_counts"],
                "steps": int(rollout_result["steps"]),
                "bddl_file": task_case.bddl_file,
            }
            subset_rows[subset].append(row)
            subset_metadata[subset]["episodes"].append(row)
            manifest["episodes"].append(row)
            print(
                f"accepted {subset} {len(subset_rows[subset]):03d}/"
                f"{args.train_count if subset == 'train' else args.test_count:03d} "
                f"{case_id} bucket={'|'.join(bucket)} disp={metrics['displacement_m'] * 100:.1f}cm",
                flush=True,
            )
            if int(args.autosave_every) > 0 and len(manifest["episodes"]) % int(args.autosave_every) == 0:
                autosave()
    finally:
        autosave()

    missing_train = {k: train_quota[k] - train_counts.get(k, 0) for k in train_quota if train_counts.get(k, 0) < train_quota[k]}
    missing_test = {k: test_quota[k] - test_counts.get(k, 0) for k in test_quota if test_counts.get(k, 0) < test_quota[k]}
    if missing_train or missing_test:
        raise RuntimeError(f"Missing buckets train={missing_train} test={missing_test}; manifest={manifest_path}")

    print(f"manifest={manifest_path}")
    print(f"train_root={output_root / 'train_lerobot'}")
    print(f"test_root={output_root / 'test_lerobot'}")


if __name__ == "__main__":
    main()

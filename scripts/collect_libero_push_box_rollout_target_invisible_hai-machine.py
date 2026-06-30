#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_libero_push_box_rollout_target_lerobot_dataset as base  # noqa: E402


def create_hidden_datasets(args: Any) -> dict[tuple[str, str], Any]:
    output_prefix = args.output_prefix.resolve()
    selected_splits = set(str(split) for split in args.splits)
    roots = {
        ("observation", split): base.dataset_root(output_prefix, "observation", split)
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
        key: base.LeRobotDataset.create(
            repo_id=f"{args.repo_id_prefix}_hidden_{key[1]}",
            root=root,
            fps=int(args.fps),
            features=base.build_features(int(args.camera_resolution)),
            use_videos=True,
            video_codec=args.video_codec,
            is_compute_episode_stats_image=False,
        )
        for key, root in roots.items()
    }


def main() -> None:
    args = base.parse_args()
    base.patch_lerobot_video_crf(int(args.video_crf))
    repo_root = args.repo_root.resolve()
    bddl_dir = args.bddl_dir if args.bddl_dir.is_absolute() else repo_root / args.bddl_dir
    output_prefix = args.output_prefix.resolve()
    datasets = create_hidden_datasets(args)
    candidates = base.build_candidates(args)
    dimensions = list(args.balance_dimensions)
    displacement_edges = [float(v) for v in args.displacement_bin_edges]
    target_buckets = base.build_target_buckets(
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
    accepted_frictions = {base.mu_tag(float(mu)): 0 for mu in args.frictions}

    subset_rows: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in datasets}
    subset_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for key in datasets:
        subset_metadata[key] = {
            "created_at": dt.datetime.now().isoformat(),
            "dataset_type": "libero_push_box_rollout_target_invisible_hai-machine_lerobot",
            "domain": key[0],
            "target_visible": False,
            "split": key[1],
            "camera_resolution": int(args.camera_resolution),
            "fps": int(args.fps),
            "video_codec": str(args.video_codec),
            "video_crf": int(args.video_crf),
            "jpeg_quality": int(args.jpeg_quality),
            "seed": int(args.seed),
            "output_root": str(base.dataset_root(output_prefix, key[0], key[1])),
            "episodes": [],
        }

    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_rollout_target_invisible_hai-machine_collection",
        "target_visible": False,
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
            f"{domain}_{split}": str(base.dataset_root(output_prefix, domain, split))
            for domain, split in datasets
        },
        "generation_args": base.to_jsonable(vars(args)),
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
            base.write_dataset_metadata(base.dataset_root(output_prefix, key[0], key[1]), metadata, subset_rows[key])

    try:
        for candidate in candidates:
            candidate_possible_keys = base.possible_bucket_keys(
                candidate,
                dimensions,
                displacement_edges=displacement_edges,
            )
            if all(
                accepted_buckets.get(possible_key, 0) >= target_buckets.get(possible_key, 0)
                for possible_key in candidate_possible_keys
            ):
                continue

            base_id = base.make_case_id(candidate)
            init_xy = tuple(float(v) for v in candidate["init_xy"])
            direction = base.direction_xy(float(candidate["angle_deg"]))
            dummy_target_xy = (
                float(init_xy[0] + direction[0] * float(args.dummy_target_distance)),
                float(init_xy[1] + direction[1] * float(args.dummy_target_distance)),
            )
            probe_bddl = base.write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / "probe",
                geometry_id=f"{base_id}_probe_invisible",
                init_xy=init_xy,
                target_xy=dummy_target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            probe = base.build_probe_case(
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
            result = base.rollout(probe, repo_root=repo_root, seed=int(args.seed))
            accepted, metrics = base.accept_rollout(
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
                    "displacement_bin": base.value_bin(
                        float(metrics["displacement_m"]),
                        displacement_edges,
                        "disp",
                    ),
                    "scale_bin": str(candidate["scale_bin"]),
                }
            )
            key = base.bucket_key(candidate, dimensions, displacement_bin=str(metrics["displacement_bin"]))
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
                and accepted_frictions.get(base.mu_tag(float(candidate["friction_mu"])), 0)
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
                    and accepted_frictions.get(base.mu_tag(float(candidate["friction_mu"])), 0)
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
            invisible_bddl = base.write_geometry_bddl(
                repo_root=repo_root,
                bddl_dir=bddl_dir / "observation",
                geometry_id=base_id,
                init_xy=init_xy,
                target_xy=target_xy,
                init_half_size=float(args.init_half_size),
                target_radius=float(args.target_radius),
                target_rgba=(0.0, 0.0, 0.0, 0.0),
            )
            observation_case = replace(
                probe,
                case_id=f"observation_{base_id}",
                domain="observation",
                geometry_id=base_id,
                bddl_file=invisible_bddl,
                target_xy=target_xy,
                target_distance=float(metrics["displacement_m"]),
                camera_resolution=int(args.camera_resolution),
            )

            obs_rollout = base.collect_case_frames(observation_case, repo_root=repo_root, seed=int(args.seed))
            if not obs_rollout["success"]:
                manifest["rejected"].append(
                    {
                        "case_id": base_id,
                        "reason": "observation_rollout_failed",
                        "candidate": candidate,
                        "metrics": metrics,
                        "observation_success": bool(obs_rollout["success"]),
                    }
                )
                continue

            split = str(candidate["split"])
            episode_index = base.write_frames_to_dataset(
                datasets[("observation", split)],
                rollout_result=obs_rollout,
                task=base.prompt_for_case("observation", split),
                fps=int(args.fps),
                jpeg_quality=int(args.jpeg_quality),
            )
            pair_record = {
                "pair_id": base_id,
                "bucket": "|".join(key),
                "candidate": candidate,
                "target_visible": False,
                "target_xy": list(target_xy),
                "metrics": metrics,
                "observation_case": observation_case.as_dict(),
                "episode_indices": {"observation": int(episode_index)},
                "steps": {"observation": int(obs_rollout["steps"])},
                "phase_counts": {"observation": obs_rollout["phase_counts"]},
            }
            manifest["pairs"].append(pair_record)
            accepted_buckets[key] = accepted_buckets.get(key, 0) + 1
            accepted_frictions[base.mu_tag(float(candidate["friction_mu"]))] = (
                accepted_frictions.get(base.mu_tag(float(candidate["friction_mu"])), 0) + 1
            )

            subset_key = ("observation", split)
            row = {
                "episode_index": int(episode_index),
                "pair_id": base_id,
                "case_id": observation_case.case_id,
                "domain": "observation",
                "target_visible": False,
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
                "phase_counts": obs_rollout["phase_counts"],
                "steps": int(obs_rollout["steps"]),
                "bddl_file": observation_case.bddl_file,
            }
            subset_rows[subset_key].append(row)
            subset_metadata[subset_key]["episodes"].append(row)

            print(
                f"accepted {len(manifest['pairs']):04d}/{target_pair_count:04d} "
                f"{base_id} bucket={'|'.join(key)} disp={metrics['displacement_m'] * 100:.1f}cm "
                f"episode={episode_index}",
                flush=True,
            )
            if int(args.autosave_every) > 0 and len(manifest["pairs"]) % int(args.autosave_every) == 0:
                autosave()
            if int(args.max_pairs) > 0 and len(manifest["pairs"]) >= int(args.max_pairs):
                break
            if all(accepted_buckets.get(bucket, 0) >= target for bucket, target in target_buckets.items()):
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
        print(f"{key[0]} {key[1]} root={base.dataset_root(output_prefix, key[0], key[1])}")


if __name__ == "__main__":
    main()

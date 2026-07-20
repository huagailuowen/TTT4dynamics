#!/usr/bin/env python3
"""Collect 100 preselected mass/friction environments with all nine actions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_COLLECTOR = (
    REPO_ROOT
    / "scripts/collect_libero_two_box_mass_friction_boundary_3mass_3friction_9action_lerobot_2026-07-18_hai-machine.py"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/libero_two_box_mass_friction_balanced100env_9action_2026-07-18_hai-machine.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data/mass_friction/balanced100env_20mass_20friction_9action_2026-07-18_hai-machine"
)
DATASET_NAME = (
    "libero_two_box_mass_friction_balanced100env_9action_900eps_"
    "lerobot_2026-07-18_hai-machine"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the balanced 100-environment mass/friction dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--video-codec", choices=("h264", "hevc", "libsvtav1", "h264_nvenc"), default="h264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def selection_hash(pairs: list[dict[str, Any]]) -> str:
    payload = json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_balanced_pairs(
    masses: list[float], frictions: list[float], *, seed: int
) -> list[dict[str, Any]]:
    if len(masses) != 20 or len(frictions) != 20:
        raise RuntimeError("Formal balanced design requires exactly 20 mass and 20 friction levels")
    size = len(masses)
    permutations: list[tuple[str, list[int]]] = [
        ("identity", list(range(size))),
        ("reverse", list(reversed(range(size)))),
    ]
    used_edges = {
        (mass_index, friction_index)
        for _, permutation in permutations
        for mass_index, friction_index in enumerate(permutation)
    }
    rng = np.random.default_rng(int(seed))
    for random_round in range(3):
        for _ in range(10000):
            permutation = rng.permutation(size).astype(int).tolist()
            edges = {(mass_index, friction_index) for mass_index, friction_index in enumerate(permutation)}
            if not edges & used_edges:
                permutations.append((f"random_permutation_{random_round}", permutation))
                used_edges.update(edges)
                break
        else:
            raise RuntimeError("Failed to construct five unique balanced parameter matchings")

    pairs: list[dict[str, Any]] = []
    pair_index = 0
    for matching_index, (matching_name, permutation) in enumerate(permutations):
        for mass_index, friction_index in enumerate(permutation):
            pairs.append(
                {
                    "pair_index": pair_index,
                    "matching_index": matching_index,
                    "matching_name": matching_name,
                    "mass_index": mass_index,
                    "target_mass_kg": float(masses[mass_index]),
                    "friction_index": friction_index,
                    "target_table_friction_mu": float(frictions[friction_index]),
                }
            )
            pair_index += 1
    if len(pairs) != 100 or len(used_edges) != 100:
        raise RuntimeError("Balanced design did not produce 100 unique pairs")
    if any(sum(pair["mass_index"] == index for pair in pairs) != 5 for index in range(size)):
        raise RuntimeError("A mass level does not occur exactly five times")
    if any(sum(pair["friction_index"] == index for pair in pairs) != 5 for index in range(size)):
        raise RuntimeError("A friction level does not occur exactly five times")
    return pairs


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    boundary = None
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("mass_friction_boundary_collector_hai_machine", BOUNDARY_COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load boundary collector: {BOUNDARY_COLLECTOR}")
    boundary = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = boundary
    spec.loader.exec_module(boundary)
    collector = boundary.load_module(boundary.BASE_COLLECTOR, "mass_friction_formal_base_collector_hai_machine")
    demo = collector.load_collision_demo()
    collector.configure_demo(demo, config)

    implementation = config["friction_implementation"]
    friction_state = boundary.install_target_friction_router(
        demo,
        projectile_mu=float(config["projectile_table_friction_mu"]),
        projectile_priority=int(implementation["projectile_geom_priority"]),
        target_priority=int(implementation["target_geom_priority"]),
        table_priority=int(implementation["table_geom_priority"]),
    )
    masses = [float(value) for value in config["target_masses_kg"]]
    frictions = [float(value) for value in config["target_table_friction_values"]]
    actions_cfg = list(config["actions"])
    pairs = build_balanced_pairs(
        masses,
        frictions,
        seed=int(config["sampling_policy"]["selection_seed"]),
    )
    pair_hash = selection_hash(pairs)
    collector.write_json(output_root / "selected_environment_pairs.json", pairs)

    execution_rng = np.random.default_rng(int(config["sampling_policy"]["selection_seed"]) + 1)
    execution_order = execution_rng.permutation(len(pairs)).astype(int).tolist()
    execution_pairs = [dict(pairs[index], execution_index=position) for position, index in enumerate(execution_order)]
    collector.write_json(output_root / "environment_execution_order.json", execution_pairs)

    expected_total = len(pairs) * len(actions_cfg)
    if expected_total != 900:
        raise RuntimeError(f"Formal dataset must contain 900 episodes, got {expected_total}")
    bddl_file = demo.write_two_box_bddl(output_root)
    collector.patch_lerobot_video_crf(int(args.video_crf))
    dataset_root = output_root / DATASET_NAME
    dataset = collector.create_dataset(dataset_root, config, str(args.video_codec))

    metadata: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "episode_count_expected": expected_total,
        "environment_count": len(pairs),
        "episodes_per_environment": len(actions_cfg),
        "projectile_mass_kg": float(config["projectile_mass_kg"]),
        "projectile_table_friction_mu": float(config["projectile_table_friction_mu"]),
        "target_masses_kg": masses,
        "target_table_friction_values": frictions,
        "sampling_policy": config["sampling_policy"],
        "selection_sha256": pair_hash,
        "selected_environment_pairs": pairs,
        "friction_implementation": implementation,
        "camera_resolution": int(config["camera_resolution"]),
        "recorded_steps": int(config["recorded_steps"]),
        "fps": int(config["fps"]),
        "video_codec": str(args.video_codec),
        "video_crf": int(args.video_crf),
        "jpeg_quality": int(args.jpeg_quality),
        "action_source": "exact unchanged 7D LIBERO OSC actions from the 2026-07-17 9-action dataset",
        "episodes": [],
    }
    manifest: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "output_root": str(output_root),
        "lerobot_root": str(dataset_root),
        "config_path": str(args.config.resolve()),
        "bddl_file": str(bddl_file),
        "selection_sha256": pair_hash,
        "selected_environment_pairs_file": str(output_root / "selected_environment_pairs.json"),
        "episodes": [],
    }
    rows: list[dict[str, Any]] = []
    expected_action_hashes: dict[int, str] = {}

    def autosave() -> None:
        collector.write_json(output_root / "manifest.json", manifest)
        with (output_root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(collector.to_jsonable(row)) + "\n")
        collector.write_dataset_metadata(dataset_root, metadata, rows)

    count = 0
    for environment in execution_pairs:
        target_mass_kg = float(environment["target_mass_kg"])
        target_friction_mu = float(environment["target_table_friction_mu"])
        environment_id = f"env_{int(environment['pair_index']):03d}"
        friction_state["target_mu"] = target_friction_mu
        for action_cfg in actions_cfg:
            action_id = int(action_cfg["action_id"])
            episode_seed = int(args.seed) + count
            records, diagnostics, fixed_actions = collector.rollout_episode(
                demo,
                bddl_file=str(bddl_file),
                target_mass_kg=target_mass_kg,
                action_cfg=action_cfg,
                recorded_steps=int(config["recorded_steps"]),
                seed=episode_seed,
            )
            diagnostics.update(
                boundary.enrich_physics_metrics(
                    records,
                    diagnostics,
                    target_mass_kg=target_mass_kg,
                    projectile_mass_kg=float(config["projectile_mass_kg"]),
                    target_friction_mu=target_friction_mu,
                    fps=int(config["fps"]),
                )
            )
            diagnostics["applied_contact_configuration"] = dict(friction_state["applied"])
            diagnostics["sampling_decision_made_before_any_rollout"] = True
            diagnostics["outcome_used_for_selection_or_replacement"] = False

            current_hash = collector.hash_actions(fixed_actions)
            expected_hash = expected_action_hashes.setdefault(action_id, current_hash)
            if current_hash != expected_hash:
                raise RuntimeError(f"Action changed across environments for action_id={action_id}")
            episode_index = collector.save_lerobot_episode(
                dataset,
                records,
                fps=int(config["fps"]),
                jpeg_quality=int(args.jpeg_quality),
            )
            case_id = f"{environment_id}_a{action_id:02d}"
            diagnostics_file = output_root / "diagnostics" / f"episode_{episode_index:06d}_{case_id}.json"
            collector.write_json(diagnostics_file, collector.diagnostics_rows(records))
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "environment_id": environment_id,
                "pair_index": int(environment["pair_index"]),
                "execution_index": int(environment["execution_index"]),
                "matching_index": int(environment["matching_index"]),
                "matching_name": str(environment["matching_name"]),
                "selection_sha256": pair_hash,
                "episode_seed": episode_seed,
                "mass_index": int(environment["mass_index"]),
                "target_mass_kg": target_mass_kg,
                "target_mass_g": target_mass_kg * 1000.0,
                "friction_index": int(environment["friction_index"]),
                "target_table_friction_mu": target_friction_mu,
                "projectile_table_friction_mu": float(config["projectile_table_friction_mu"]),
                "action_id": action_id,
                "A": float(action_cfg["A"]),
                "push_steps": int(action_cfg["push_steps"]),
                "launch_profile": collector.profile_for_action(action_cfg).astype(float).tolist(),
                "action_sha256": current_hash,
                "sampling_policy": config["sampling_policy"],
                "diagnostics_file": str(diagnostics_file),
                "metrics": diagnostics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            manifest["episodes"].append(row)
            count += 1
            print(
                f"collect {count:03d}/{expected_total:03d} {case_id} "
                f"pair={environment['pair_index']} mass={target_mass_kg:.6f}kg "
                f"mu={target_friction_mu:.4f} action={action_id} "
                f"v_post={diagnostics['target_postcollision_vx_mps']:.4f}m/s",
                flush=True,
            )
            autosave()

    summary = {
        "episode_count": len(rows),
        "expected_episode_count": expected_total,
        "environment_count": len(pairs),
        "selection_sha256": pair_hash,
        "count_by_mass_index": dict(Counter(int(row["mass_index"]) for row in rows)),
        "count_by_friction_index": dict(Counter(int(row["friction_index"]) for row in rows)),
        "count_by_action": dict(Counter(int(row["action_id"]) for row in rows)),
        "near_zero_motion_count": sum(int(row["metrics"]["near_zero_motion"]) for row in rows),
        "off_table_at_frame_60_count": sum(
            int(not row["metrics"]["on_table_at_frame_60"]) for row in rows
        ),
        "outcome_conditioned_selection": False,
        "replacement_or_resampling_count": 0,
        "action_sha256_by_action": {
            str(action_id): value for action_id, value in sorted(expected_action_hashes.items())
        },
        "lerobot_root": str(dataset_root),
    }
    collector.write_json(output_root / "summary.json", summary)
    collector.write_json(output_root / "config_used.json", config)
    autosave()
    print(json.dumps(collector.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

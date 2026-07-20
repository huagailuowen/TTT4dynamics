#!/usr/bin/env python3
"""Collect the fixed 3-mass x 3-friction x 9-action boundary dataset.

Environment pairs are fixed before rollout. Outcomes are never used to reject,
replace, or resample a pair. Offscreen and near-zero-motion outcomes are valid.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_COLLECTOR = (
    REPO_ROOT
    / "scripts/collect_libero_two_box_collision_9speed_20mass_linear_theory_distance_lerobot_2026-07-17_hai-machine.py"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/libero_two_box_mass_friction_boundary_3mass_3friction_9action_2026-07-18_hai-machine.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data/mass_friction/boundary_3mass_3friction_9action_2026-07-18_hai-machine"
)
DATASET_NAME = (
    "libero_two_box_mass_friction_boundary_3mass_3friction_9action_81eps_"
    "lerobot_2026-07-18_hai-machine"
)
GRAVITY_MPS2 = 9.81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the mass/friction boundary LeRobot dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--video-codec", choices=("h264", "hevc", "libsvtav1", "h264_nvenc"), default="h264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contact_segments(records: list[dict[str, Any]]) -> list[tuple[int, int]]:
    positions = [index for index, record in enumerate(records) if record["block_block_contact"]]
    if not positions:
        return []
    segments: list[list[int]] = [[positions[0], positions[0]]]
    for position in positions[1:]:
        if position - segments[-1][1] <= 2:
            segments[-1][1] = position
        else:
            segments.append([position, position])
    return [(start, end) for start, end in segments]


def fit_slope(times: list[float], values: list[float]) -> float | None:
    if len(times) < 3:
        return None
    time_mean = float(np.mean(times))
    value_mean = float(np.mean(values))
    denominator = sum((time - time_mean) ** 2 for time in times)
    if denominator <= 0.0:
        return None
    return sum(
        (time - time_mean) * (value - value_mean)
        for time, value in zip(times, values)
    ) / denominator


def enrich_physics_metrics(
    records: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    target_mass_kg: float,
    projectile_mass_kg: float,
    target_friction_mu: float,
    fps: int,
) -> dict[str, Any]:
    segments = contact_segments(records)
    if segments:
        first_start, first_end = segments[0]
        post_position = min(first_end + 1, len(records) - 1)
        speed_source = "first_free_frame_after_first_contact_cluster"
    else:
        first_event_frame = int(diagnostics["first_block_collision_frame"])
        post_position = next(
            (
                index
                for index, record in enumerate(records)
                if int(record["frame_index"]) >= first_event_frame
            ),
            len(records) - 1,
        )
        speed_source = "first_transfer_frame_no_sampled_contact"

    post_record = records[post_position]
    post_vx = float(post_record["target_vel"][0])
    post_vy = float(post_record["target_vel"][1])
    post_speed_xy = math.hypot(post_vx, post_vy)
    free_times: list[float] = []
    free_vx: list[float] = []
    for position in range(post_position, min(len(records), post_position + 12)):
        record = records[position]
        vx = float(record["target_vel"][0])
        target_z = float(record["target_xyz"][2])
        if record["block_block_contact"] or target_z < 0.85 or vx <= 0.005:
            break
        free_times.append(float(record["frame_index"]) / float(fps))
        free_vx.append(vx)
    slope = fit_slope(free_times, free_vx)
    measured_deceleration = -slope if slope is not None else None
    estimated_effective_mu = (
        max(0.0, measured_deceleration / GRAVITY_MPS2)
        if measured_deceleration is not None
        else None
    )
    nominal_stop_distance_m = (
        max(post_vx, 0.0) ** 2 / (2.0 * target_friction_mu * GRAVITY_MPS2)
    )
    nominal_stop_time_s = max(post_vx, 0.0) / (target_friction_mu * GRAVITY_MPS2)
    preimpact_vx = float(diagnostics["preimpact_projectile_vx_mps"])
    elastic_upper_vx = (
        2.0 * projectile_mass_kg / (projectile_mass_kg + target_mass_kg) * preimpact_vx
    )
    collision_upper_ratio = post_vx / elastic_upper_vx if elastic_upper_vx > 0.0 else None
    final_target_z = float(diagnostics["final_target_xyz"][2])
    return {
        "target_postcollision_vx_mps": post_vx,
        "target_postcollision_vy_mps": post_vy,
        "target_postcollision_speed_xy_mps": post_speed_xy,
        "postcollision_speed_source": speed_source,
        "postcollision_sample_frame": int(post_record["frame_index"]),
        "block_contact_segment_count": len(segments),
        "free_deceleration_fit_frame_count": len(free_times),
        "measured_free_deceleration_mps2": measured_deceleration,
        "estimated_effective_friction_mu": estimated_effective_mu,
        "configured_target_table_friction_mu": target_friction_mu,
        "nominal_stop_distance_cm": nominal_stop_distance_m * 100.0,
        "nominal_stop_time_s": nominal_stop_time_s,
        "ideal_elastic_target_vx_upper_mps": elastic_upper_vx,
        "post_vx_to_elastic_upper_ratio": collision_upper_ratio,
        "near_zero_motion": post_speed_xy < 0.005,
        "on_table_at_frame_60": final_target_z >= 0.85,
        "outcome_retained_unconditionally": True,
    }


def install_target_friction_router(
    demo: Any,
    *,
    projectile_mu: float,
    projectile_priority: int,
    target_priority: int,
    table_priority: int,
) -> dict[str, Any]:
    original = demo.set_object_contact_properties
    state: dict[str, Any] = {"target_mu": None, "applied": {}}

    def routed(env: Any, name: str, *, rgba: tuple[float, float, float, float]) -> None:
        desired_mu = float(state["target_mu"] if name == demo.TARGET_NAME else projectile_mu)
        previous_mu = float(demo.FRICTION_MU)
        demo.FRICTION_MU = desired_mu
        try:
            original(env, name, rgba=rgba)
        finally:
            demo.FRICTION_MU = previous_mu

        model = env.inner_env.sim.model
        if not hasattr(model, "geom_priority"):
            raise RuntimeError("MuJoCo model does not expose geom_priority")
        priority = target_priority if name == demo.TARGET_NAME else projectile_priority
        geom_ids = sorted(int(value) for value in demo.object_geom_ids(env, name))
        for geom_id in geom_ids:
            model.geom_priority[geom_id] = int(priority)
            model.geom_friction[geom_id, 0] = desired_mu

        table_rows = []
        for table_name in ("table_collision", "main_table_collision"):
            try:
                table_id = int(model.geom_name2id(table_name))
            except Exception:
                continue
            if table_id < 0:
                continue
            model.geom_priority[table_id] = int(table_priority)
            model.geom_friction[table_id, 0] = float(projectile_mu)
            table_rows.append(
                {
                    "name": table_name,
                    "geom_id": table_id,
                    "priority": int(model.geom_priority[table_id]),
                    "sliding_friction": float(model.geom_friction[table_id, 0]),
                }
            )
        state["applied"][name] = {
            "geom_ids": geom_ids,
            "priority": priority,
            "sliding_friction": desired_mu,
            "table_geoms": table_rows,
        }
        env.inner_env.sim.forward()

    demo.set_object_contact_properties = routed
    return state


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    collector = load_module(BASE_COLLECTOR, "mass_friction_base_collector_hai_machine")
    demo = collector.load_collision_demo()
    collector.configure_demo(demo, config)
    implementation = config["friction_implementation"]
    friction_state = install_target_friction_router(
        demo,
        projectile_mu=float(config["projectile_table_friction_mu"]),
        projectile_priority=int(implementation["projectile_geom_priority"]),
        target_priority=int(implementation["target_geom_priority"]),
        table_priority=int(implementation["table_geom_priority"]),
    )

    actions_cfg = list(config["actions"])
    masses = [float(value) for value in config["target_masses_kg"]]
    frictions = [float(value) for value in config["target_table_friction_values"]]
    expected_total = len(actions_cfg) * len(masses) * len(frictions)
    if expected_total != 81:
        raise RuntimeError(f"Boundary grid must contain 81 episodes, got {expected_total}")

    bddl_file = demo.write_two_box_bddl(output_root)
    collector.patch_lerobot_video_crf(int(args.video_crf))
    dataset_root = output_root / DATASET_NAME
    dataset = collector.create_dataset(dataset_root, config, str(args.video_codec))
    metadata: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "episode_count_expected": expected_total,
        "grid": {
            "action_count": len(actions_cfg),
            "target_mass_count": len(masses),
            "target_friction_count": len(frictions),
            "environment_count": len(masses) * len(frictions),
        },
        "projectile_mass_kg": float(config["projectile_mass_kg"]),
        "projectile_table_friction_mu": float(config["projectile_table_friction_mu"]),
        "target_masses_kg": masses,
        "target_table_friction_values": frictions,
        "sampling_policy": config["sampling_policy"],
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
    environment_index = 0
    for mass_index, target_mass_kg in enumerate(masses):
        for friction_index, target_friction_mu in enumerate(frictions):
            environment_id = f"env_m{mass_index:02d}_mu{int(round(target_friction_mu * 10000)):04d}"
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
                    enrich_physics_metrics(
                        records,
                        diagnostics,
                        target_mass_kg=target_mass_kg,
                        projectile_mass_kg=float(config["projectile_mass_kg"]),
                        target_friction_mu=target_friction_mu,
                        fps=int(config["fps"]),
                    )
                )
                diagnostics["applied_contact_configuration"] = dict(friction_state["applied"])
                diagnostics["sampling_decision_made_before_rollout"] = True
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
                    "environment_index": environment_index,
                    "episode_seed": episode_seed,
                    "mass_index": mass_index,
                    "target_mass_kg": target_mass_kg,
                    "target_mass_g": target_mass_kg * 1000.0,
                    "friction_index": friction_index,
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
                    f"mass={target_mass_kg:.4f}kg mu={target_friction_mu:.4f} "
                    f"action={action_id} v_post={diagnostics['target_postcollision_vx_mps']:.4f}m/s "
                    f"mu_eff={diagnostics['estimated_effective_friction_mu']}",
                    flush=True,
                )
                autosave()
            environment_index += 1

    speed_by_environment: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        speed_by_environment[str(row["environment_id"])].append(
            float(row["metrics"]["target_postcollision_vx_mps"])
        )
    summary = {
        "episode_count": len(rows),
        "expected_episode_count": expected_total,
        "environment_count": len(speed_by_environment),
        "lerobot_root": str(dataset_root),
        "count_by_mass_kg": dict(Counter(float(row["target_mass_kg"]) for row in rows)),
        "count_by_target_friction_mu": dict(
            Counter(float(row["target_table_friction_mu"]) for row in rows)
        ),
        "count_by_action": dict(Counter(int(row["action_id"]) for row in rows)),
        "near_zero_motion_count": sum(
            int(row["metrics"]["near_zero_motion"]) for row in rows
        ),
        "off_table_at_frame_60_count": sum(
            int(not row["metrics"]["on_table_at_frame_60"]) for row in rows
        ),
        "outcome_conditioned_selection": False,
        "replacement_or_resampling_count": 0,
        "action_sha256_by_action": {
            str(action_id): value for action_id, value in sorted(expected_action_hashes.items())
        },
    }
    collector.write_json(output_root / "summary.json", summary)
    collector.write_json(output_root / "config_used.json", config)
    autosave()
    print(json.dumps(collector.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit and collect the 30-mass linear-theory dataset without visual mass leakage."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_COLLECTOR = (
    REPO_ROOT
    / "scripts/collect_libero_two_box_collision_9speed_20mass_linear_theory_distance_lerobot_2026-07-17_hai-machine.py"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/libero_two_box_mass_linear_theory_30mass_noleak_2026-08-27_hai-machine.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data/mass/linear_theory_distance_9speed_30mass_noleak_2026-08-27_hai-machine"
)
DATASET_NAME = (
    "libero_two_box_collision_9speed_30mass_linear_theory_distance_noleak_270eps_"
    "lerobot_2026-08-27_hai-machine"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit or collect the no-leak 30-mass dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-codec", choices=("h264", "hevc", "libsvtav1", "h264_nvenc"), default="h264")
    parser.add_argument("--video-crf", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for ordered_index in order[index:end]:
            ranks[ordered_index] = rank
        index = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def build_mass_levels(config: dict[str, Any]) -> list[dict[str, Any]]:
    original = [float(value) for value in config["original_target_masses_kg"]]
    projectile_mass = float(config["projectile_mass_kg"])
    sampling = config["additional_mass_sampling"]
    count = int(sampling["count"])
    rng = np.random.default_rng(int(sampling["seed"]))
    minimum_mass = min(original)
    maximum_mass = max(original)
    q_min = 1.0 / (maximum_mass + projectile_mass) ** 2
    q_max = 1.0 / (minimum_mass + projectile_mass) ** 2
    additional: list[float] = []
    for stratum in range(count):
        low = q_min + (q_max - q_min) * stratum / count
        high = q_min + (q_max - q_min) * (stratum + 1) / count
        q_value = float(rng.uniform(low, high))
        additional.append(1.0 / math.sqrt(q_value) - projectile_mass)

    rows = [
        {"target_mass_kg": mass, "source": "original_20", "random_stratum": None}
        for mass in original
    ]
    rows.extend(
        {
            "target_mass_kg": mass,
            "source": "additional_stratified_random_10",
            "random_stratum": index,
        }
        for index, mass in enumerate(additional)
    )
    rows.sort(key=lambda row: float(row["target_mass_kg"]), reverse=True)
    for index, row in enumerate(rows):
        row["mass_index_short_to_long"] = index
        row["target_mass_g"] = float(row["target_mass_kg"]) * 1000.0
        row["theory_coordinate_q"] = 1.0 / (
            float(row["target_mass_kg"]) + projectile_mass
        ) ** 2
    if len(rows) != 30 or len({row["target_mass_kg"] for row in rows}) != 30:
        raise RuntimeError("Mass selection did not produce 30 unique values")
    return rows


def install_mass_independent_pose_and_contacts(
    collector: Any, demo: Any, config: dict[str, Any]
) -> dict[str, float]:
    pose_state = {"object_center_z_m": float(config["canonical_object_center_z_m"])}
    canonical_quat = np.asarray(config["canonical_object_quat_wxyz"], dtype=np.float64)
    original_contact_setter = demo.set_object_contact_properties
    contact_config = config["contact_configuration"]

    def place_objects_at_mass_independent_pose(demo_arg: Any, env: Any) -> dict[str, float]:
        projectile = demo_arg.object_instance(env, demo_arg.PROJECTILE_NAME)
        target = demo_arg.object_instance(env, demo_arg.TARGET_NAME)
        projectile_qpos = np.asarray(
            env.inner_env.sim.data.get_joint_qpos(projectile.joints[-1]), dtype=np.float64
        ).copy()
        target_qpos = np.asarray(
            env.inner_env.sim.data.get_joint_qpos(target.joints[-1]), dtype=np.float64
        ).copy()
        projectile_qpos[:2] = np.asarray(demo_arg.PROJECTILE_INIT_XY, dtype=np.float64)
        target_qpos[:2] = np.asarray(demo_arg.TARGET_INIT_XY, dtype=np.float64)
        projectile_qpos[2] = pose_state["object_center_z_m"]
        target_qpos[2] = pose_state["object_center_z_m"]
        projectile_qpos[3:7] = canonical_quat
        target_qpos[3:7] = canonical_quat
        env.inner_env.sim.data.set_joint_qpos(projectile.joints[-1], projectile_qpos)
        env.inner_env.sim.data.set_joint_qpos(target.joints[-1], target_qpos)
        env.inner_env.sim.data.set_joint_qvel(projectile.joints[-1], np.zeros(6, dtype=np.float64))
        env.inner_env.sim.data.set_joint_qvel(target.joints[-1], np.zeros(6, dtype=np.float64))
        env.inner_env.sim.forward()
        projectile_xyz, _, _ = demo_arg.object_state(env, demo_arg.PROJECTILE_NAME)
        target_xyz, _, _ = demo_arg.object_state(env, demo_arg.TARGET_NAME)
        return {
            "center_gap_m": float(target_xyz[0] - projectile_xyz[0]),
            "lateral_offset_m": float(target_xyz[1] - projectile_xyz[1]),
            "projectile_z_m": float(projectile_xyz[2]),
            "target_z_m": float(target_xyz[2]),
        }

    def set_fixed_contact_properties(
        env: Any, name: str, *, rgba: tuple[float, float, float, float]
    ) -> None:
        original_contact_setter(env, name, rgba=rgba)
        model = env.inner_env.sim.model
        friction = float(contact_config["sliding_friction_mu"])
        priority = int(contact_config["object_geom_priority"])
        for geom_id in demo.object_geom_ids(env, name):
            model.geom_priority[int(geom_id)] = priority
            model.geom_friction[int(geom_id), 0] = friction
        for table_name in ("table_collision", "main_table_collision"):
            try:
                table_id = int(model.geom_name2id(table_name))
            except Exception:
                continue
            if table_id >= 0:
                model.geom_priority[table_id] = int(contact_config["table_geom_priority"])
                model.geom_friction[table_id, 0] = friction
        env.inner_env.sim.forward()

    collector.place_objects_at_exact_initial_poses = place_objects_at_mass_independent_pose
    demo.set_object_contact_properties = set_fixed_contact_properties
    return pose_state


def prepare_runtime(config: dict[str, Any]) -> tuple[Any, Any, dict[str, float]]:
    collector = load_module(BASE_COLLECTOR, "mass_linear_30_noleak_base_collector")
    demo = collector.load_collision_demo()
    compatibility_config = dict(config)
    compatibility_config["target_masses_kg"] = list(config["original_target_masses_kg"])
    collector.configure_demo(demo, compatibility_config)
    pose_state = install_mass_independent_pose_and_contacts(collector, demo, config)
    return collector, demo, pose_state


def capture_initial_visual(
    collector: Any,
    demo: Any,
    *,
    bddl_file: str,
    target_mass_kg: float,
    seed: int,
) -> dict[str, Any]:
    case = demo.make_case(bddl_file=bddl_file)
    env = collector.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed))
    try:
        env.reset()
        demo.set_object_mass(env, demo.PROJECTILE_NAME, float(demo.PROJECTILE_MASS_KG))
        demo.set_object_mass(env, demo.TARGET_NAME, float(target_mass_kg))
        demo.set_object_contact_properties(env, demo.PROJECTILE_NAME, rgba=(0.10, 0.35, 0.95, 1.0))
        demo.set_object_contact_properties(env, demo.TARGET_NAME, rgba=(0.95, 0.25, 0.05, 1.0))
        collector.set_zero_object_velocity(demo, env, demo.PROJECTILE_NAME)
        collector.set_zero_object_velocity(demo, env, demo.TARGET_NAME)
        collector.place_objects_at_exact_initial_poses(demo, env)
        env._last_obs = env._refresh_obs()
        projectile_geoms = demo.object_geom_ids(env, demo.PROJECTILE_NAME)
        demo.establish_projectile_contact(env, projectile_geoms)
        alignment = collector.place_objects_at_exact_initial_poses(demo, env)
        env._last_obs = env._refresh_obs()
        obs = collector.copy_obs(env._last_obs)
        agent, wrist = collector._obs_to_images(obs)
        projectile_xyz, projectile_quat, _ = demo.object_state(env, demo.PROJECTILE_NAME)
        target_xyz, target_quat, _ = demo.object_state(env, demo.TARGET_NAME)
        return {
            "agent": agent,
            "wrist": wrist,
            "agent_sha256": array_sha256(agent),
            "wrist_sha256": array_sha256(wrist),
            "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=float).tolist(),
            "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=float).tolist(),
            "projectile_xyz": np.asarray(projectile_xyz, dtype=float).tolist(),
            "projectile_quat": np.asarray(projectile_quat, dtype=float).tolist(),
            "target_xyz": np.asarray(target_xyz, dtype=float).tolist(),
            "target_quat": np.asarray(target_quat, dtype=float).tolist(),
            "alignment": alignment,
        }
    finally:
        env.close()


def run_visual_audit(
    collector: Any,
    demo: Any,
    pose_state: dict[str, float],
    config: dict[str, Any],
    mass_levels: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    bddl_file = demo.write_two_box_bddl(output_root)
    seed = int(config["visual_no_leak_policy"]["initialization_seed"])
    base_z = float(config["canonical_object_center_z_m"])
    offsets = [float(value) for value in config["action_height_offsets_m"]]
    rows: list[dict[str, Any]] = []
    references: dict[int, dict[str, Any]] = {}
    maximum_agent_difference = 0
    maximum_wrist_difference = 0

    for action_id, offset in enumerate(offsets):
        pose_state["object_center_z_m"] = base_z + offset
        for mass_level in mass_levels:
            result = capture_initial_visual(
                collector,
                demo,
                bddl_file=str(bddl_file),
                target_mass_kg=float(mass_level["target_mass_kg"]),
                seed=seed,
            )
            reference = references.setdefault(action_id, result)
            agent_difference = int(
                np.max(
                    np.abs(
                        result["agent"].astype(np.int16)
                        - reference["agent"].astype(np.int16)
                    )
                )
            )
            wrist_difference = int(
                np.max(
                    np.abs(
                        result["wrist"].astype(np.int16)
                        - reference["wrist"].astype(np.int16)
                    )
                )
            )
            maximum_agent_difference = max(maximum_agent_difference, agent_difference)
            maximum_wrist_difference = max(maximum_wrist_difference, wrist_difference)
            rows.append(
                {
                    "action_id": action_id,
                    "mass_index_short_to_long": int(mass_level["mass_index_short_to_long"]),
                    "target_mass_kg": float(mass_level["target_mass_kg"]),
                    "assigned_object_center_z_m": pose_state["object_center_z_m"],
                    "agent_sha256": result["agent_sha256"],
                    "wrist_sha256": result["wrist_sha256"],
                    "agent_max_abs_difference_from_action_reference": agent_difference,
                    "wrist_max_abs_difference_from_action_reference": wrist_difference,
                    "eef_pos": result["eef_pos"],
                    "eef_quat": result["eef_quat"],
                    "projectile_xyz": result["projectile_xyz"],
                    "target_xyz": result["target_xyz"],
                }
            )
            print(
                f"audit action={action_id} mass={mass_level['target_mass_kg']:.9f}kg "
                f"agent_diff={agent_difference} wrist_diff={wrist_difference}",
                flush=True,
            )

    masses = [float(row["target_mass_kg"]) for row in rows]
    heights = [float(row["assigned_object_center_z_m"]) for row in rows]
    pearson = correlation(masses, heights)
    spearman = correlation(average_ranks(masses), average_ranks(heights))
    threshold = float(config["visual_no_leak_policy"]["mass_height_pearson_required_abs_max"])
    passed = (
        maximum_agent_difference == 0
        and maximum_wrist_difference == 0
        and abs(pearson) <= threshold
        and abs(spearman) <= float(
            config["visual_no_leak_policy"]["mass_height_spearman_required_abs_max"]
        )
    )
    audit = {
        "created_at": dt.datetime.now().isoformat(),
        "passed": passed,
        "definition": "Within each action, all 30 masses must produce bit-identical raw agent and wrist first frames.",
        "mass_count": len(mass_levels),
        "action_count": len(offsets),
        "audited_initializations": len(rows),
        "maximum_agent_pixel_abs_difference": maximum_agent_difference,
        "maximum_wrist_pixel_abs_difference": maximum_wrist_difference,
        "mass_height_pearson": pearson,
        "mass_height_spearman": spearman,
        "height_values_m": [base_z + offset for offset in offsets],
        "initialization_seed_shared_by_all_masses": seed,
        "rows": rows,
    }
    audit_dir = output_root / "visual_leak_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for action_id, reference in references.items():
        Image.fromarray(reference["agent"]).save(audit_dir / f"action_{action_id}_agent_reference.png")
        Image.fromarray(reference["wrist"]).save(audit_dir / f"action_{action_id}_wrist_reference.png")
    write_json(output_root / "visual_leak_audit.json", audit)
    if not passed:
        raise RuntimeError(
            "Visual no-leak audit failed: "
            f"agent_diff={maximum_agent_difference}, wrist_diff={maximum_wrist_difference}, "
            f"pearson={pearson}, spearman={spearman}"
        )
    return audit


def collect_dataset(
    collector: Any,
    demo: Any,
    pose_state: dict[str, float],
    config: dict[str, Any],
    mass_levels: list[dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    audit = json.loads((output_root / "visual_leak_audit.json").read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("Formal collection requires a passed visual no-leak audit")
    dataset_root = output_root / DATASET_NAME
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset_root}")
        shutil.rmtree(dataset_root)

    bddl_file = demo.write_two_box_bddl(output_root)
    collector.patch_lerobot_video_crf(int(args.video_crf))
    dataset_config = dict(config)
    dataset_config["target_masses_kg"] = [float(row["target_mass_kg"]) for row in mass_levels]
    dataset = collector.create_dataset(dataset_root, dataset_config, str(args.video_codec))
    reference_hashes = {
        int(row["action_id"]): (str(row["agent_sha256"]), str(row["wrist_sha256"]))
        for row in audit["rows"]
    }
    metadata: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "episode_count_expected": 270,
        "mass_levels": mass_levels,
        "actions": config["actions"],
        "visual_no_leak_policy": config["visual_no_leak_policy"],
        "visual_leak_audit_file": str(output_root / "visual_leak_audit.json"),
        "sampling_policy": config["sampling_policy"],
        "friction_mu": float(config["friction_mu"]),
        "camera_resolution": int(config["camera_resolution"]),
        "video_codec": str(args.video_codec),
        "video_crf": int(args.video_crf),
        "jpeg_quality": int(args.jpeg_quality),
        "episodes": [],
    }
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "lerobot_root": str(dataset_root),
        "config_path": str(args.config.resolve()),
        "episodes": [],
    }
    action_hashes: dict[int, str] = {}
    seed = int(config["visual_no_leak_policy"]["initialization_seed"])
    base_z = float(config["canonical_object_center_z_m"])
    offsets = [float(value) for value in config["action_height_offsets_m"]]

    def autosave() -> None:
        write_json(output_root / "manifest.json", manifest)
        with (output_root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(collector.to_jsonable(row)) + "\n")
        collector.write_dataset_metadata(dataset_root, metadata, rows)

    count = 0
    for action_cfg in config["actions"]:
        action_id = int(action_cfg["action_id"])
        pose_state["object_center_z_m"] = base_z + offsets[action_id]
        for mass_level in mass_levels:
            records, diagnostics, actions = collector.rollout_episode(
                demo,
                bddl_file=str(bddl_file),
                target_mass_kg=float(mass_level["target_mass_kg"]),
                action_cfg=action_cfg,
                recorded_steps=int(config["recorded_steps"]),
                seed=seed,
            )
            agent, wrist = collector._obs_to_images(records[0]["obs"])
            agent_hash = array_sha256(agent)
            wrist_hash = array_sha256(wrist)
            expected_agent, expected_wrist = reference_hashes[action_id]
            if agent_hash != expected_agent or wrist_hash != expected_wrist:
                raise RuntimeError(
                    f"Visual leak gate failed during formal collection for action={action_id}, "
                    f"mass={mass_level['target_mass_kg']}"
                )
            expected_z = pose_state["object_center_z_m"]
            initial_target_z = float(diagnostics["initial_target_xyz"][2])
            initial_projectile_z = float(diagnostics["initial_projectile_xyz"][2])
            if abs(initial_target_z - expected_z) > 1e-12 or abs(initial_projectile_z - expected_z) > 1e-12:
                raise RuntimeError(
                    f"Initial z changed with mass: expected={expected_z}, "
                    f"target={initial_target_z}, projectile={initial_projectile_z}"
                )
            action_hash = collector.hash_actions(actions)
            if action_id in action_hashes and action_hashes[action_id] != action_hash:
                raise RuntimeError(f"Action hash changed for action={action_id}")
            action_hashes[action_id] = action_hash
            episode_index = collector.save_lerobot_episode(
                dataset,
                records,
                fps=int(config["fps"]),
                jpeg_quality=int(args.jpeg_quality),
            )
            case_id = f"a{action_id:02d}_m{int(mass_level['mass_index_short_to_long']):02d}"
            diagnostics_file = output_root / "diagnostics" / f"episode_{episode_index:06d}_{case_id}.json"
            write_json(diagnostics_file, collector.diagnostics_rows(records))
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "action_id": action_id,
                "A": float(action_cfg["A"]),
                "push_steps": int(action_cfg["push_steps"]),
                "action_sha256": action_hash,
                "mass_index_short_to_long": int(mass_level["mass_index_short_to_long"]),
                "target_mass_kg": float(mass_level["target_mass_kg"]),
                "target_mass_g": float(mass_level["target_mass_g"]),
                "mass_source": str(mass_level["source"]),
                "initialization_seed": seed,
                "assigned_object_center_z_m": expected_z,
                "raw_first_frame_agent_sha256": agent_hash,
                "raw_first_frame_wrist_sha256": wrist_hash,
                "visual_leak_gate_passed": True,
                "diagnostics_file": str(diagnostics_file),
                "metrics": diagnostics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            manifest["episodes"].append(row)
            count += 1
            print(
                f"collect {count:03d}/270 action={action_id} "
                f"mass={mass_level['target_mass_kg']:.9f}kg z={expected_z:.9f}m "
                "visual_gate=pass",
                flush=True,
            )
            autosave()

    summary = {
        "episode_count": len(rows),
        "expected_episode_count": 270,
        "mass_count": len(mass_levels),
        "original_mass_count": sum(row["source"] == "original_20" for row in mass_levels),
        "additional_random_mass_count": sum(
            row["source"] == "additional_stratified_random_10" for row in mass_levels
        ),
        "count_by_action": dict(Counter(row["action_id"] for row in rows)),
        "visual_leak_audit_passed": True,
        "all_formal_first_frame_gates_passed": all(
            row["visual_leak_gate_passed"] for row in rows
        ),
        "mass_height_pearson": audit["mass_height_pearson"],
        "mass_height_spearman": audit["mass_height_spearman"],
        "outcome_conditioned_selection": False,
        "replacement_or_resampling_count": 0,
        "action_sha256_by_action": {str(key): value for key, value in sorted(action_hashes.items())},
        "lerobot_root": str(dataset_root),
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "config_used.json", dataset_config)
    autosave()
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mass_levels = build_mass_levels(config)
    write_json(output_root / "selected_mass_levels.json", mass_levels)
    collector, demo, pose_state = prepare_runtime(config)

    audit_file = output_root / "visual_leak_audit.json"
    if args.audit_only or not audit_file.exists():
        audit = run_visual_audit(
            collector,
            demo,
            pose_state,
            config,
            mass_levels,
            output_root,
        )
        print(
            f"VISUAL_AUDIT_PASS={audit['passed']} "
            f"agent_diff={audit['maximum_agent_pixel_abs_difference']} "
            f"wrist_diff={audit['maximum_wrist_pixel_abs_difference']} "
            f"pearson={audit['mass_height_pearson']:.3e} "
            f"spearman={audit['mass_height_spearman']:.3e}",
            flush=True,
        )
    if args.audit_only:
        return
    collect_dataset(collector, demo, pose_state, config, mass_levels, output_root, args)


if __name__ == "__main__":
    main()

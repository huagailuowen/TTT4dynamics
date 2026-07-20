#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_plus_push_box_full_trajectory_preview_lerobot_2026-07-18_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_plus_push_box_official_scenes_full_trajectory_9demo_2026-07-18_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "libero_plus_official_assets_full_trajectory_preview"
    / "libero_plus_push_box_official_assets_full_trajectory_9eps_lerobot_2026-07-18_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full = load_module(SOURCE_SCRIPT, "official_asset_full_trajectory_source_hai_machine")
legacy = full.legacy
original_rollout = full.full_trajectory_rollout
original_gripper_collision_ids = legacy._native_gripper_collision_ids


def native_finger_collision_ids(env: Any) -> list[int]:
    finger_ids = [
        geom_id
        for geom_id in original_gripper_collision_ids(env)
        if "hand_collision" not in (env.inner_env.sim.model.geom_id2name(geom_id) or "")
    ]
    if not finger_ids:
        raise RuntimeError("No native Panda finger collision geoms found")
    return finger_ids


def native_target_collision_ids(env: Any) -> list[int]:
    sim = env.inner_env.sim
    obj = env.inner_env.get_object(env.case.box_name)
    prefix = f"{obj.name}_"
    result = [
        geom_id
        for geom_id in _named_ids(sim.model, "geom", prefix)
        if int(sim.model.geom_contype[geom_id]) != 0
        or int(sim.model.geom_conaffinity[geom_id]) != 0
    ]
    if not result:
        raise RuntimeError(f"No collision geoms found for push target {obj.name}")
    return result


def _target_projection_bounds(env: Any, axis: np.ndarray) -> tuple[float, float]:
    sim = env.inner_env.sim
    lower = np.inf
    upper = -np.inf
    for geom_id in native_target_collision_ids(env):
        center = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
        projection = float(np.dot(center, axis))
        support = legacy._geom_support(env, geom_id, axis)
        lower = min(lower, projection - support)
        upper = max(upper, projection + support)
    return float(lower), float(upper)


def native_target_contact_footprint(
    env: Any,
    direction: np.ndarray,
    target_eef_z: float,
) -> dict[str, Any]:
    sim = env.inner_env.sim
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    direction3 = np.asarray([direction[0], direction[1], 0.0], dtype=np.float64)
    lateral3 = np.asarray([lateral[0], lateral[1], 0.0], dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
    target_center, _ = env.box_pose()
    target_center = np.asarray(target_center, dtype=np.float64)
    target_rear, _ = _target_projection_bounds(env, direction3)
    target_z_min, target_z_max = _target_projection_bounds(env, up)
    candidates = []
    for geom_id in native_finger_collision_ids(env):
        center = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
        relative = center - eef
        predicted_z = float(target_eef_z + relative[2])
        z_support = legacy._geom_support(env, geom_id, up)
        if predicted_z + z_support < target_z_min - 0.003:
            continue
        if predicted_z - z_support > target_z_max + 0.003:
            continue
        candidates.append(
            {
                "geom_id": int(geom_id),
                "name": sim.model.geom_id2name(geom_id),
                "front_offset_m": float(
                    np.dot(relative[:2], direction)
                    + legacy._geom_support(env, geom_id, direction3)
                ),
                "lateral_offset_m": float(np.dot(relative[:2], lateral)),
                "lateral_support_m": legacy._geom_support(env, geom_id, lateral3),
                "predicted_center_z_m": predicted_z,
            }
        )
    if not candidates:
        raise RuntimeError(
            f"No native gripper collision geom overlaps target height at eef_z={target_eef_z}"
        )
    front_offset = max(float(item["front_offset_m"]) for item in candidates)
    leading = [
        item
        for item in candidates
        if front_offset - float(item["front_offset_m"]) <= 0.004
    ]
    return {
        "box_center": target_center,
        "box_support_m": float(np.dot(target_center, direction3) - target_rear),
        "front_offset_from_eef_m": front_offset,
        "leading_lateral_offset_from_eef_m": float(
            np.mean([float(item["lateral_offset_m"]) for item in leading])
        ),
        "leading_geoms": leading,
        "candidate_geoms": candidates,
        "target_collision_geom_ids": native_target_collision_ids(env),
        "target_z_interval_m": [target_z_min, target_z_max],
    }


def native_all_gripper_box_contact(env: Any) -> bool:
    sim = env.inner_env.sim
    target_ids = set(native_target_collision_ids(env))
    gripper_ids = set(original_gripper_collision_ids(env))
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair.intersection(target_ids) and pair.intersection(gripper_ids):
            return True
    return False


def _active_config_path() -> Path:
    for index, argument in enumerate(sys.argv):
        if argument == "--config" and index + 1 < len(sys.argv):
            requested = Path(sys.argv[index + 1])
            return requested if requested.is_absolute() else (Path.cwd() / requested).resolve()
        if argument.startswith("--config="):
            requested = Path(argument.split("=", 1)[1])
            return requested if requested.is_absolute() else (Path.cwd() / requested).resolve()
    return CONFIG_PATH


def _insert_before_section_close(
    text: str,
    section: str,
    next_section: str,
    insertion: str,
) -> str:
    section_start = text.index(section)
    next_start = text.index(next_section, section_start)
    section_close = text.rfind(")", section_start, next_start)
    if section_close < 0:
        raise RuntimeError(f"Could not locate the closing parenthesis for {section}")
    return text[:section_close] + insertion + text[section_close:]


def _sample_distractors(
    distractor_cfg: dict[str, Any],
    seed: int,
    excluded_bddl_type: str,
) -> list[dict[str, Any]]:
    if not distractor_cfg.get("enabled", False):
        return []
    count_cycle = [int(value) for value in distractor_cfg["count_cycle"]]
    count = count_cycle[seed % len(count_cycle)]
    asset_pool = [
        item
        for item in distractor_cfg["asset_pool"]
        if str(item["bddl_type"]) != excluded_bddl_type
    ]
    anchors = list(distractor_cfg["safe_anchor_xy_m"])
    if count > min(len(asset_pool), len(anchors)):
        raise ValueError("Distractor count exceeds the unique asset or safe-anchor pool")
    rng = np.random.default_rng(seed ^ 0xD157AC7)
    asset_ids = rng.choice(len(asset_pool), size=count, replace=False)
    anchor_ids = rng.choice(len(anchors), size=count, replace=False)
    selected = []
    for slot, (asset_id, anchor_id) in enumerate(zip(asset_ids, anchor_ids), start=1):
        asset = dict(asset_pool[int(asset_id)])
        center_xy = [float(value) for value in anchors[int(anchor_id)]]
        asset.update(
            {
                "object_name": f"{asset['bddl_type']}_{slot}",
                "region_name": f"distractor_{slot:02d}_region",
                "region_reference": f"main_table_distractor_{slot:02d}_region",
                "safe_anchor_id": int(anchor_id),
                "safe_anchor_xy_m": center_xy,
            }
        )
        selected.append(asset)
    return selected


def _inject_distractors(
    text: str,
    distractors: list[dict[str, Any]],
    distractor_cfg: dict[str, Any],
) -> str:
    if not distractors:
        return text
    half_x, half_y = [float(value) for value in distractor_cfg["region_half_extent_xy_m"]]
    yaw_min, yaw_max = [float(value) for value in distractor_cfg["yaw_range_rad"]]
    region_blocks = ""
    for item in distractors:
        center_x, center_y = item["safe_anchor_xy_m"]
        region_blocks += (
            f"\n      ({item['region_name']}\n"
            "        (:target main_table)\n"
            "        (:ranges (\n"
            f"          ({center_x - half_x:.6f} {center_y - half_y:.6f} "
            f"{center_x + half_x:.6f} {center_y + half_y:.6f})\n"
            "        ))\n"
            "        (:yaw_rotation (\n"
            f"          ({yaw_min:.6f} {yaw_max:.6f})\n"
            "        ))\n"
            "      )\n"
        )
    text = _insert_before_section_close(text, "(:regions", "(:fixtures", region_blocks)
    object_lines = "".join(
        f"    {item['object_name']} - {item['bddl_type']}\n" for item in distractors
    )
    text = _insert_before_section_close(text, "(:objects", "(:obj_of_interest", object_lines)
    interest_lines = "".join(f"    {item['object_name']}\n" for item in distractors)
    text = _insert_before_section_close(text, "(:obj_of_interest", "(:init", interest_lines)
    init_lines = "".join(
        f"    (On {item['object_name']} {item['region_reference']})\n"
        for item in distractors
    )
    return _insert_before_section_close(text, "(:init", "(:goal", init_lines)


def _asset_case(
    case: Any,
    variant: dict[str, Any],
    distractor_cfg: dict[str, Any],
    seed: int,
) -> tuple[Any, list[dict[str, Any]]]:
    asset_type = str(variant["bddl_type"])
    source = Path(case.bddl_file)
    if not source.is_absolute():
        source = REPO_ROOT / source
    text = source.read_text(encoding="utf-8")
    declaration = "cream_cheese_1 - cream_cheese"
    if declaration not in text:
        raise RuntimeError(f"Expected object declaration not found in {source}: {declaration}")
    text = text.replace(declaration, f"cream_cheese_1 - {asset_type}", 1)
    text = text.replace(
        "push the cream cheese box",
        f"push the {str(variant['display_name']).lower()}",
        1,
    )
    distractors = _sample_distractors(distractor_cfg, seed, asset_type)
    text = _inject_distractors(text, distractors, distractor_cfg)
    clutter_suffix = f"_clutter_seed{seed}" if distractor_cfg.get("enabled", False) else ""
    destination = source.with_name(
        f"{source.stem}_official_{asset_type}{clutter_suffix}{source.suffix}"
    )
    destination.write_text(text, encoding="utf-8")
    updated = replace(case, bddl_file=str(destination))
    for name, value in vars(case).items():
        if not hasattr(updated, name):
            object.__setattr__(updated, name, value)
    return updated, distractors


def _named_ids(model: Any, kind: str, prefix: str) -> list[int]:
    count = int(getattr(model, f"n{kind}"))
    name_for_id = getattr(model, f"{kind}_id2name")
    return [index for index in range(count) if (name_for_id(index) or "").startswith(prefix)]


def set_official_asset_variant_and_pose(
    env: Any,
    *,
    variant: dict[str, Any],
    rng: np.random.Generator,
    base_init_xy: np.ndarray,
    position_cfg: dict[str, Any],
    table_rgb: np.ndarray,
    minimum_contrast: float,
) -> dict[str, Any]:
    del table_rgb, minimum_contrast
    sim = env.inner_env.sim
    model = sim.model
    obj = env.inner_env.get_object(env.case.box_name)
    prefix = f"{obj.name}_"
    geom_ids = _named_ids(model, "geom", prefix)
    if not geom_ids:
        raise RuntimeError(f"No compiled geoms found for official asset {obj.name}")

    body_ids = [
        body_id
        for body_id in _named_ids(model, "body", prefix)
        if float(model.body_mass[body_id]) > 0.0
    ]
    complete_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    complete_qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    original_mass = float(sum(float(model.body_mass[body_id]) for body_id in body_ids))
    target_mass = float(variant["target_mass_kg"])
    if body_ids and original_mass > 0.0:
        scale = target_mass / original_mass
        for body_id in body_ids:
            model.body_mass[body_id] *= scale
            model.body_inertia[body_id] *= scale
        native_model = getattr(model, "_model", None)
        native_data = getattr(sim.data, "_data", None)
        if native_model is not None and native_data is not None:
            mujoco.mj_setConst(native_model, native_data)
            sim.data.qpos[:] = complete_qpos
            sim.data.qvel[:] = complete_qvel
            sim.forward()

    joint_name = obj.joints[-1]
    qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    collision_ids = native_target_collision_ids(env)
    original_target_z_min, original_target_z_max = _target_projection_bounds(env, up)
    table_surface_z = original_target_z_min
    preview_seed = int(variant.get("_preview_seed", 0))
    front_back_cycle = position_cfg.get("front_back_x_offset_cycle_m")
    horizontal_cycle = position_cfg.get("horizontal_y_offset_cycle_m")
    front_back_offset = (
        float(front_back_cycle[preview_seed % len(front_back_cycle)])
        if front_back_cycle
        else float(rng.uniform(*position_cfg["front_back_x_jitter_m"]))
    )
    horizontal_offset = (
        float(horizontal_cycle[preview_seed % len(horizontal_cycle)])
        if horizontal_cycle
        else float(rng.uniform(*position_cfg["horizontal_y_jitter_m"]))
    )
    qpos[0] = float(base_init_xy[0] + front_back_offset)
    qpos[1] = float(base_init_xy[1] + horizontal_offset)
    qpos[3:7] = np.asarray(variant["initial_quat_wxyz"], dtype=np.float64)
    sim.data.set_joint_qpos(joint_name, qpos)
    sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()
    standing_target_z_min, standing_target_z_max = _target_projection_bounds(env, up)
    standing_vertical_support = 0.5 * (standing_target_z_max - standing_target_z_min)
    support_adjustment = table_surface_z - standing_target_z_min
    qpos[2] += support_adjustment
    sim.data.set_joint_qpos(joint_name, qpos)
    sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()

    direction_angle = np.deg2rad(float(env.case.pusher_push_angle_deg))
    direction = np.asarray([np.cos(direction_angle), np.sin(direction_angle)], dtype=np.float64)
    configured_contact_z = float(
        variant.get("native_contact_eef_z_m", env.case.pusher_contact_z)
    )
    valid_contact_heights = []
    for candidate_z in np.linspace(
        configured_contact_z - 0.35,
        configured_contact_z + 0.35,
        1401,
        dtype=np.float64,
    ):
        try:
            footprint = legacy._native_contact_footprint(env, direction, float(candidate_z))
        except RuntimeError:
            continue
        box_center_z = float(np.asarray(footprint["box_center"], dtype=np.float64)[2])
        leading_center_z = float(
            np.mean([float(item["predicted_center_z_m"]) for item in footprint["leading_geoms"]])
        )
        valid_contact_heights.append(
            (
                abs(float(candidate_z) - configured_contact_z),
                abs(leading_center_z - box_center_z),
                float(candidate_z),
                footprint,
            )
        )
    if not valid_contact_heights:
        diagnostic_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
        diagnostic_geoms = [
            {
                "name": model.geom_id2name(geom_id),
                "world_xyz": np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64).tolist(),
                "relative_to_obs_eef_xyz": (
                    np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64) - diagnostic_eef
                ).tolist(),
            }
            for geom_id in legacy._native_gripper_collision_ids(env)
        ]
        raise RuntimeError(
            f"No geometry-valid native-gripper EEF height found for {variant['name']} around "
            f"configured z={configured_contact_z:.4f} m; obs_eef={diagnostic_eef.tolist()}; "
            f"target_geom_world={[np.asarray(sim.data.geom_xpos[geom_id], dtype=float).tolist() for geom_id in collision_ids]}; "
            f"gripper_geoms={diagnostic_geoms}"
        )
    _, _, contact_eef_z, selected_footprint = min(valid_contact_heights, key=lambda item: item[:2])
    leading = selected_footprint["leading_geoms"]
    object.__setattr__(env.case, "pusher_contact_z", contact_eef_z)

    for geom_id in collision_ids:
        model.geom_friction[geom_id][0] = float(env.case.friction_mu)
    visual_ids = [geom_id for geom_id in geom_ids if geom_id not in collision_ids]
    material_names = []
    for geom_id in visual_ids:
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            material_names.append(
                mujoco.mj_id2name(
                    model._model,
                    mujoco.mjtObj.mjOBJ_MATERIAL,
                    material_id,
                )
                or ""
            )
    return {
        "object_id": int(variant["object_id"]),
        "name": str(variant["name"]),
        "display_name": str(variant["display_name"]),
        "bddl_type": str(variant["bddl_type"]),
        "asset_xml": str(variant["asset_xml"]),
        "collision_model": str(variant["collision_model"]),
        "collision_half_size_local_m": [float(value) for value in variant["collision_half_size_local_m"]],
        "visual_materials": sorted(set(material_names)),
        "compiled_geom_count": len(geom_ids),
        "compiled_collision_geom_count": len(collision_ids),
        "original_mass_kg": original_mass,
        "target_mass_kg": target_mass,
        "requested_initial_xyz_m": qpos[:3].astype(float).tolist(),
        "requested_initial_quat_wxyz": qpos[3:7].astype(float).tolist(),
        "standing_vertical_support_m": float(standing_vertical_support),
        "original_target_z_interval_m": [original_target_z_min, original_target_z_max],
        "standing_target_z_interval_m": [standing_target_z_min, standing_target_z_max],
        "vertical_support_adjustment_m": float(support_adjustment),
        "geometry_adaptive_contact_eef_z_m": contact_eef_z,
        "leading_native_gripper_geom_ids": [int(item["geom_id"]) for item in leading],
        "official_visual_mesh_and_uv_preserved": True,
    }


def official_asset_rollout(case: Any, *args: Any, **kwargs: Any) -> Any:
    variant = kwargs.get("variant")
    if variant is None:
        raise TypeError("official_asset_rollout requires variant metadata")
    rollout_experiment = kwargs.get("experiment")
    if not isinstance(rollout_experiment, dict):
        rollout_experiment = experiment
    distractor_cfg = dict(rollout_experiment.get("distractor_randomization", {}))
    seed = int(kwargs.get("seed", getattr(case, "seed", 0)))
    rollout_variant = dict(variant)
    rollout_variant["_preview_seed"] = seed
    kwargs["variant"] = rollout_variant
    updated_case, distractors = _asset_case(case, rollout_variant, distractor_cfg, seed)
    result = original_rollout(updated_case, *args, **kwargs)
    result[1]["distractors"] = {
        "count": len(distractors),
        "objects": distractors,
        "target_only_action_and_labels": bool(
            distractor_cfg.get("target_only_action_and_labels", True)
        ),
    }
    print(
        "FULL_TRAJECTORY_ERRORS "
        + json.dumps(result[1]["full_trajectory"], sort_keys=True),
        flush=True,
    )
    return result


experiment = json.loads(_active_config_path().read_text(encoding="utf-8"))
legacy.CONFIG_PATH = CONFIG_PATH
legacy.DEFAULT_OUTPUT = DEFAULT_OUTPUT
legacy._NATIVE_PRESETS = experiment["background_randomization"]["presets"]
legacy._native_gripper_collision_ids = native_finger_collision_ids
legacy._native_contact_footprint = native_target_contact_footprint
legacy.board_box_contact = native_all_gripper_box_contact
legacy.set_object_variant_and_pose = set_official_asset_variant_and_pose
legacy.rollout_preview = official_asset_rollout


if __name__ == "__main__":
    legacy.main()

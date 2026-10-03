#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
from typing import Any, Iterable

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTED_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_event_tap_segmented80_10action_lerobot_2026-07-05_hai-machine.py"
)
OFFICIAL_ASSET_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_plus_push_box_official_assets_full_trajectory_preview_lerobot_2026-07-18_hai-machine.py"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "libero_plus_push_box_segmented40_3env_matched_physics_2026-07-27_hai-machine.json"
)
DEFAULT_PREVIEW_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "libero_plus_push_box_segmented80_physics_3env_3friction_3action_comparison_preview_2026-07-27_hai-machine"
)
DEFAULT_FORMAL_OUTPUT = (
    REPO_ROOT
    / "data"
    / "pushbox_various_env"
    / "libero_plus_push_box_event_tap_segmented40_10action_3env_hidden_lerobot_A500_offset160_stop_2026-07-27_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


official = load_module(OFFICIAL_ASSET_SCRIPT, "official_visual_source_20260727")
segmented = load_module(SEGMENTED_SCRIPT, "segmented80_matched_visual_source_20260727")
OriginalLiberoPushBoxEnv = official.legacy.base.LiberoPushBoxEnv

_ACTIVE_SETUP: dict[str, Any] = {}
_ACTIVE_RESULT: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a physics-matched three-environment preview or approved formal "
            "LeRobot dataset using the exact segmented80 event-tap rollout."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["preview", "formal"], default="preview")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-codec",
        choices=["h264", "hevc", "libsvtav1", "h264_nvenc"],
    )
    parser.add_argument("--video-crf", type=int)
    parser.add_argument("--jpeg-quality", type=int)
    parser.add_argument("--skip-comparison-videos", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(segmented.base.to_jsonable(value), indent=2),
        encoding="utf-8",
    )


def build_visual_overlay_case(
    case: Any,
    *,
    environment: dict[str, Any],
) -> Any:
    source = Path(case.bddl_file)
    if not source.is_absolute():
        source = REPO_ROOT / source
    text = source.read_text(encoding="utf-8")
    variant = environment["object_variant"]
    requested_overlay_type = str(variant["bddl_type"])
    dummy_overlay = requested_overlay_type == "cream_cheese"
    overlay_type = "butter" if dummy_overlay else requested_overlay_type
    overlay_object_name = (
        f"{overlay_type}_1"
    )
    region_block = (
        "\n      (visual_overlay_staging_region\n"
        "        (:target main_table)\n"
        "        (:ranges (\n"
        "          (0.400000 0.180000 0.402000 0.182000)\n"
        "        ))\n"
        "        (:yaw_rotation (\n"
        "          (0.000000 0.000000)\n"
        "        ))\n"
        "      )\n"
    )
    text = official._insert_before_section_close(
        text,
        "(:regions",
        "(:fixtures",
        region_block,
    )
    text = official._insert_before_section_close(
        text,
        "(:objects",
        "(:obj_of_interest",
        f"    {overlay_object_name} - {overlay_type}\n",
    )
    text = official._insert_before_section_close(
        text,
        "(:obj_of_interest",
        "(:init",
        f"    {overlay_object_name}\n",
    )
    text = official._insert_before_section_close(
        text,
        "(:init",
        "(:goal",
        (
            f"    (On {overlay_object_name} "
            "main_table_visual_overlay_staging_region)\n"
        ),
    )
    text = text.replace(
        "push the cream cheese box",
        f"push the {str(variant['display_name']).lower()}",
        1,
    )
    destination = source.with_name(
        f"{source.stem}_visual_overlay_{requested_overlay_type}{source.suffix}"
    )
    destination.write_text(text, encoding="utf-8")
    updated = replace(case, bddl_file=str(destination))
    for name, value in vars(case).items():
        if not hasattr(updated, name):
            object.__setattr__(updated, name, value)
    object.__setattr__(
        updated,
        "hai_visual_overlay_object_name",
        overlay_object_name,
    )
    object.__setattr__(
        updated,
        "hai_visual_overlay_is_dummy",
        dummy_overlay,
    )
    return updated


def friction_values(config: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for segment in config["friction_schedule"]["segments"]:
        values.extend(
            np.linspace(
                float(segment["start"]),
                float(segment["stop"]),
                int(segment["count"]),
                endpoint=bool(segment["endpoint"]),
                dtype=np.float64,
            ).astype(float).tolist()
        )
    expected = int(config["friction_schedule"]["expected_count"])
    if len(values) != expected or len(set(values)) != expected:
        raise RuntimeError(
            f"Invalid friction schedule: got {len(values)} values, expected {expected}"
        )
    return values


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(result, left, right)
    return result


def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_negQuat(result, quaternion)
    return result


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    result = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, quaternion)
    return result.reshape(3, 3)


def normalized_quaternion(values: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(list(values), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("Quaternion cannot be zero")
    return quaternion / norm


def normalize_target_physics_and_pose(
    env: Any,
    *,
    environment: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    obj = env.inner_env.get_object(env.case.box_name)
    prefix = f"{obj.name}_"
    geom_ids = official._named_ids(model, "geom", prefix)
    collision_ids = official.native_target_collision_ids(env)
    visual_ids = [geom_id for geom_id in geom_ids if geom_id not in collision_ids]
    body_ids = [
        body_id
        for body_id in official._named_ids(model, "body", prefix)
        if float(model.body_mass[body_id]) > 0.0
    ]
    if len(collision_ids) != 1:
        raise RuntimeError(
            f"Expected one target collision geom for {environment['environment_id']}, "
            f"got {collision_ids}"
        )
    if len(body_ids) != 1:
        raise RuntimeError(
            f"Expected one dynamic target body for {environment['environment_id']}, "
            f"got {body_ids}"
        )

    collision_id = collision_ids[0]
    body_id = body_ids[0]
    variant = environment["object_variant"]
    source_visual_quat = normalized_quaternion(variant["visual_pose_quat_wxyz"])
    canonical_qpos = np.asarray(
        canonical["compiled_initial_free_joint_qpos"],
        dtype=np.float64,
    )
    canonical_body_quat = normalized_quaternion(canonical_qpos[3:7])
    canonical_collision_quat = normalized_quaternion(
        canonical["collision_quat_local_wxyz"]
    )
    canonical_inertia_quat = normalized_quaternion(canonical["body_iquat_wxyz"])
    desired_world_collision_quat = quaternion_multiply(
        canonical_body_quat,
        canonical_collision_quat,
    )
    desired_world_inertia_quat = quaternion_multiply(
        canonical_body_quat,
        canonical_inertia_quat,
    )
    source_visual_inverse = quaternion_inverse(source_visual_quat)
    matched_local_collision_quat = quaternion_multiply(
        source_visual_inverse,
        desired_world_collision_quat,
    )
    matched_local_inertia_quat = quaternion_multiply(
        source_visual_inverse,
        desired_world_inertia_quat,
    )

    source_collision_size = np.asarray(
        variant["native_collision_half_size_local_m"],
        dtype=np.float64,
    )
    source_collision_pos = np.asarray(
        variant["native_collision_pos_local_m"],
        dtype=np.float64,
    )
    source_collision_quat = normalized_quaternion(
        variant["native_collision_quat_local_wxyz"]
    )
    source_body_rotation = quaternion_matrix(source_visual_quat)
    source_collision_rotation = (
        source_body_rotation @ quaternion_matrix(source_collision_quat)
    )
    source_collision_center_world = source_body_rotation @ source_collision_pos
    source_visual_vertical_support = float(
        np.sum(np.abs(source_collision_rotation[2, :]) * source_collision_size)
    )
    source_visual_bottom_relative_z = float(
        source_collision_center_world[2] - source_visual_vertical_support
    )
    canonical_collision_size = np.asarray(
        canonical["collision_half_size_local_m"],
        dtype=np.float64,
    )
    canonical_world_collision_rotation = quaternion_matrix(
        desired_world_collision_quat
    )
    canonical_vertical_support = float(
        np.sum(
            np.abs(canonical_world_collision_rotation[2, :])
            * canonical_collision_size
        )
    )
    visual_world_z_shift = float(
        -canonical_vertical_support - source_visual_bottom_relative_z
    )
    visual_local_shift = source_body_rotation.T @ np.asarray(
        [0.0, 0.0, visual_world_z_shift],
        dtype=np.float64,
    )
    for geom_id in visual_ids:
        model.geom_pos[geom_id] += visual_local_shift
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    model.body_mass[body_id] = float(canonical["body_mass_kg"])
    model.body_inertia[body_id][:] = np.asarray(
        canonical["body_inertia_kg_m2"],
        dtype=np.float64,
    )
    model.body_ipos[body_id][:] = np.asarray(
        canonical["body_ipos_m"],
        dtype=np.float64,
    )
    model.body_iquat[body_id][:] = matched_local_inertia_quat

    model.geom_type[collision_id] = int(mujoco.mjtGeom.mjGEOM_BOX)
    model.geom_size[collision_id][:] = canonical_collision_size
    model.geom_pos[collision_id][:] = np.asarray(
        canonical["collision_pos_local_m"],
        dtype=np.float64,
    )
    model.geom_quat[collision_id][:] = matched_local_collision_quat
    model.geom_contype[collision_id] = 1
    model.geom_conaffinity[collision_id] = 1
    model.geom_condim[collision_id] = int(canonical["collision_condim"])
    model.geom_margin[collision_id] = float(canonical["collision_margin_m"])
    model.geom_gap[collision_id] = float(canonical["collision_gap_m"])
    model.geom_solref[collision_id][:] = np.asarray(
        canonical["collision_solref"],
        dtype=np.float64,
    )
    model.geom_solimp[collision_id][:] = np.asarray(
        canonical["collision_solimp"],
        dtype=np.float64,
    )
    model.geom_friction[collision_id][:] = np.asarray(
        [
            float(env.case.friction_mu),
            float(canonical["friction_torsional"]),
            float(canonical["friction_rolling"]),
        ],
        dtype=np.float64,
    )

    native_model = getattr(model, "_model", None)
    native_data = getattr(sim.data, "_data", None)
    if native_model is None or native_data is None:
        raise RuntimeError("Native MuJoCo model/data handles are unavailable")
    mujoco.mj_setConst(native_model, native_data)
    canonical_collision_aabb = np.concatenate(
        [
            np.asarray(canonical["collision_pos_local_m"], dtype=np.float64),
            canonical_collision_size,
        ]
    )
    canonical_collision_rbound = float(np.linalg.norm(canonical_collision_size))
    model.geom_aabb[collision_id][:] = canonical_collision_aabb
    model.geom_rbound[collision_id] = canonical_collision_rbound

    joint_name = obj.joints[-1]
    target_qpos = canonical_qpos.copy()
    target_qpos[3:7] = source_visual_quat
    sim.data.set_joint_qpos(joint_name, target_qpos)
    sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()

    actual_world_collision_rotation = np.asarray(
        sim.data.geom_xmat[collision_id],
        dtype=np.float64,
    ).reshape(3, 3)
    if not np.allclose(
        actual_world_collision_rotation,
        canonical_world_collision_rotation,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"World collision rotation mismatch for {environment['environment_id']}"
        )
    if not np.allclose(
        np.asarray(model.geom_size[collision_id], dtype=np.float64),
        canonical_collision_size,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Collision size mismatch for {environment['environment_id']}"
        )
    if not np.allclose(
        np.asarray(model.geom_aabb[collision_id], dtype=np.float64),
        canonical_collision_aabb,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Collision AABB mismatch for {environment['environment_id']}"
        )

    return {
        "environment_id": str(environment["environment_id"]),
        "target_object_name": str(obj.name),
        "dynamic_body_id": int(body_id),
        "collision_geom_id": int(collision_id),
        "visual_geom_ids": [int(value) for value in visual_ids],
        "actual_body_mass_kg": float(model.body_mass[body_id]),
        "actual_body_inertia_kg_m2": np.asarray(
            model.body_inertia[body_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_half_size_local_m": np.asarray(
            model.geom_size[collision_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_aabb_local_m": np.asarray(
            model.geom_aabb[collision_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_rbound_m": float(model.geom_rbound[collision_id]),
        "actual_collision_friction": np.asarray(
            model.geom_friction[collision_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_world_rotation": actual_world_collision_rotation.astype(
            float
        ).tolist(),
        "requested_free_joint_qpos": target_qpos.astype(float).tolist(),
        "visual_world_z_shift_m": visual_world_z_shift,
        "visual_local_shift_m": visual_local_shift.astype(float).tolist(),
        "visual_mesh_and_uv_preserved": True,
        "visual_geoms_noncolliding": True,
        "physics_collision_and_world_inertia_matched_to_segmented80": True,
    }


def configure_classic_physical_target(
    env: Any,
    *,
    canonical: dict[str, Any],
    hide_visual: bool,
) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    obj = env.inner_env.get_object(env.case.box_name)
    prefix = f"{obj.name}_"
    geom_ids = official._named_ids(model, "geom", prefix)
    collision_ids = official.native_target_collision_ids(env)
    visual_ids = [geom_id for geom_id in geom_ids if geom_id not in collision_ids]
    body_ids = [
        body_id
        for body_id in official._named_ids(model, "body", prefix)
        if float(model.body_mass[body_id]) > 0.0
    ]
    if len(collision_ids) != 1 or len(body_ids) != 1:
        raise RuntimeError("Classic target did not compile to one body and one collision")
    collision_id = collision_ids[0]
    body_id = body_ids[0]
    joint_name = obj.joints[-1]
    canonical_qpos = np.asarray(
        canonical["compiled_initial_free_joint_qpos"],
        dtype=np.float64,
    )
    sim.data.set_joint_qpos(joint_name, canonical_qpos)
    sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    if hide_visual:
        for geom_id in visual_ids:
            model.geom_rgba[geom_id][3] = 0.0
            material_id = int(model.geom_matid[geom_id])
            if material_id >= 0:
                model.mat_rgba[material_id][3] = 0.0
    sim.forward()

    expected_size = np.asarray(
        canonical["collision_half_size_local_m"],
        dtype=np.float64,
    )
    expected_inertia = np.asarray(
        canonical["body_inertia_kg_m2"],
        dtype=np.float64,
    )
    checks = [
        np.isclose(
            float(model.body_mass[body_id]),
            float(canonical["body_mass_kg"]),
            atol=1e-12,
            rtol=0.0,
        ),
        np.allclose(
            np.asarray(model.body_inertia[body_id], dtype=np.float64),
            expected_inertia,
            atol=1e-12,
            rtol=0.0,
        ),
        np.allclose(
            np.asarray(model.geom_size[collision_id], dtype=np.float64),
            expected_size,
            atol=1e-12,
            rtol=0.0,
        ),
        np.allclose(
            np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64),
            canonical_qpos,
            atol=1e-12,
            rtol=0.0,
        ),
    ]
    if not all(checks):
        raise RuntimeError("Compiled classic target does not match segmented80")
    return {
        "target_object_name": str(obj.name),
        "dynamic_body_id": int(body_id),
        "collision_geom_id": int(collision_id),
        "hidden_classic_visual_geom_ids": [int(value) for value in visual_ids],
        "classic_visual_hidden": bool(hide_visual),
        "actual_body_mass_kg": float(model.body_mass[body_id]),
        "actual_body_inertia_kg_m2": np.asarray(
            model.body_inertia[body_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_half_size_local_m": np.asarray(
            model.geom_size[collision_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_aabb_local_m": np.asarray(
            model.geom_aabb[collision_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_rbound_m": float(model.geom_rbound[collision_id]),
        "actual_collision_friction": np.asarray(
            model.geom_friction[collision_id],
            dtype=np.float64,
        ).astype(float).tolist(),
        "actual_collision_world_rotation": np.asarray(
            sim.data.geom_xmat[collision_id],
            dtype=np.float64,
        ).reshape(3, 3).astype(float).tolist(),
        "actual_free_joint_qpos": np.asarray(
            sim.data.get_joint_qpos(joint_name),
            dtype=np.float64,
        ).astype(float).tolist(),
        "compiled_as_original_segmented80_cream_cheese": True,
    }


def sync_visual_overlay(env: Any) -> None:
    state = getattr(env, "_hai_visual_overlay_state", None)
    if not state:
        return
    sim = env.inner_env.sim
    target_qpos = np.asarray(
        sim.data.get_joint_qpos(state["target_joint_name"]),
        dtype=np.float64,
    )
    target_quat = normalized_quaternion(target_qpos[3:7])
    overlay_qpos = target_qpos.copy()
    overlay_qpos[:3] = (
        target_qpos[:3]
        + quaternion_matrix(target_quat)
        @ np.asarray(state["relative_position_local_m"], dtype=np.float64)
    )
    overlay_qpos[3:7] = quaternion_multiply(
        target_quat,
        np.asarray(state["relative_quat_wxyz"], dtype=np.float64),
    )
    sim.data.set_joint_qpos(state["overlay_joint_name"], overlay_qpos)
    sim.data.set_joint_qvel(
        state["overlay_joint_name"],
        np.zeros(6, dtype=np.float64),
    )
    sim.forward()


def configure_visual_overlay(
    env: Any,
    *,
    environment: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    target = env.inner_env.get_object(env.case.box_name)

    variant = environment["object_variant"]
    canonical_qpos = np.asarray(
        canonical["compiled_initial_free_joint_qpos"],
        dtype=np.float64,
    )
    canonical_quat = normalized_quaternion(canonical_qpos[3:7])
    visual_quat = normalized_quaternion(variant["visual_pose_quat_wxyz"])
    relative_quat = quaternion_multiply(
        quaternion_inverse(canonical_quat),
        visual_quat,
    )
    source_collision_size = np.asarray(
        variant["native_collision_half_size_local_m"],
        dtype=np.float64,
    )
    source_collision_pos = np.asarray(
        variant["native_collision_pos_local_m"],
        dtype=np.float64,
    )
    source_collision_quat = normalized_quaternion(
        variant["native_collision_quat_local_wxyz"]
    )
    visual_rotation = quaternion_matrix(visual_quat)
    source_world_collision_rotation = (
        visual_rotation @ quaternion_matrix(source_collision_quat)
    )
    source_center_world = visual_rotation @ source_collision_pos
    source_vertical_support = float(
        np.sum(
            np.abs(source_world_collision_rotation[2, :])
            * source_collision_size
        )
    )
    canonical_world_collision_rotation = quaternion_matrix(
        quaternion_multiply(
            canonical_quat,
            normalized_quaternion(canonical["collision_quat_local_wxyz"]),
        )
    )
    canonical_size = np.asarray(
        canonical["collision_half_size_local_m"],
        dtype=np.float64,
    )
    canonical_vertical_support = float(
        np.sum(
            np.abs(canonical_world_collision_rotation[2, :])
            * canonical_size
        )
    )
    visual_world_z_shift = float(
        source_vertical_support
        - float(source_center_world[2])
        - canonical_vertical_support
    )
    relative_position_local = quaternion_matrix(canonical_quat).T @ np.asarray(
        [0.0, 0.0, visual_world_z_shift],
        dtype=np.float64,
    )
    overlay_object_name = getattr(
        env.case,
        "hai_visual_overlay_object_name",
        None,
    )
    if overlay_object_name is None:
        target_prefix = f"{target.name}_"
        target_collision_ids = set(official.native_target_collision_ids(env))
        target_visual_ids = [
            geom_id
            for geom_id in official._named_ids(model, "geom", target_prefix)
            if geom_id not in target_collision_ids
        ]
        for geom_id in target_visual_ids:
            original_local_quat = normalized_quaternion(model.geom_quat[geom_id])
            model.geom_quat[geom_id][:] = quaternion_multiply(
                relative_quat,
                original_local_quat,
            )
            model.geom_pos[geom_id] += relative_position_local
            model.geom_rgba[geom_id][3] = 1.0
            material_id = int(model.geom_matid[geom_id])
            if material_id >= 0:
                model.mat_rgba[material_id][3] = 1.0
        sim.forward()
        return {
            "overlay_object_name": None,
            "official_asset_xml": str(variant["asset_xml"]),
            "official_visual_geom_ids": [
                int(value) for value in target_visual_ids
            ],
            "disabled_overlay_collision_geom_ids": [],
            "all_overlay_geoms_noncolliding": True,
            "overlay_gravity_compensation": None,
            "relative_position_local_m": relative_position_local.astype(
                float
            ).tolist(),
            "relative_quat_wxyz": relative_quat.astype(float).tolist(),
            "visual_world_z_shift_m": visual_world_z_shift,
            "official_mesh_uv_and_material_preserved": True,
            "uses_classic_target_native_visual_geom": True,
            "follows_classic_physical_target_each_render": True,
        }

    overlay = env.inner_env.get_object(str(overlay_object_name))
    prefix = f"{overlay.name}_"
    geom_ids = official._named_ids(model, "geom", prefix)
    body_ids = [
        body_id
        for body_id in official._named_ids(model, "body", prefix)
        if float(model.body_mass[body_id]) > 0.0
    ]
    if not geom_ids or len(body_ids) != 1:
        raise RuntimeError(
            f"Visual overlay {environment['environment_id']} did not compile correctly"
        )
    visual_geom_ids = []
    disabled_collision_geom_ids = []
    for geom_id in geom_ids:
        was_collision = bool(
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        )
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        if was_collision:
            disabled_collision_geom_ids.append(int(geom_id))
            model.geom_rgba[geom_id][3] = 0.0
        else:
            visual_geom_ids.append(int(geom_id))
    dummy_overlay = bool(
        getattr(env.case, "hai_visual_overlay_is_dummy", False)
    )
    if dummy_overlay:
        for geom_id in geom_ids:
            model.geom_rgba[geom_id][3] = 0.0
            material_id = int(model.geom_matid[geom_id])
            if material_id >= 0:
                model.mat_rgba[material_id][3] = 0.0
        visual_geom_ids = []
    body_id = body_ids[0]
    model.body_gravcomp[body_id] = 1.0
    env._hai_visual_overlay_state = {
        "target_joint_name": target.joints[-1],
        "overlay_joint_name": overlay.joints[-1],
        "relative_position_local_m": relative_position_local.astype(float).tolist(),
        "relative_quat_wxyz": relative_quat.astype(float).tolist(),
    }
    sync_visual_overlay(env)
    return {
        "overlay_object_name": str(overlay.name),
        "overlay_body_id": int(body_id),
        "official_asset_xml": str(variant["asset_xml"]),
        "carrier_bddl_type": "butter" if dummy_overlay else str(variant["bddl_type"]),
        "dummy_visual_carrier": dummy_overlay,
        "official_visual_geom_ids": visual_geom_ids,
        "disabled_overlay_collision_geom_ids": disabled_collision_geom_ids,
        "all_overlay_geoms_noncolliding": True,
        "overlay_gravity_compensation": float(model.body_gravcomp[body_id]),
        "relative_position_local_m": relative_position_local.astype(float).tolist(),
        "relative_quat_wxyz": relative_quat.astype(float).tolist(),
        "visual_world_z_shift_m": visual_world_z_shift,
        "official_mesh_uv_and_material_preserved": True,
        "follows_classic_physical_target_each_render": True,
    }


def normalize_robot_state(
    env: Any,
    *,
    canonical: dict[str, Any],
) -> dict[str, Any]:
    sim = env.inner_env.sim
    robot = env.inner_env.robots[0]
    arm_pos_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    arm_vel_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    gripper_pos_indexes = np.asarray(
        robot._ref_gripper_joint_pos_indexes,
        dtype=np.int64,
    )
    gripper_vel_indexes = np.asarray(
        robot._ref_gripper_joint_vel_indexes,
        dtype=np.int64,
    )
    arm_qpos = np.asarray(canonical["robot_arm_joint_qpos"], dtype=np.float64)
    gripper_qpos = np.asarray(
        canonical["robot_gripper_joint_qpos"],
        dtype=np.float64,
    )
    if len(arm_pos_indexes) != len(arm_qpos):
        raise RuntimeError("Canonical arm qpos length does not match Panda model")
    if len(gripper_pos_indexes) != len(gripper_qpos):
        raise RuntimeError("Canonical gripper qpos length does not match Panda model")
    sim.data.qpos[arm_pos_indexes] = arm_qpos
    sim.data.qvel[arm_vel_indexes] = 0.0
    sim.data.qpos[gripper_pos_indexes] = gripper_qpos
    sim.data.qvel[gripper_vel_indexes] = 0.0
    sim.forward()
    env._last_obs = env._refresh_obs()
    actual_eef = np.asarray(
        env._last_obs["robot0_eef_pos"],
        dtype=np.float64,
    )
    expected_eef = np.asarray(
        canonical["robot_reset_eef_xyz_m"],
        dtype=np.float64,
    )
    if not np.allclose(actual_eef, expected_eef, atol=1e-8, rtol=0.0):
        raise RuntimeError(
            "Canonical robot reset did not reproduce segmented80 EEF: "
            f"actual={actual_eef.tolist()} expected={expected_eef.tolist()}"
        )
    return {
        "arm_joint_qpos": np.asarray(
            sim.data.qpos[arm_pos_indexes],
            dtype=np.float64,
        ).astype(float).tolist(),
        "gripper_joint_qpos": np.asarray(
            sim.data.qpos[gripper_pos_indexes],
            dtype=np.float64,
        ).astype(float).tolist(),
        "arm_joint_qvel": np.asarray(
            sim.data.qvel[arm_vel_indexes],
            dtype=np.float64,
        ).astype(float).tolist(),
        "gripper_joint_qvel": np.asarray(
            sim.data.qvel[gripper_vel_indexes],
            dtype=np.float64,
        ).astype(float).tolist(),
        "reset_eef_xyz_m": actual_eef.astype(float).tolist(),
        "matched_to_segmented80": True,
    }


def install_wand_import_stub() -> None:
    if "wand.api" in sys.modules:
        return

    class UnavailableWandFunction:
        argtypes: tuple[Any, ...] = ()

        def __call__(self, *unused_args, **unused_kwargs):
            raise RuntimeError(
                "Wand motion blur is unavailable in this simulation collector"
            )

    class UnavailableWandImage:
        pass

    wand_package = types.ModuleType("wand")
    wand_package.__path__ = []
    wand_api = types.ModuleType("wand.api")
    wand_api.library = types.SimpleNamespace(
        MagickMotionBlurImage=UnavailableWandFunction()
    )
    wand_image = types.ModuleType("wand.image")
    wand_image.Image = UnavailableWandImage
    wand_package.api = wand_api
    wand_package.image = wand_image
    sys.modules["wand"] = wand_package
    sys.modules["wand.api"] = wand_api
    sys.modules["wand.image"] = wand_image


def apply_active_environment_setup(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    if not _ACTIVE_SETUP:
        return obs
    environment = _ACTIVE_SETUP["environment"]
    preset = _ACTIVE_SETUP["background_preset"]
    experiment = _ACTIVE_SETUP["experiment"]
    seed = int(_ACTIVE_SETUP["seed"])
    background_config = dict(experiment["background_randomization"])
    background_config["presets"] = [preset]
    background = official.legacy.randomize_background(
        env,
        np.random.default_rng(seed),
        background_config,
    )
    physical_target = configure_classic_physical_target(
        env,
        canonical=experiment["canonical_physics"],
        hide_visual=bool(
            getattr(env.case, "hai_visual_overlay_object_name", None)
        )
        and not bool(
            getattr(env.case, "hai_visual_overlay_is_dummy", False)
        ),
    )
    visual_overlay = configure_visual_overlay(
        env,
        environment=environment,
        canonical=experiment["canonical_physics"],
    )
    robot_state = normalize_robot_state(
        env,
        canonical=experiment["canonical_physics"],
    )
    env._last_obs = env._refresh_obs()
    _ACTIVE_RESULT.clear()
    _ACTIVE_RESULT.update(
        {
            "background": background,
            "physical_target": physical_target,
            "visual_overlay": visual_overlay,
            "robot_state": robot_state,
        }
    )
    return env._last_obs


def reset_with_native_style(
    env: Any,
    native_reset: Any,
    preset: dict[str, Any],
) -> dict[str, Any]:
    from libero.libero.envs.arenas.table_arena import TableArena

    original_init = TableArena.__init__
    style_calls: list[dict[str, str]] = []

    def styled_init(arena_self, *arena_args, **arena_kwargs):
        arena_kwargs["xml"] = str(
            official.legacy.LIBERO_ROOT
            / "libero"
            / "assets"
            / preset["arena_xml"]
        )
        arena_kwargs["floor_style"] = str(preset["floor_style"])
        arena_kwargs["wall_style"] = str(preset["wall_style"])
        style_calls.append(
            {
                "xml": str(arena_kwargs["xml"]),
                "floor_style": str(arena_kwargs["floor_style"]),
                "wall_style": str(arena_kwargs["wall_style"]),
                "table_texture_file": str(preset["table_texture_file"]),
            }
        )
        result = original_init(arena_self, *arena_args, **arena_kwargs)
        texture_file = str(
            official.legacy.LIBERO_ROOT
            / "libero"
            / "assets"
            / preset["table_texture_file"]
        )
        for texture_name in ("tex-table", "tex-table-legs"):
            texture = arena_self.asset.find(
                f"./texture[@name='{texture_name}']"
            )
            if texture is not None:
                texture.set("file", texture_file)
        return result

    TableArena.__init__ = styled_init
    try:
        obs = native_reset()
    finally:
        TableArena.__init__ = original_init
    if not style_calls:
        raise RuntimeError("Styled TableArena reset hook was not invoked")
    env._native_background_preset = {
        **preset,
        "arena_constructor_calls": style_calls,
    }
    return obs


def matched_visual_env_factory(
    case: Any,
    *,
    repo_root: Path,
    seed: int,
) -> Any:
    del repo_root
    if not _ACTIVE_SETUP:
        raise RuntimeError("Active visual environment setup is missing")
    install_wand_import_stub()
    preset = dict(_ACTIVE_SETUP["background_preset"])
    official.legacy._NATIVE_PRESETS = [preset]
    env = official.legacy._native_style_env(case, seed=int(seed))
    native_reset = env.reset
    native_refresh_obs = env._refresh_obs

    def matched_refresh_obs() -> dict[str, Any]:
        sync_visual_overlay(env)
        return native_refresh_obs()

    def matched_reset() -> dict[str, Any]:
        obs = reset_with_native_style(env, native_reset, preset)
        obs = apply_active_environment_setup(env, obs)
        # native_reset caches the BDDL sampler's pose before the canonical
        # segmented80 pose is restored. The scripted approach is relative to
        # this cache, so refresh it after normalization to keep the pusher
        # centered on the actual physical collision body.
        env._initial_box_xyz, _ = env.box_pose()
        return obs

    env._refresh_obs = matched_refresh_obs
    env.reset = matched_reset
    return env


segmented.base.LiberoPushBoxEnv = matched_visual_env_factory


def create_dataset(
    root: Path,
    *,
    config: dict[str, Any],
    video_codec: str,
    mode: str,
) -> Any:
    return segmented.base.LeRobotDataset.create(
        repo_id=f"libero_plus_push_box_segmented40_3env_{mode}_hai_machine",
        root=root,
        fps=int(config["fps"]),
        features=segmented.base.build_features(int(config["camera_resolution"])),
        use_videos=True,
        video_codec=video_codec,
        is_compute_episode_stats_image=False,
    )


def build_physics_config(
    experiment: dict[str, Any],
    frictions: list[float],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    config = segmented.configure_current_dataset(
        segmented.base.load_config(segmented.CONFIG_PATH)
    )
    config["frictions"] = [float(value) for value in frictions]
    config["friction_count"] = len(frictions)
    config["actions"] = [dict(value) for value in actions]
    config["action_count"] = len(actions)
    config["camera_resolution"] = int(experiment["camera_resolution"])
    config["fps"] = int(experiment["fps"])
    config["dataset_name"] = (
        "libero_plus_push_box_segmented40_3env_matched_physics_hai-machine"
    )
    return config


def iter_cases(
    *,
    mode: str,
    frictions: list[float],
    actions: list[dict[str, Any]],
    environments: list[dict[str, Any]],
) -> Iterable[tuple[int, float, dict[str, Any], dict[str, Any]]]:
    if mode == "preview":
        for mu_index, mu in enumerate(frictions):
            for action in actions:
                for environment in environments:
                    yield mu_index, float(mu), action, environment
        return
    for environment in environments:
        for mu_index, mu in enumerate(frictions):
            for action in actions:
                yield mu_index, float(mu), action, environment


def find_episode_video(dataset_root: Path, episode_index: int) -> Path:
    matches = list(
        dataset_root.glob(
            "videos/**/observation.images.image/"
            f"episode_{int(episode_index):06d}.mp4"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one agent-view video for episode {episode_index}, got {matches}"
        )
    return matches[0]


def make_preview_videos(
    *,
    output_root: Path,
    dataset_root: Path,
    rows: list[dict[str, Any]],
    environments: list[dict[str, Any]],
    video_crf: int,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create comparison videos")
    individual_root = output_root / "individual_videos"
    comparison_root = output_root / "comparison_videos_vertical"
    individual_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)
    lookup: dict[tuple[int, int, int], Path] = {}
    individual_rows = []
    for row in rows:
        source = find_episode_video(dataset_root, int(row["episode_index"]))
        destination = (
            individual_root
            / (
                f"{row['case_id']}_{row['environment_id']}_"
                f"{row['mu_tag']}_a{int(row['action_id']):02d}_"
                f"A{int(round(float(row['A']) * 1000)):03d}.mp4"
            )
        )
        shutil.copy2(source, destination)
        key = (
            int(row["mu_index"]),
            int(row["action_id"]),
            int(row["environment_index"]),
        )
        lookup[key] = destination
        individual_rows.append(
            {
                "episode_index": int(row["episode_index"]),
                "environment_id": str(row["environment_id"]),
                "mu": float(row["mu"]),
                "action_id": int(row["action_id"]),
                "A": float(row["A"]),
                "video": str(destination),
            }
        )

    comparison_rows = []
    combinations = sorted(
        {
            (int(row["mu_index"]), float(row["mu"]), int(row["action_id"]), float(row["A"]))
            for row in rows
        }
    )
    environment_order = [
        (int(value["environment_index"]), str(value["environment_id"]))
        for value in environments
    ]
    for mu_index, mu, action_id, amplitude in combinations:
        inputs = [
            lookup[(mu_index, action_id, environment_index)]
            for environment_index, _ in environment_order
        ]
        destination = (
            comparison_root
            / (
                f"compare_{segmented.base.mu_tag(mu)}_"
                f"a{action_id:02d}_A{int(round(amplitude * 1000)):03d}_"
                "env00-top_env01-middle_env02-bottom.mp4"
            )
        )
        command = [ffmpeg, "-y", "-loglevel", "error"]
        for source in inputs:
            command.extend(["-i", str(source)])
        command.extend(
            [
                "-filter_complex",
                (
                    "[0:v]scale=224:224,setsar=1[v0];"
                    "[1:v]scale=224:224,setsar=1[v1];"
                    "[2:v]scale=224:224,setsar=1[v2];"
                    "[v0][v1][v2]vstack=inputs=3[out]"
                ),
                "-map",
                "[out]",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                str(int(video_crf)),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ]
        )
        subprocess.run(command, check=True)
        comparison_rows.append(
            {
                "mu_index": mu_index,
                "mu": mu,
                "action_id": action_id,
                "A": amplitude,
                "vertical_order": [
                    {
                        "position": position,
                        "environment_index": environment_index,
                        "environment_id": environment_id,
                    }
                    for position, (environment_index, environment_id) in zip(
                        ["top", "middle", "bottom"],
                        environment_order,
                    )
                ],
                "input_videos": [str(value) for value in inputs],
                "comparison_video": str(destination),
            }
        )
    manifest = {
        "layout": "three 224x224 agent views stacked vertically into 224x672",
        "individual_videos": individual_rows,
        "comparison_videos": comparison_rows,
    }
    write_json(output_root / "comparison_video_manifest.json", manifest)
    return manifest


def collect(
    experiment: dict[str, Any],
    *,
    mode: str,
    output_root: Path,
    overwrite: bool,
    seed: int,
    video_codec: str,
    video_crf: int,
    jpeg_quality: int,
    make_comparisons: bool,
) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    formal_frictions = friction_values(experiment)
    all_actions = [dict(value) for value in experiment["actions"]]
    if mode == "preview":
        frictions = [float(value) for value in experiment["preview"]["frictions"]]
        selected_ids = {int(value) for value in experiment["preview"]["action_ids"]}
        actions = [
            value for value in all_actions if int(value["action_id"]) in selected_ids
        ]
    else:
        frictions = formal_frictions
        actions = all_actions
    environments = sorted(
        [dict(value) for value in experiment["environments"]],
        key=lambda value: int(value["environment_index"]),
    )
    expected = len(frictions) * len(actions) * len(environments)
    declared_expected = int(
        experiment[mode][
            "expected_episode_count"
        ]
    )
    if expected != declared_expected:
        raise RuntimeError(
            f"{mode} episode count mismatch: computed {expected}, declared {declared_expected}"
        )

    presets = {
        str(value["preset_id"]): dict(value)
        for value in experiment["background_randomization"]["presets"]
    }
    physics_config = build_physics_config(experiment, frictions, actions)
    segmented.base.patch_lerobot_video_crf(int(video_crf))
    dataset_root = output_root / "hidden_straight_lerobot"
    dataset = create_dataset(
        dataset_root,
        config=experiment,
        video_codec=video_codec,
        mode=mode,
    )
    rows: list[dict[str, Any]] = []
    generation_metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": (
            f"libero_plus_push_box_segmented40_3env_matched_physics_{mode}_"
            "lerobot_hai-machine"
        ),
        "mode": mode,
        "target_visible": False,
        "split": str(experiment["split"]),
        "camera_resolution": int(experiment["camera_resolution"]),
        "fps": int(experiment["fps"]),
        "video_codec": str(video_codec),
        "video_crf": int(video_crf),
        "jpeg_quality": int(jpeg_quality),
        "state_source": (
            "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to "
            "axis-angle, robot0_gripper_qpos"
        ),
        "action_source": "exact segmented80 environment action conversion",
        "segmented80_control": dict(experiment["segmented80_control"]),
        "formal_friction_values": formal_frictions,
        "active_friction_values": frictions,
        "active_actions": actions,
        "environments": environments,
        "canonical_physics": dict(experiment["canonical_physics"]),
        "episodes": [],
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "mode": mode,
        "output_root": str(output_root),
        "hidden_straight_lerobot": str(dataset_root),
        "config": experiment,
        "episodes": [],
    }

    def autosave() -> None:
        write_json(output_root / "manifest.json", manifest)
        segmented.base.write_dataset_metadata(
            dataset_root,
            generation_metadata,
            rows,
        )

    count = 0
    for mu_index, mu, action_cfg, environment in iter_cases(
        mode=mode,
        frictions=frictions,
        actions=actions,
        environments=environments,
    ):
        environment_index = int(environment["environment_index"])
        action_id = int(action_cfg["action_id"])
        amplitude = float(action_cfg["A"])
        case_id = (
            f"{mode}_e{environment_index:02d}_m{mu_index:02d}_"
            f"{segmented.base.mu_tag(mu)}_a{action_id:02d}_"
            f"A{int(round(amplitude * 1000)):03d}_"
            f"n{int(action_cfg['push_steps']):02d}"
        )
        base_bddl = segmented.base.write_hidden_bddl(
            physics_config,
            bddl_dir=output_root / "bddl",
            geometry_id=case_id,
        )
        case = segmented.base.build_fixed_case(
            physics_config,
            mu=mu,
            action_cfg=action_cfg,
            case_id=case_id,
            bddl_file=base_bddl,
            camera_resolution=int(experiment["camera_resolution"]),
        )
        rollout_seed = int(seed)
        asset_case = build_visual_overlay_case(
            case,
            environment=environment,
        )
        preset_id = str(environment["background_preset_id"])
        _ACTIVE_SETUP.clear()
        _ACTIVE_SETUP.update(
            {
                "environment": environment,
                "background_preset": presets[preset_id],
                "experiment": experiment,
                "seed": rollout_seed,
            }
        )
        _ACTIVE_RESULT.clear()
        episode_index, metrics = segmented.rollout_to_lerobot_event_tap(
            asset_case,
            dataset=dataset,
            seed=rollout_seed,
            fps=int(experiment["fps"]),
            jpeg_quality=int(jpeg_quality),
        )
        if not _ACTIVE_RESULT:
            raise RuntimeError(f"Environment setup result missing for {case_id}")
        profile = segmented.base.profile_for_steps(
            int(action_cfg["push_steps"])
        ).astype(float).tolist()
        row = {
            "episode_index": int(episode_index),
            "case_id": case_id,
            "mode": mode,
            "environment_index": environment_index,
            "environment_id": str(environment["environment_id"]),
            "source_preview_case": str(environment["source_preview_case"]),
            "background_preset_id": preset_id,
            "object_variant": dict(environment["object_variant"]),
            "environment_setup": dict(_ACTIVE_RESULT),
            "mu_index": int(mu_index),
            "mu": mu,
            "mu_tag": segmented.base.mu_tag(mu),
            "action_id": action_id,
            "A": amplitude,
            "push_steps": int(action_cfg["push_steps"]),
            "profile": profile,
            "profile_area": float(sum(profile) * amplitude),
            "init_xy": [
                float(value) for value in physics_config["init_xy"]
            ],
            "target_xy": list(
                segmented.base.fixed_scene_target_xy(physics_config)
            ),
            "bddl_file": str(asset_case.bddl_file),
            "event_tap": dict(physics_config["event_tap"]),
            "rollout_seed": rollout_seed,
            "metrics": metrics,
        }
        rows.append(row)
        generation_metadata["episodes"].append(row)
        manifest["episodes"].append(row)
        count += 1
        print(
            f"collect {count:04d}/{expected:04d} {case_id} "
            f"env={environment['environment_id']} "
            f"disp={metrics['final_displacement_m'] * 100.0:.2f}cm "
            f"peak_vx={metrics.get('peak_vx')} "
            f"contact={metrics.get('contact_local')}",
            flush=True,
        )
        autosave()

    comparison_manifest = None
    if mode == "preview" and make_comparisons:
        comparison_manifest = make_preview_videos(
            output_root=output_root,
            dataset_root=dataset_root,
            rows=rows,
            environments=environments,
            video_crf=video_crf,
        )

    count_by_environment = Counter(str(row["environment_id"]) for row in rows)
    count_by_mu = Counter(str(row["mu_tag"]) for row in rows)
    count_by_action = Counter(f"a{int(row['action_id']):02d}" for row in rows)
    summary = {
        "mode": mode,
        "episode_count": len(rows),
        "expected_episode_count": expected,
        "hidden_straight_lerobot": str(dataset_root),
        "count_by_environment": dict(sorted(count_by_environment.items())),
        "count_by_mu": dict(sorted(count_by_mu.items())),
        "count_by_action": dict(sorted(count_by_action.items())),
        "comparison_video_count": (
            0
            if comparison_manifest is None
            else len(comparison_manifest["comparison_videos"])
        ),
        "formal_friction_values": formal_frictions,
        "formal_friction_segment_counts": [
            int(value["count"])
            for value in experiment["friction_schedule"]["segments"]
        ],
    }
    write_json(output_root / "summary.json", summary)
    autosave()
    print(json.dumps(segmented.base.to_jsonable(summary), indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    experiment = json.loads(config_path.read_text(encoding="utf-8"))
    recording = experiment["recording"]
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (
            DEFAULT_PREVIEW_OUTPUT
            if args.mode == "preview"
            else DEFAULT_FORMAL_OUTPUT
        )
    )
    collect(
        experiment,
        mode=str(args.mode),
        output_root=output_root,
        overwrite=bool(args.overwrite),
        seed=int(args.seed),
        video_codec=str(args.video_codec or recording["video_codec"]),
        video_crf=int(
            args.video_crf
            if args.video_crf is not None
            else recording["video_crf"]
        ),
        jpeg_quality=int(
            args.jpeg_quality
            if args.jpeg_quality is not None
            else recording["jpeg_quality"]
        ),
        make_comparisons=not bool(args.skip_comparison_videos),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import copy
import datetime as dt
from dataclasses import replace
import importlib.util
import importlib.machinery
import json
import math
import os
from pathlib import Path
import shutil
import sys
import types
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT.parent / "LIBERO-plus" / "libero"
LIBERO_CONFIG_ROOT = REPO_ROOT / "configs" / "libero_plus_runtime_2026-07-18_hai-machine"
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_ROOT)

# Both LIBERO variants expose the same namespace and the shared venv keeps the
# vanilla repository editable-installed. Pin this process to the Plus source
# tree before any transitive collector import can resolve ``libero.libero``.
libero_namespace = types.ModuleType("libero")
libero_namespace.__path__ = [str(LIBERO_ROOT)]
libero_namespace.__package__ = "libero"
libero_namespace.__spec__ = importlib.machinery.ModuleSpec(
    "libero", loader=None, is_package=True
)
libero_namespace.__spec__.submodule_search_locations = [str(LIBERO_ROOT)]
sys.modules["libero"] = libero_namespace
FORMAL_SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_board_touch_20fric_30action_fixed5cm_A050_lerobot_2026-07-17_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_plus_push_box_native_gripper_backgrounds_9demo_2026-07-18_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "libero_plus_native_gripper_backgrounds_preview"
    / "libero_plus_push_box_native_gripper_backgrounds_9eps_lerobot_2026-07-18_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


formal = load_module(FORMAL_SOURCE_SCRIPT, "randomized_scene_preview_formal_source_hai_machine")
ramp = formal.ramp
collector = formal.collector
base = formal.base
touch = ramp.touch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect randomized-scene directional PushBox preview episodes.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    value = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def yaw_quaternion(angle_rad: float) -> np.ndarray:
    return np.asarray([math.cos(angle_rad / 2.0), 0.0, 0.0, math.sin(angle_rad / 2.0)])


def hsv_color(rng: np.random.Generator, *, saturation: tuple[float, float], value: tuple[float, float]) -> np.ndarray:
    rgb = colorsys.hsv_to_rgb(
        float(rng.uniform(0.0, 1.0)),
        float(rng.uniform(*saturation)),
        float(rng.uniform(*value)),
    )
    return np.asarray(rgb, dtype=np.float64)


def material_id(model: Any, material_name: str) -> int:
    native = model._model
    for index in range(int(model.nmat)):
        name = mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_MATERIAL, index)
        if name == material_name:
            return int(index)
    raise KeyError(material_name)


def randomize_background(env: Any, rng: np.random.Generator, cfg: dict[str, Any]) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    colors = {
        "floorplane": hsv_color(rng, saturation=(0.15, 0.75), value=(0.30, 0.80)),
        "table_texture": hsv_color(rng, saturation=(0.20, 0.80), value=(0.35, 0.85)),
        "table_legs": hsv_color(rng, saturation=(0.10, 0.65), value=(0.18, 0.65)),
        "walls_mat": hsv_color(rng, saturation=(0.15, 0.75), value=(0.35, 0.92)),
    }
    texture_state: dict[str, str] = {}
    for name, rgb in colors.items():
        index = material_id(model, name)
        model.mat_rgba[index] = np.asarray([*rgb, 1.0], dtype=np.float64)
        if rng.random() > float(cfg["retain_texture_probability"]):
            model.mat_texid[index] = -1
            texture_state[name] = "disabled"
        else:
            texture_state[name] = "retained"

    for light_id in range(int(model.nlight)):
        intensity = float(rng.uniform(*cfg["light_diffuse_range"]))
        tint = hsv_color(rng, saturation=(0.0, 0.18), value=(0.90, 1.0))
        model.light_diffuse[light_id] = intensity * tint
        ambient = float(rng.uniform(*cfg["light_ambient_range"]))
        model.light_ambient[light_id] = ambient * tint
        model.light_specular[light_id] = float(rng.uniform(0.10, 0.45)) * tint
        model.light_pos[light_id, :2] += rng.uniform(
            -float(cfg["light_xy_jitter_m"]),
            float(cfg["light_xy_jitter_m"]),
            size=2,
        )
    sim.forward()
    return {
        "material_rgb": {name: value.astype(float).tolist() for name, value in colors.items()},
        "texture_state": texture_state,
        "lights": [
            {
                "position_m": model.light_pos[index].astype(float).tolist(),
                "diffuse": model.light_diffuse[index].astype(float).tolist(),
                "ambient": model.light_ambient[index].astype(float).tolist(),
            }
            for index in range(int(model.nlight))
        ],
    }


def set_object_variant_and_pose(
    env: Any,
    *,
    variant: dict[str, Any],
    rng: np.random.Generator,
    base_init_xy: np.ndarray,
    position_cfg: dict[str, Any],
    table_rgb: np.ndarray,
    minimum_contrast: float,
) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    visual_id = model.geom_name2id("cream_cheese_1_g0")
    collision_id = model.geom_name2id("cream_cheese_1_g1")
    original_size = np.asarray(model.geom_size[collision_id], dtype=np.float64).copy()
    collision_rotation = np.asarray(sim.data.geom_xmat[collision_id], dtype=np.float64).reshape(3, 3).copy()
    size = np.asarray(variant["collision_half_size_local_m"], dtype=np.float64)
    world_up_local = collision_rotation.T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    original_vertical_support = float(np.sum(np.abs(world_up_local) * original_size))
    new_vertical_support = float(np.sum(np.abs(world_up_local) * size))
    model.geom_type[visual_id] = int(mujoco.mjtGeom.mjGEOM_BOX)
    model.geom_type[collision_id] = int(mujoco.mjtGeom.mjGEOM_BOX)
    model.geom_size[visual_id] = size
    model.geom_size[collision_id] = size
    model.geom_pos[visual_id] = model.geom_pos[collision_id]
    model.geom_quat[visual_id] = model.geom_quat[collision_id]
    model.geom_matid[visual_id] = -1
    object_rgb = hsv_color(rng, saturation=(0.45, 0.95), value=(0.45, 0.95))
    for _ in range(30):
        if float(np.linalg.norm(object_rgb - table_rgb)) >= minimum_contrast:
            break
        object_rgb = hsv_color(rng, saturation=(0.45, 0.95), value=(0.45, 0.95))
    model.geom_rgba[visual_id] = np.asarray([*object_rgb, 1.0], dtype=np.float64)
    model.geom_rgba[collision_id, 3] = 0.0

    obj = env.inner_env.get_object(env.case.box_name)
    joint_name = obj.joints[-1]
    qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    qpos[0] = float(base_init_xy[0] + rng.uniform(*position_cfg["front_back_x_jitter_m"]))
    qpos[1] = float(base_init_xy[1] + rng.uniform(*position_cfg["horizontal_y_jitter_m"]))
    qpos[2] += new_vertical_support - original_vertical_support
    sim.data.set_joint_qpos(joint_name, qpos)
    sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()
    return {
        "object_id": int(variant["object_id"]),
        "name": str(variant["name"]),
        "collision_half_size_local_m": size.astype(float).tolist(),
        "visual_rgb": object_rgb.astype(float).tolist(),
        "requested_initial_xyz_m": qpos[:3].astype(float).tolist(),
        "vertical_support_adjustment_m": float(new_vertical_support - original_vertical_support),
    }


def board_normal(env: Any) -> np.ndarray:
    sim = env.inner_env.sim
    geom_id = sim.model.geom_name2id("gripper0_board_tool_collision")
    rotation = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    normal = rotation[:, 1].copy()
    normal[2] = 0.0
    return normal / np.linalg.norm(normal)


def configure_directional_board(env: Any, direction: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    body_id = model.body_name2id("gripper0_board_tool")
    collision_id = model.geom_name2id("gripper0_board_tool_collision")
    visual_id = model.geom_name2id("gripper0_board_tool_visual")
    handle_id = model.geom_name2id("gripper0_board_handle_visual")
    model.body_pos[body_id] = np.asarray(cfg["board_body_position_eef_m"], dtype=np.float64)
    model.geom_size[collision_id] = np.asarray(cfg["board_half_size_m"], dtype=np.float64)
    model.geom_size[visual_id] = np.asarray(cfg["board_half_size_m"], dtype=np.float64)
    model.geom_margin[collision_id] = float(cfg["contact_margin_m"])
    if bool(cfg["hide_handle_visual"]):
        model.geom_rgba[handle_id, 3] = 0.0
    if bool(cfg["disable_non_board_gripper_collision"]):
        for geom_id in range(int(model.ngeom)):
            name = model.geom_id2name(geom_id) or ""
            if name.startswith("gripper0_") and "collision" in name and geom_id != collision_id:
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0

    default_quat = np.asarray(touch.board_probe.BOARD_QUAT_EEF_WXYZ, dtype=np.float64)
    angle = math.atan2(float(direction[1]), float(direction[0]))
    candidates = []
    for sign in (-1.0, 1.0):
        yaw = yaw_quaternion(sign * angle)
        candidates.extend([quaternion_multiply(yaw, default_quat), quaternion_multiply(default_quat, yaw)])
    best_quat = default_quat
    best_alignment = -1.0
    best_normal = None
    for candidate in candidates:
        model.body_quat[body_id] = candidate
        sim.forward()
        normal = board_normal(env)
        alignment = abs(float(np.dot(normal[:2], direction)))
        if alignment > best_alignment:
            best_alignment = alignment
            best_quat = candidate.copy()
            best_normal = normal.copy()
    model.body_quat[body_id] = best_quat
    sim.forward()
    final_normal = board_normal(env)
    error = math.degrees(math.acos(np.clip(abs(float(np.dot(final_normal[:2], direction))), -1.0, 1.0)))
    return {
        "body_position_eef_m": model.body_pos[body_id].astype(float).tolist(),
        "body_quaternion_eef_wxyz": best_quat.astype(float).tolist(),
        "contact_margin_m": float(model.geom_margin[collision_id]),
        "measured_normal_xy": final_normal[:2].astype(float).tolist(),
        "normal_alignment_error_deg": float(error),
        "candidate_normal_xy": None if best_normal is None else best_normal[:2].astype(float).tolist(),
    }


def box_and_board_geometry(env: Any, direction: np.ndarray) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    box_id = model.geom_name2id("cream_cheese_1_g1")
    board_id = model.geom_name2id("gripper0_board_tool_collision")
    box_center = np.asarray(sim.data.geom_xpos[box_id], dtype=np.float64)
    board_center = np.asarray(sim.data.geom_xpos[board_id], dtype=np.float64)
    box_rotation = np.asarray(sim.data.geom_xmat[box_id], dtype=np.float64).reshape(3, 3)
    board_rotation = np.asarray(sim.data.geom_xmat[board_id], dtype=np.float64).reshape(3, 3)
    direction3 = np.asarray([direction[0], direction[1], 0.0], dtype=np.float64)
    box_support = float(np.sum(np.abs(box_rotation.T @ direction3) * model.geom_size[box_id]))
    board_support = float(np.sum(np.abs(board_rotation.T @ direction3) * model.geom_size[board_id]))
    gap = float(np.dot(box_center - board_center, direction3) - box_support - board_support)
    return {
        "box_center": box_center,
        "board_center": board_center,
        "box_support_m": box_support,
        "board_support_m": board_support,
        "gap_m": gap,
    }


def board_box_contact(env: Any) -> bool:
    return bool(touch.board_box_contacts(env))


def move_eef(env: Any, target_xyz: np.ndarray, *, steps: int, max_action: float, gain: float) -> None:
    for _ in range(int(steps)):
        env.step_count = 0
        eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(float(gain) * (target_xyz - eef), -float(max_action), float(max_action))
        action[-1] = 1.0
        env.step(action)


def prepare_directional_touch(
    env: Any,
    *,
    direction: np.ndarray,
    lateral_offset_m: float,
) -> dict[str, Any]:
    eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
    geometry = box_and_board_geometry(env, direction)
    board_offset = geometry["board_center"] - eef
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    desired_board_xy = (
        geometry["box_center"][:2]
        - direction * (geometry["box_support_m"] + geometry["board_support_m"] + 0.012)
        + lateral * float(lateral_offset_m)
    )
    desired_eef = np.asarray(
        [
            desired_board_xy[0] - board_offset[0],
            desired_board_xy[1] - board_offset[1],
            geometry["box_center"][2] - board_offset[2],
        ],
        dtype=np.float64,
    )
    above = desired_eef.copy()
    above[2] += 0.10
    move_eef(env, above, steps=38, max_action=0.16, gain=4.0)

    geometry = box_and_board_geometry(env, direction)
    eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
    board_offset = geometry["board_center"] - eef
    desired_board_xy = (
        geometry["box_center"][:2]
        - direction * (geometry["box_support_m"] + geometry["board_support_m"] + 0.008)
        + lateral * float(lateral_offset_m)
    )
    desired_eef = np.asarray(
        [
            desired_board_xy[0] - board_offset[0],
            desired_board_xy[1] - board_offset[1],
            geometry["box_center"][2] - board_offset[2],
        ],
        dtype=np.float64,
    )
    move_eef(env, desired_eef, steps=75, max_action=0.24, gain=6.0)

    touch_geometry = box_and_board_geometry(env, direction)
    touch_target = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64).copy()
    touch_start_eef = touch_target.copy()
    touch_start_box = np.asarray(touch_geometry["box_center"], dtype=np.float64).copy()
    touch_target[:2] += direction * (max(0.0, float(touch_geometry["gap_m"])) + 0.001)
    touch_steps = 0
    touch_trigger = ""
    projected_box_motion_m = 0.0
    projected_box_speed_mps = 0.0
    for touch_steps in range(1, 201):
        if board_box_contact(env):
            touch_trigger = "contact_before_step"
            break
        eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(20.0 * (touch_target - eef), -0.20, 0.20)
        action[-1] = 1.0
        env.step_count = 0
        env.step(action)
        after_step_geometry = box_and_board_geometry(env, direction)
        projected_box_motion_m = float(
            np.dot(
                np.asarray(after_step_geometry["box_center"][:2], dtype=np.float64)
                - touch_start_box[:2],
                direction,
            )
        )
        projected_box_speed_mps = float(np.dot(env.box_velocity()[:2], direction))
        if board_box_contact(env):
            touch_trigger = "contact_after_step"
            break
        if projected_box_motion_m >= 0.0001:
            touch_trigger = "box_motion"
            break
        if projected_box_speed_mps >= 0.002:
            touch_trigger = "box_speed"
            break
    if not touch_trigger:
        final_geometry = box_and_board_geometry(env, direction)
        center_delta = final_geometry["box_center"] - final_geometry["board_center"]
        sim = env.inner_env.sim
        raw_contacts = []
        for contact_index in range(int(sim.data.ncon)):
            contact = sim.data.contact[contact_index]
            raw_contacts.append(
                {
                    "geom1": sim.model.geom_id2name(int(contact.geom1)),
                    "geom2": sim.model.geom_id2name(int(contact.geom2)),
                    "distance_m": float(contact.dist),
                }
            )
        raise RuntimeError(
            "Directional touch failed after "
            f"{touch_steps} steps; gap={final_geometry['gap_m']}; "
            f"center_delta={center_delta.astype(float).tolist()}; "
            f"perpendicular_delta={float(np.dot(center_delta[:2], lateral))}; "
            f"touch_start_eef={touch_start_eef.astype(float).tolist()}; "
            f"touch_target={touch_target.astype(float).tolist()}; "
            f"touch_final_eef={np.asarray(env._last_obs['robot0_eef_pos'], dtype=float).tolist()}; "
            f"board_contype={int(env.inner_env.sim.model.geom_contype[env.inner_env.sim.model.geom_name2id('gripper0_board_tool_collision')])}; "
            f"box_conaffinity={int(env.inner_env.sim.model.geom_conaffinity[env.inner_env.sim.model.geom_name2id('cream_cheese_1_g1')])}; "
            f"raw_contacts={raw_contacts}"
        )

    # A board-box collision can begin and end inside one MuJoCo control step, so
    # ncon alone is not a reliable event trigger. Once the first physical response
    # is observed, place the box back at a zero-speed, 20-micron-overlap touch pose.
    # This correction is sub-millimetre and happens before any recorded action.
    sim = env.inner_env.sim
    box_body_id = sim.model.body_name2id("cream_cheese_1_main")
    box_joint_id = int(sim.model.body_jntadr[box_body_id])
    box_qpos_adr = int(sim.model.jnt_qposadr[box_joint_id])
    box_dof_adr = int(sim.model.jnt_dofadr[box_joint_id])
    pre_latch_geometry = box_and_board_geometry(env, direction)
    target_gap_m = -0.00002
    correction_xy = direction * (target_gap_m - float(pre_latch_geometry["gap_m"]))
    sim.data.qpos[box_qpos_adr : box_qpos_adr + 2] += correction_xy
    sim.data.qvel[box_dof_adr : box_dof_adr + 6] = 0.0
    sim.forward()
    latched_geometry = box_and_board_geometry(env, direction)
    if abs(float(latched_geometry["gap_m"]) - target_gap_m) > 0.0001:
        raise RuntimeError(
            "Directional touch latch failed; "
            f"trigger={touch_trigger}; gap={latched_geometry['gap_m']}; "
            f"target_gap={target_gap_m}"
        )
    return {
        "touch_steps": int(touch_steps),
        "touch_trigger": touch_trigger,
        "projected_box_motion_before_latch_m": float(projected_box_motion_m),
        "projected_box_speed_before_latch_mps": float(projected_box_speed_mps),
        "geometry": {
            key: value.astype(float).tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in latched_geometry.items()
        },
        "launch_box_speed_mps": float(np.linalg.norm(env.box_velocity()[:2])),
    }


def absolute_action(observation_state: np.ndarray, env_action: np.ndarray, scale_m: float) -> np.ndarray:
    relative = np.asarray(base._env_action_to_fastwam_action(env_action.astype(np.float32)), dtype=np.float32)
    value = relative.copy()
    value[:3] = np.asarray(observation_state[:3], dtype=np.float32) + np.float32(scale_m) * relative[:3]
    return value


def rollout_preview(
    case: Any,
    *,
    dataset: Any,
    action_cfg: dict[str, Any],
    variant: dict[str, Any],
    experiment: dict[str, Any],
    seed: int,
) -> tuple[int, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    env = touch.make_env(case, seed=int(seed))
    controller = experiment["controller"]
    direction_angle = math.radians(float(action_cfg["angle_deg"]))
    direction = np.asarray([math.cos(direction_angle), math.sin(direction_angle)], dtype=np.float64)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    contact_frames: list[int] = []
    contact_episode_count = 0
    contact_active = False
    phase = "drive"
    brake_frames = 0
    brake_trigger_frame = None
    try:
        background = randomize_background(env, rng, experiment["background_randomization"])
        object_state = set_object_variant_and_pose(
            env,
            variant=variant,
            rng=rng,
            base_init_xy=np.asarray(formal.base.fixed_scene_target_xy({**formal.configure_dataset(json.loads(formal.CONFIG_PATH.read_text(encoding='utf-8'))), "dummy_target_distance": 0.0}), dtype=np.float64),
            position_cfg=experiment["initial_position"],
            table_rgb=np.asarray(background["material_rgb"]["table_texture"], dtype=np.float64),
            minimum_contrast=float(experiment["background_randomization"]["minimum_object_table_rgb_distance"]),
        )
        board_state = configure_directional_board(env, direction, experiment["pusher"])
        base.preposition_fixed_start(env)
        settled_box_xyz, _ = env.box_pose()
        touch_state = prepare_directional_touch(
            env,
            direction=direction,
            lateral_offset_m=float(action_cfg["contact_lateral_offset_m"]),
        )
        launch_box_xyz, _ = env.box_pose()
        launch_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64).copy()
        previous_eef = launch_eef.copy()
        hold_z = float(launch_eef[2])
        hold_perpendicular = float(np.dot(launch_eef[:2], lateral))
        amplitude = float(action_cfg["A"])
        target_projected_travel_m = float(
            action_cfg.get("target_projected_travel_m", controller["target_projected_travel_m"])
        )

        base.remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        task = base.prompt_for_case("observation", str(action_cfg["kind"]))
        for frame_index in range(int(controller["recorded_steps"])):
            obs_for_frame = ramp.copy_obs(env._last_obs)
            eef_before = np.asarray(obs_for_frame["robot0_eef_pos"], dtype=np.float64)
            velocity = (eef_before - previous_eef) * float(experiment["fps"]) if frame_index else np.zeros(3)
            previous_eef = eef_before.copy()
            projected_travel = float(np.dot(eef_before[:2] - launch_eef[:2], direction))
            projected_v = float(np.dot(velocity[:2], direction))
            remaining = float(target_projected_travel_m - projected_travel)

            if phase == "drive" and frame_index > 0:
                lookahead = float(controller["brake_trigger_lookahead_frames"]) * max(0.0, projected_v) / float(experiment["fps"])
                if remaining <= lookahead:
                    phase = "brake"
                    brake_trigger_frame = int(frame_index)
            if phase == "drive":
                command = amplitude * float(controller["first_frame_fraction"]) if frame_index == 0 else amplitude
            elif phase == "brake":
                if brake_frames >= int(controller["maximum_brake_frames"]) or (
                    brake_frames > 0 and projected_v <= float(controller["stop_speed_mps"])
                ):
                    phase = "locked_zero"
                    command = 0.0
                else:
                    command = float(np.clip(-float(controller["brake_gain_action_per_mps"]) * max(0.0, projected_v), -amplitude, 0.0))
                    brake_frames += 1
            else:
                command = 0.0

            action = np.zeros(7, dtype=np.float64)
            if phase != "locked_zero":
                perpendicular_error = hold_perpendicular - float(np.dot(eef_before[:2], lateral))
                hold_limit = float(controller["hold_max_action"])
                perpendicular_action = float(np.clip(float(controller["perpendicular_hold_gain"]) * perpendicular_error, -hold_limit, hold_limit))
                action[:2] = command * direction + perpendicular_action * lateral
                action[2] = float(np.clip(float(controller["height_hold_gain"]) * (hold_z - eef_before[2]), -hold_limit, hold_limit))
            action[:3] = np.clip(action[:3], -float(controller["pusher_max_pos_action"]), float(controller["pusher_max_pos_action"]))
            action[-1] = 1.0
            state = np.asarray(base._obs_to_state(obs_for_frame), dtype=np.float32)
            stored_action = absolute_action(
                state,
                action,
                float(experiment["absolute_action"]["translation_scale_m_per_normalized_unit"]),
            )
            push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
            env.step_count = push_start + min(frame_index, 15)
            _, _, _, info = env.step(action)
            row = dict(info["push_box"])
            row.update(
                {
                    "frame_index": int(frame_index),
                    "phase": phase,
                    "projected_eef_travel_before_m": projected_travel,
                    "projected_eef_velocity_before_mps": projected_v,
                    "remaining_before_m": remaining,
                    "target_projected_travel_m": target_projected_travel_m,
                    "command_along_direction": command,
                    "eef_xyz_before_m": eef_before.astype(float).tolist(),
                }
            )
            rows.append(row)
            has_contact = board_box_contact(env)
            if has_contact:
                contact_frames.append(int(frame_index))
                if not contact_active:
                    contact_episode_count += 1
            contact_active = has_contact

            agent, wrist = base._obs_to_images(obs_for_frame)
            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": state,
                "action": stored_action,
            }
            dataset.add_frame(frame, task=task, timestamp=float(frame_index) / float(experiment["fps"]))
            base.write_image_for_last_frame(dataset, "observation.images.image", frame_index, agent, jpeg_quality=int(experiment["recording"]["jpeg_quality"]))
            base.write_image_for_last_frame(dataset, "observation.images.wrist_image", frame_index, wrist, jpeg_quality=int(experiment["recording"]["jpeg_quality"]))

        final_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64).copy()
        final_box_xyz, _ = env.box_pose()
        dataset.save_episode()
    finally:
        env.close()

    eef_positions = np.asarray([row["eef_xyz_before_m"] for row in rows] + [final_eef.astype(float).tolist()], dtype=np.float64)
    eef_delta = eef_positions[:, :2] - launch_eef[:2]
    projected = eef_delta @ direction
    perpendicular = eef_delta @ lateral
    box_delta = np.asarray(final_box_xyz[:2]) - np.asarray(launch_box_xyz[:2])
    box_velocity = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    metrics = {
        "friction_mu": float(case.friction_mu),
        "action_id": int(action_cfg["action_id"]),
        "kind": str(action_cfg["kind"]),
        "angle_deg": float(action_cfg["angle_deg"]),
        "A": float(action_cfg["A"]),
        "target_projected_travel_m": target_projected_travel_m,
        "direction_xy": direction.astype(float).tolist(),
        "settled_box_xyz_m": np.asarray(settled_box_xyz, dtype=float).tolist(),
        "launch_box_xyz_m": np.asarray(launch_box_xyz, dtype=float).tolist(),
        "final_box_xyz_m": np.asarray(final_box_xyz, dtype=float).tolist(),
        "box_projected_displacement_m": float(np.dot(box_delta, direction)),
        "box_perpendicular_displacement_m": float(np.dot(box_delta, lateral)),
        "peak_box_projected_velocity_mps": float(np.max(box_velocity @ direction)),
        "maximum_projected_eef_travel_m": float(np.max(projected)),
        "final_projected_eef_travel_m": float(np.dot(final_eef[:2] - launch_eef[:2], direction)),
        "maximum_absolute_perpendicular_eef_travel_m": float(np.max(np.abs(perpendicular))),
        "brake_trigger_frame": brake_trigger_frame,
        "brake_frames": int(brake_frames),
        "contact_frames": contact_frames,
        "contact_episode_count": int(contact_episode_count),
        "background": background,
        "object": object_state,
        "board": board_state,
        "touch_preparation": touch_state,
    }
    gates = experiment["quality_gates"]
    checks = {
        "projected_travel": bool(
            float(gates["minimum_projected_eef_travel_m"])
            <= metrics["maximum_projected_eef_travel_m"]
            <= float(gates["maximum_projected_eef_travel_m"])
        ),
        "perpendicular_drift": bool(
            metrics["maximum_absolute_perpendicular_eef_travel_m"]
            <= float(gates["maximum_perpendicular_eef_travel_m"])
        ),
        "board_alignment": bool(
            float(board_state["normal_alignment_error_deg"])
            <= float(gates["maximum_board_normal_error_deg"])
        ),
        "touch_reached": bool(touch_state["touch_steps"] <= 200),
    }
    metrics["quality_checks"] = checks
    metrics["quality_pass"] = bool(all(checks.values()))
    return episode_index, metrics


def main() -> None:
    args = parse_args()
    experiment = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_experiment = json.loads(formal.CONFIG_PATH.read_text(encoding="utf-8"))
    source_config = formal.configure_dataset(source_experiment)
    actions = {int(row["action_id"]): row for row in experiment["actions"]}
    variants = {int(row["object_id"]): row for row in experiment["object_variants"]}
    preview_cases = list(experiment["preview_cases"])
    expected = int(experiment["expected_preview_episode_count"])
    if len(preview_cases) != expected:
        raise RuntimeError(f"Configured {len(preview_cases)} preview cases, expected {expected}")

    base.patch_lerobot_video_crf(int(experiment["recording"]["video_crf"]))
    features = copy.deepcopy(base.build_features(int(experiment["camera_resolution"])))
    features["action"]["names"] = list(experiment["absolute_action"]["names"])
    dataset_root = output_root / "hidden_multidirection_lerobot"
    dataset = base.LeRobotDataset.create(
        repo_id="libero_push_box_libero_plus_native_gripper_backgrounds_preview_hai_machine",
        root=dataset_root,
        fps=int(experiment["fps"]),
        features=features,
        use_videos=True,
        video_codec=str(experiment["recording"]["video_codec"]),
        is_compute_episode_stats_image=False,
    )

    created_at = dt.datetime.now().isoformat()
    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": created_at,
        "dataset_type": "libero_push_box_randomized_scene_preview_lerobot_2026-07-18_hai-machine",
        "purpose": "preview only for a proposed 25-friction x 20-action randomized-scene dataset",
        "formal_collection_started": False,
        "experiment_config": experiment,
        "episodes": [],
    }
    manifest = {
        "created_at": created_at,
        "config_path": str(args.config.resolve()),
        "output_root": str(output_root),
        "lerobot_root": str(dataset_root),
        "episodes": [],
    }

    def autosave() -> None:
        write_json(output_root / "manifest.json", manifest)
        base.write_dataset_metadata(dataset_root, metadata, rows)

    for preview_index, preview in enumerate(preview_cases):
        action_cfg = actions[int(preview["action_id"])]
        variant = variants[int(preview["object_id"])]
        mu = float(preview["friction_mu"])
        case_id = (
            f"preview_{preview_index:02d}_{variant['name']}_mu{int(round(mu * 10000)):04d}_"
            f"a{int(action_cfg['action_id']):02d}_{int(round(float(action_cfg['angle_deg']))):+04d}deg"
        )
        source_action_cfg = {
            "action_id": int(action_cfg["action_id"]),
            "A": float(action_cfg["A"]),
            "push_steps": 16,
        }
        bddl = base.write_hidden_bddl(source_config, bddl_dir=output_root / "bddl", geometry_id=case_id)
        base_case = collector.make_case(
            source_config,
            mu=mu,
            action_cfg=source_action_cfg,
            case_id=case_id,
            bddl_file=bddl,
        )
        case = ramp.preserve_case_attributes(
            base_case,
            replace(base_case, pusher_max_pos_action=float(experiment["controller"]["pusher_max_pos_action"])),
        )
        episode_index, metrics = rollout_preview(
            case,
            dataset=dataset,
            action_cfg=action_cfg,
            variant=variant,
            experiment=experiment,
            seed=int(preview["seed"]),
        )
        if not bool(metrics["quality_pass"]):
            raise RuntimeError(
                f"Preview quality gate failed for {case_id}: "
                f"{metrics['quality_checks']}"
            )
        row = {
            "episode_index": int(episode_index),
            "preview_index": int(preview_index),
            "case_id": case_id,
            "seed": int(preview["seed"]),
            "mu": mu,
            "action": action_cfg,
            "object_variant": variant,
            "metrics": metrics,
        }
        rows.append(row)
        metadata["episodes"].append(row)
        manifest["episodes"].append(row)
        autosave()
        print(
            f"preview {preview_index + 1:02d}/{expected:02d} object={variant['name']} "
            f"mu={mu:.3f} action={action_cfg['action_id']} angle={action_cfg['angle_deg']:+.0f} "
            f"A={action_cfg['A']:.2f} eef={metrics['maximum_projected_eef_travel_m'] * 100:.2f}cm "
            f"box={metrics['box_projected_displacement_m'] * 100:.2f}cm "
            f"quality={metrics['quality_pass']}",
            flush=True,
        )

    preview_video_root = output_root / "preview_videos"
    preview_video_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        episode_index = int(row["episode_index"])
        action = row["action"]
        variant = row["object_variant"]
        source_video = dataset_root / "videos" / "chunk-000" / "observation.images.image" / f"episode_{episode_index:06d}.mp4"
        output_video = preview_video_root / (
            f"ep{episode_index:02d}_{variant['name']}_mu{int(round(float(row['mu']) * 10000)):04d}_"
            f"a{int(action['action_id']):02d}_{int(round(float(action['angle_deg']))):+04d}deg_A{int(round(float(action['A']) * 1000)):03d}.mp4"
        )
        if output_video.exists():
            output_video.unlink()
        os.link(source_video, output_video)
        row["preview_video"] = str(output_video)

    summary = {
        "experiment": experiment["experiment"],
        "episode_count": len(rows),
        "quality_pass_count": int(sum(bool(row["metrics"]["quality_pass"]) for row in rows)),
        "formal_collection_started": False,
        "lerobot_root": str(dataset_root),
        "preview_video_root": str(preview_video_root),
        "results": rows,
    }
    write_json(output_root / "summary.json", summary)
    autosave()
    print(json.dumps(base.to_jsonable(summary), indent=2), flush=True)



# Native-gripper / native-LIBERO-background overrides for this machine.
_NATIVE_PRESETS = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["background_randomization"]["presets"]


def _native_style_env(case: Any, *, seed: int) -> Any:
    from libero.libero.envs.arenas.table_arena import TableArena

    preset = dict(_NATIVE_PRESETS[int(seed) % len(_NATIVE_PRESETS)])
    original_init = TableArena.__init__
    style_calls: list[dict[str, str]] = []

    def styled_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["xml"] = str(
            LIBERO_ROOT / "libero" / "assets" / str(preset["arena_xml"])
        )
        kwargs["floor_style"] = str(preset["floor_style"])
        kwargs["wall_style"] = str(preset["wall_style"])
        style_calls.append(
            {
                "xml": str(kwargs["xml"]),
                "floor_style": str(kwargs["floor_style"]),
                "wall_style": str(kwargs["wall_style"]),
            }
        )
        original_init(self, *args, **kwargs)
        texture_file = str(
            LIBERO_ROOT / "libero" / "assets" / str(preset["table_texture_file"])
        )
        for texture_name in ("tex-table", "tex-table-legs"):
            texture = self.asset.find(f"./texture[@name='{texture_name}']")
            if texture is not None:
                texture.set("file", texture_file)

    TableArena.__init__ = styled_init
    try:
        env = base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed))
        env.reset()
    finally:
        TableArena.__init__ = original_init
    if not style_calls:
        env.close()
        raise RuntimeError("Styled TableArena constructor was not called")
    geom_names = [env.inner_env.sim.model.geom_id2name(i) or "" for i in range(env.inner_env.sim.model.ngeom)]
    if any("board_tool" in name for name in geom_names):
        env.close()
        raise RuntimeError("Native-gripper preview unexpectedly contains a board-tool geom")
    preset["arena_constructor_calls"] = style_calls
    env._native_background_preset = preset
    return env


def randomize_background(env: Any, rng: np.random.Generator, cfg: dict[str, Any]) -> dict[str, Any]:
    del rng, cfg
    preset = dict(env._native_background_preset)
    lighting = dict(preset["lighting"])
    model = env.inner_env.sim.model
    diffuse_tint = np.asarray(lighting["diffuse_tint"], dtype=np.float64)
    ambient_tint = np.asarray(lighting["ambient_tint"], dtype=np.float64)
    specular_tint = np.asarray(lighting["specular_tint"], dtype=np.float64)
    position_offset = np.asarray(lighting["position_offset_m"], dtype=np.float64)
    for light_id in range(int(model.nlight)):
        model.light_diffuse[light_id] = np.clip(
            model.light_diffuse[light_id]
            * float(lighting["diffuse_scale"])
            * diffuse_tint,
            0.0,
            1.0,
        )
        model.light_ambient[light_id] = np.clip(
            model.light_ambient[light_id]
            * float(lighting["ambient_scale"])
            * ambient_tint,
            0.0,
            1.0,
        )
        model.light_specular[light_id] = np.clip(
            model.light_specular[light_id]
            * float(lighting["specular_scale"])
            * specular_tint,
            0.0,
            1.0,
        )
        model.light_pos[light_id] += position_offset
    env.inner_env.sim.forward()
    return {
        "mode": "libero_plus_official_texture_preset",
        "preset_id": str(preset["preset_id"]),
        "floor_style": str(preset["floor_style"]),
        "wall_style": str(preset["wall_style"]),
        "table_texture_file": str(preset["table_texture_file"]),
        "material_rgb": {"table_texture": list(preset["table_rgb"])},
        "lighting": {
            "preset": lighting,
            "resolved_lights": [
                {
                    "position_m": model.light_pos[index].astype(float).tolist(),
                    "diffuse": model.light_diffuse[index].astype(float).tolist(),
                    "ambient": model.light_ambient[index].astype(float).tolist(),
                    "specular": model.light_specular[index].astype(float).tolist(),
                }
                for index in range(int(model.nlight))
            ],
        },
        "arbitrary_rgba_randomization": False,
    }


def _native_box_geom_id(env: Any) -> int:
    return int(env.inner_env.sim.model.geom_name2id("cream_cheese_1_g1"))


def _native_gripper_collision_ids(env: Any) -> list[int]:
    model = env.inner_env.sim.model
    result = []
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or ""
        if not (name.startswith("gripper0_") and "collision" in name):
            continue
        if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]):
            result.append(int(geom_id))
    if not result:
        raise RuntimeError("No native Panda gripper collision geoms found")
    return result


def _geom_support(env: Any, geom_id: int, axis: np.ndarray) -> float:
    sim = env.inner_env.sim
    rotation = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    return float(np.sum(np.abs(rotation.T @ axis) * np.asarray(sim.model.geom_size[geom_id], dtype=np.float64)))


def _native_contact_footprint(env: Any, direction: np.ndarray, target_eef_z: float) -> dict[str, Any]:
    sim = env.inner_env.sim
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    direction3 = np.asarray([direction[0], direction[1], 0.0], dtype=np.float64)
    lateral3 = np.asarray([lateral[0], lateral[1], 0.0], dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
    box_id = _native_box_geom_id(env)
    box_center = np.asarray(sim.data.geom_xpos[box_id], dtype=np.float64).copy()
    box_z_support = _geom_support(env, box_id, up)
    candidates = []
    for geom_id in _native_gripper_collision_ids(env):
        center = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64).copy()
        relative = center - eef
        predicted_z = float(target_eef_z + relative[2])
        z_support = _geom_support(env, geom_id, up)
        if abs(predicted_z - box_center[2]) > box_z_support + z_support + 0.003:
            continue
        candidates.append(
            {
                "geom_id": int(geom_id),
                "name": sim.model.geom_id2name(geom_id),
                "front_offset_m": float(np.dot(relative[:2], direction) + _geom_support(env, geom_id, direction3)),
                "lateral_offset_m": float(np.dot(relative[:2], lateral)),
                "lateral_support_m": _geom_support(env, geom_id, lateral3),
                "predicted_center_z_m": predicted_z,
            }
        )
    if not candidates:
        raise RuntimeError(f"No native gripper collision geom overlaps box height at eef_z={target_eef_z}")
    front_offset = max(float(item["front_offset_m"]) for item in candidates)
    leading = [item for item in candidates if front_offset - float(item["front_offset_m"]) <= 0.004]
    lateral_offset = float(np.mean([float(item["lateral_offset_m"]) for item in leading]))
    return {
        "box_center": box_center,
        "box_support_m": _geom_support(env, box_id, direction3),
        "front_offset_from_eef_m": front_offset,
        "leading_lateral_offset_from_eef_m": lateral_offset,
        "leading_geoms": leading,
        "candidate_geoms": candidates,
    }


def _native_desired_eef(
    env: Any,
    direction: np.ndarray,
    *,
    lateral_offset_m: float,
    standoff_m: float,
    target_eef_z: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    footprint = _native_contact_footprint(env, direction, target_eef_z)
    box_center = np.asarray(footprint["box_center"], dtype=np.float64)
    desired_projection = (
        float(np.dot(box_center[:2], direction))
        - float(footprint["box_support_m"])
        - float(standoff_m)
        - float(footprint["front_offset_from_eef_m"])
    )
    desired_lateral = (
        float(np.dot(box_center[:2], lateral))
        + float(lateral_offset_m)
        - float(footprint["leading_lateral_offset_from_eef_m"])
    )
    xy = direction * desired_projection + lateral * desired_lateral
    return np.asarray([xy[0], xy[1], float(target_eef_z)], dtype=np.float64), footprint


def board_box_contact(env: Any) -> bool:
    sim = env.inner_env.sim
    box_id = _native_box_geom_id(env)
    gripper_ids = set(_native_gripper_collision_ids(env))
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if box_id in pair and pair.intersection(gripper_ids):
            return True
    return False


def configure_directional_board(env: Any, direction: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    env._native_pusher_cfg = dict(cfg)
    return {
        "tool": "native_panda_gripper",
        "board_present": False,
        "normal_alignment_error_deg": 0.0,
        "path_direction_xy": np.asarray(direction, dtype=float).tolist(),
        "collision_geoms": [
            env.inner_env.sim.model.geom_id2name(geom_id) for geom_id in _native_gripper_collision_ids(env)
        ],
    }


def _native_touch_geometry(env: Any, direction: np.ndarray) -> dict[str, Any]:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64).copy()
    box_xyz, _ = env.box_pose()
    footprint = _native_contact_footprint(env, direction, float(eef[2]))
    pusher_front = float(np.dot(eef[:2], direction) + footprint["front_offset_from_eef_m"])
    box_rear = float(np.dot(box_xyz[:2], direction) - footprint["box_support_m"])
    return {
        "box_center": np.asarray(box_xyz, dtype=float).tolist(),
        "eef_center": eef.astype(float).tolist(),
        "pusher_front_projection_m": pusher_front,
        "box_rear_projection_m": box_rear,
        "estimated_gap_m": float(box_rear - pusher_front),
        "center_lateral_error_m": float(
            np.dot(np.asarray(box_xyz[:2], dtype=np.float64) - eef[:2], lateral)
            - footprint["leading_lateral_offset_from_eef_m"]
        ),
        "leading_geoms": footprint["leading_geoms"],
        "actual_contact": bool(board_box_contact(env)),
    }


def box_and_board_geometry(env: Any, direction: np.ndarray) -> dict[str, Any]:
    geometry = _native_touch_geometry(env, direction)
    return {
        "box_center": np.asarray(geometry["box_center"], dtype=np.float64),
        "board_center": np.asarray(geometry["eef_center"], dtype=np.float64),
        "box_support_m": 0.0,
        "board_support_m": 0.0,
        "gap_m": float(geometry["estimated_gap_m"]),
    }


def prepare_directional_touch(
    env: Any,
    *,
    direction: np.ndarray,
    lateral_offset_m: float,
) -> dict[str, Any]:
    cfg = dict(env._native_pusher_cfg)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    target_z = float(env.case.pusher_contact_z)
    desired, footprint = _native_desired_eef(
        env,
        direction,
        lateral_offset_m=float(lateral_offset_m),
        standoff_m=float(cfg["standoff_m"]),
        target_eef_z=target_z,
    )
    above = desired.copy()
    above[2] += float(cfg["approach_height_m"])
    move_eef(env, above, steps=int(cfg["approach_steps"]), max_action=0.18, gain=4.0)
    desired, footprint = _native_desired_eef(
        env,
        direction,
        lateral_offset_m=float(lateral_offset_m),
        standoff_m=float(cfg["standoff_m"]),
        target_eef_z=target_z,
    )
    move_eef(env, desired, steps=int(cfg["descend_steps"]), max_action=0.24, gain=6.0)
    desired, footprint = _native_desired_eef(
        env,
        direction,
        lateral_offset_m=float(lateral_offset_m),
        standoff_m=float(cfg["standoff_m"]),
        target_eef_z=target_z,
    )
    move_eef(env, desired, steps=int(cfg["alignment_steps"]), max_action=0.12, gain=8.0)

    touch_start_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64).copy()
    touch_start_box, _ = env.box_pose()
    target_lateral = float(np.dot(desired[:2], lateral))
    touch_trigger = ""
    projected_box_motion_m = 0.0
    projected_box_speed_mps = 0.0
    touch_steps = 0
    for touch_steps in range(1, int(cfg["touch_max_steps"]) + 1):
        if board_box_contact(env):
            touch_trigger = "contact_before_step"
            break
        eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
        action = np.zeros(7, dtype=np.float64)
        lateral_error = target_lateral - float(np.dot(eef[:2], lateral))
        action[:2] = direction * float(cfg["touch_action"]) + lateral * float(np.clip(6.0 * lateral_error, -0.08, 0.08))
        action[2] = float(np.clip(6.0 * (target_z - eef[2]), -0.08, 0.08))
        action[-1] = float(cfg["gripper_command"])
        env.step_count = 0
        env.step(action)
        box_xyz, _ = env.box_pose()
        projected_box_motion_m = float(np.dot(box_xyz[:2] - touch_start_box[:2], direction))
        projected_box_speed_mps = float(np.dot(env.box_velocity()[:2], direction))
        if board_box_contact(env):
            touch_trigger = "contact_after_step"
            break
        if projected_box_motion_m >= float(cfg["contact_motion_threshold_m"]):
            touch_trigger = "box_motion"
            break
        if projected_box_speed_mps >= float(cfg["contact_speed_threshold_mps"]):
            touch_trigger = "box_speed"
            break
        travel = float(np.dot(np.asarray(env._last_obs["robot0_eef_pos"])[:2] - touch_start_eef[:2], direction))
        if travel > float(cfg["touch_max_travel_m"]):
            break
    if not touch_trigger:
        raise RuntimeError(
            f"Native gripper touch failed after {touch_steps} steps; "
            f"geometry={_native_touch_geometry(env, direction)}"
        )

    sim = env.inner_env.sim
    obj = env.inner_env.get_object(env.case.box_name)
    joint_name = obj.joints[-1]
    qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    latch_shift_m = 0.0
    search_step = float(cfg["latch_search_step_m"])
    search_count = int(round(float(cfg["latch_search_max_m"]) / search_step))
    sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()
    for _ in range(search_count + 1):
        if board_box_contact(env):
            break
        qpos[:2] -= direction * search_step
        latch_shift_m += search_step
        sim.data.set_joint_qpos(joint_name, qpos)
        sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
        sim.forward()
    if not board_box_contact(env):
        raise RuntimeError(
            f"Native gripper contact latch failed after shifting box {latch_shift_m:.6f} m; "
            f"geometry={_native_touch_geometry(env, direction)}"
        )
    sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()
    env._last_obs = env._refresh_obs()
    geometry = _native_touch_geometry(env, direction)
    return {
        "touch_steps": int(touch_steps),
        "touch_trigger": touch_trigger,
        "projected_box_motion_before_latch_m": float(projected_box_motion_m),
        "projected_box_speed_before_latch_mps": float(projected_box_speed_mps),
        "latch_box_shift_toward_gripper_m": float(latch_shift_m),
        "geometry": geometry,
        "launch_box_speed_mps": float(np.linalg.norm(env.box_velocity()[:2])),
    }


touch.make_env = _native_style_env


if __name__ == "__main__":
    main()

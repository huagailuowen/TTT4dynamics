#!/usr/bin/env python3
"""Collect a combined 20-training-ratio + 10-test-ratio mass-balance dataset.

The LeRobot `action` is an absolute EEF setpoint. It is never copied from
`observation.state`; LIBERO's internal delta OSC command is computed only at the
simulator boundary. The short rod is a free rigid body physically held by the Panda
from the first recorded frame onward. No weld, suction, mocap following, or object
state write is allowed after initialization.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT.parent / "LIBERO"
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM-TTT"
ROD_XML = REPO_ROOT / "assets" / "short_lifting_rod_2026-07-21_hai-machine.xml"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mass_balance_fixed_pose_30ratio_15support_450eps_combined_split_2026-09-02_hai-machine.json"
ROD_NAME = "short_lifting_rod_1"
TASK = [
    "mass-balance lift with a physically grasped short rod",
    "lift the asymmetric carrying beam at the commanded support position",
    "probe the hidden left-right mass ratio through physical balance",
    "absolute-EEF physical rod lifting demonstration",
]

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_ROOT / ".libero_config"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
for local_path in (REPO_ROOT, REPO_ROOT / "scripts", LIBERO_ROOT, FASTWAM_ROOT / "src", FASTWAM_ROOT):
    if local_path.exists() and str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from collect_libero_push_box_rollout_target_lerobot_dataset import (  # noqa: E402
    _obs_to_images,
    _obs_to_state,
    _write_image_for_last_frame,
    patch_lerobot_video_crf,
)
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset  # noqa: E402
from libero.libero.envs.base_object import register_object  # noqa: E402
from robosuite.models.objects import MujocoXMLObject  # noqa: E402
from robosuite.utils.errors import RandomizationError  # noqa: E402
from ttt4dynamics.push_box_libero import LiberoPushBoxCase, LiberoPushBoxEnv  # noqa: E402


@register_object
class ShortLiftingRod(MujocoXMLObject):
    def __init__(self, name: str = "short_lifting_rod"):
        super().__init__(
            str(ROD_XML),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "short_lifting_rod"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--video-codec", default="h264")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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
                "actual_eef_x_m",
                "actual_eef_y_m",
                "actual_eef_z_m",
                "actual_eef_axis_x_rad",
                "actual_eef_axis_y_rad",
                "actual_eef_axis_z_rad",
                "actual_gripper_qpos_0",
                "actual_gripper_qpos_1",
            ],
        },
        "observation.object_poses": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "beam_x_m",
                "beam_y_m",
                "beam_z_m",
                "beam_qw",
                "beam_qx",
                "beam_qy",
                "beam_qz",
                "rod_x_m",
                "rod_y_m",
                "rod_z_m",
                "rod_qw",
                "rod_qx",
                "rod_qy",
                "rod_qz",
            ],
        },
        "observation.contact_state": {
            "dtype": "float32",
            "shape": (4,),
            "names": [
                "robosuite_grasping_rod",
                "gripper_rod_contact",
                "rod_beam_contact",
                "direct_gripper_beam_contact",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "target_eef_x_m",
                "target_eef_y_m",
                "target_eef_z_m",
                "target_rx_rad",
                "target_ry_rad",
                "target_rz_rad",
                "target_gripper_open",
            ],
        },
    }


def ratio_token(index: int) -> str:
    return f"{chr(ord('a') + index // 26)}{chr(ord('a') + index % 26)}"


def ratio_category(index: int) -> str:
    return f"mass_balance_beam_ratio_{ratio_token(index)}"


def ratio_class_name(index: int) -> str:
    return f"MassBalanceBeamRatio{ratio_token(index).title()}"


def mass_spec(config: dict[str, Any], index: int, ratio: float) -> dict[str, Any]:
    end_sum = float(config["combined_end_mass_kg"])
    left = end_sum / (1.0 + float(ratio))
    right = end_sum - left
    crossbar = float(config["crossbar_mass_kg"])
    end_offset = float(config["end_mass_offset_m"])
    total = crossbar + left + right
    com = end_offset * (right - left) / total
    return {
        "ratio_index": int(index),
        "dataset_split": "training" if index < int(config["training_mass_ratio_count"]) else "test",
        "right_to_left_mass_ratio": float(ratio),
        "left_end_mass_kg": float(left),
        "right_end_mass_kg": float(right),
        "crossbar_mass_kg": float(crossbar),
        "total_mass_kg": float(total),
        "theoretical_com_offset_m": float(com),
        "category": ratio_category(index),
        "class_name": ratio_class_name(index),
        "beam_name": f"{ratio_category(index)}_1",
    }


def write_ratio_xml(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<mujoco model=\"{spec['category']}_hai_machine\">
  <worldbody>
    <body>
      <body name=\"object\" pos=\"0 0 0\">
        <geom name=\"wooden_crossbar\" type=\"box\" pos=\"0 0 0.055\" size=\"0.180 0.018 0.012\"
              mass=\"{spec['crossbar_mass_kg']:.9f}\" friction=\"0.65 0.01 0.0001\" solref=\"0.005 1\"
              solimp=\"0.995 0.995 0.001\" contype=\"1\" conaffinity=\"1\" group=\"0\" rgba=\"0.52 0.25 0.07 1\"/>
        <geom name=\"left_hidden_mass\" type=\"box\" pos=\"-0.150 0 0.040\" size=\"0.025 0.040 0.040\"
              mass=\"{spec['left_end_mass_kg']:.9f}\" friction=\"0.65 0.01 0.0001\" solref=\"0.005 1\"
              solimp=\"0.995 0.995 0.001\" contype=\"1\" conaffinity=\"1\" group=\"0\" rgba=\"0.78 0.56 0.24 1\"/>
        <geom name=\"right_hidden_mass\" type=\"box\" pos=\"0.150 0 0.040\" size=\"0.025 0.040 0.040\"
              mass=\"{spec['right_end_mass_kg']:.9f}\" friction=\"0.65 0.01 0.0001\" solref=\"0.005 1\"
              solimp=\"0.995 0.995 0.001\" contype=\"1\" conaffinity=\"1\" group=\"0\" rgba=\"0.78 0.56 0.24 1\"/>
        <geom name=\"wooden_crossbar_visual\" type=\"box\" pos=\"0 0 0.055\" size=\"0.180 0.018 0.012\"
              density=\"0\" contype=\"0\" conaffinity=\"0\" group=\"1\" rgba=\"0.52 0.25 0.07 1\"/>
        <geom name=\"left_hidden_mass_visual\" type=\"box\" pos=\"-0.150 0 0.040\" size=\"0.025 0.040 0.040\"
              density=\"0\" contype=\"0\" conaffinity=\"0\" group=\"1\" rgba=\"0.78 0.56 0.24 1\"/>
        <geom name=\"right_hidden_mass_visual\" type=\"box\" pos=\"0.150 0 0.040\" size=\"0.025 0.040 0.040\"
              density=\"0\" contype=\"0\" conaffinity=\"0\" group=\"1\" rgba=\"0.78 0.56 0.24 1\"/>
      </body>
      <site name=\"bottom_site\" pos=\"0 0 0\" size=\"0.005\" rgba=\"0 0 0 0\"/>
      <site name=\"top_site\" pos=\"0 0 0.080\" size=\"0.005\" rgba=\"0 0 0 0\"/>
      <site name=\"horizontal_radius_site\" pos=\"0.180 0.040 0.040\" size=\"0.005\" rgba=\"0 0 0 0\"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


def register_ratio_beam(spec: dict[str, Any], xml_path: Path) -> None:
    category = str(spec["category"])

    def init(self: MujocoXMLObject, name: str = category) -> None:
        MujocoXMLObject.__init__(
            self,
            str(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = category
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}

    cls = type(str(spec["class_name"]), (MujocoXMLObject,), {"__init__": init})
    register_object(cls)


def write_bddl(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language lift an asymmetric carrying beam with a physically grasped short rod)
    (:regions
      (beam_init_region
          (:target main_table)
          (:ranges ((-0.0520 -0.0020 -0.0480 0.0020)))
          (:yaw_rotation ((1.5707963 1.5707963)))
      )
      (rod_init_region
          (:target main_table)
          (:ranges ((-0.2720 -0.2220 -0.2680 -0.2180)))
          (:yaw_rotation ((0.0 0.0)))
      )
      (dummy_goal_region
          (:target main_table)
          (:ranges ((0.3000 -0.0600 0.3400 -0.0200)))
          (:rgba (0.0 0.0 0.0 0.0))
      )
    )
  (:fixtures main_table - table)
  (:objects
    {spec['beam_name']} - {spec['category']}
    {ROD_NAME} - short_lifting_rod
  )
  (:obj_of_interest {spec['beam_name']} {ROD_NAME})
  (:init
    (On {spec['beam_name']} main_table_beam_init_region)
    (On {ROD_NAME} main_table_rod_init_region)
  )
  (:goal (And (On {spec['beam_name']} main_table_dummy_goal_region)))
)
""",
        encoding="utf-8",
    )


def object_instance(env: LiberoPushBoxEnv, name: str) -> Any:
    return env.inner_env.get_object(name)


def object_state(env: LiberoPushBoxEnv, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = object_instance(env, name)
    joint = obj.joints[-1]
    qpos = np.asarray(env.inner_env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    qvel = np.asarray(env.inner_env.sim.data.get_joint_qvel(joint), dtype=np.float64).copy()
    return qpos[:3], qpos[3:7], qvel


def initialize_free_object(
    env: LiberoPushBoxEnv,
    name: str,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    obj = object_instance(env, name)
    joint = obj.joints[-1]
    env.inner_env.sim.data.set_joint_qpos(
        joint,
        np.concatenate([np.asarray(position, dtype=np.float64), np.asarray(quat_wxyz, dtype=np.float64)]),
    )
    env.inner_env.sim.data.set_joint_qvel(joint, np.zeros(6, dtype=np.float64))


def quat_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def beam_tilt_degrees(quat_wxyz: np.ndarray) -> float:
    axis = quat_rotation_matrix(quat_wxyz)[:, 0]
    return math.degrees(math.asin(float(np.clip(abs(axis[2]), 0.0, 1.0))))


def geom_ids(env: LiberoPushBoxEnv, name: str) -> set[int]:
    obj = object_instance(env, name)
    names = obj.contact_geoms() if callable(obj.contact_geoms) else obj.contact_geoms
    model = env.inner_env.sim.model
    result: set[int] = set()
    for geom_name in names:
        try:
            result.add(int(model.geom_name2id(geom_name)))
        except Exception:
            pass
    return result


def gripper_geom_ids(env: LiberoPushBoxEnv) -> set[int]:
    model = env.inner_env.sim.model
    result: set[int] = set()
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or ""
        if name.startswith("gripper0_") and ("collision" in name or "pad" in name):
            result.add(int(geom_id))
    return result


def has_contact(env: LiberoPushBoxEnv, first: set[int], second: set[int]) -> bool:
    data = env.inner_env.sim.data
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if (geom1 in first and geom2 in second) or (geom2 in first and geom1 in second):
            return True
    return False


def robosuite_grasping(env: LiberoPushBoxEnv) -> bool:
    try:
        return bool(
            env.inner_env._check_grasp(
                gripper=env.inner_env.robots[0].gripper,
                object_geoms=object_instance(env, ROD_NAME),
            )
        )
    except Exception:
        return False


def object_pose_vector(env: LiberoPushBoxEnv, beam_name: str) -> np.ndarray:
    beam_pos, beam_quat, _ = object_state(env, beam_name)
    rod_pos, rod_quat, _ = object_state(env, ROD_NAME)
    return np.concatenate([beam_pos, beam_quat, rod_pos, rod_quat]).astype(np.float32)


def copy_required_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    keys = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    return {key: np.asarray(obs[key]).copy() for key in keys}


def initialize_pregripped_episode(
    env: LiberoPushBoxEnv,
    *,
    beam_name: str,
    beam_xy: np.ndarray,
    settle_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_attempts = 0
    while True:
        reset_attempts += 1
        try:
            obs = env.reset()
            break
        except RandomizationError:
            if reset_attempts >= 8:
                raise
    sim = env.inner_env.sim
    initialize_free_object(
        env,
        beam_name,
        np.asarray([beam_xy[0], beam_xy[1], 0.9000], dtype=np.float64),
        np.asarray([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=np.float64),
    )
    gripper = env.inner_env.robots[0].gripper
    for joint, value in zip(gripper.joints, (0.016, -0.024)):
        sim.data.set_joint_qpos(joint, float(value))
    sim.forward()
    obs = env._refresh_obs()
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    initialize_free_object(
        env,
        ROD_NAME,
        eef + np.asarray([0.0009, 0.00374, -0.01114], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    sim.forward()
    close_hold = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    grasp_checks = []
    for _ in range(int(settle_steps)):
        obs, _, _, _ = env.step(close_hold)
        grasp_checks.append(robosuite_grasping(env))
    if not all(grasp_checks[-min(10, len(grasp_checks)) :]):
        raise RuntimeError("Pregripped rod did not establish a stable physical grasp")
    beam_pos, beam_quat, _ = object_state(env, beam_name)
    rod_pos, rod_quat, _ = object_state(env, ROD_NAME)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return obs, {
        "beam_xyz_m": beam_pos.tolist(),
        "beam_quat_wxyz": beam_quat.tolist(),
        "rod_xyz_m": rod_pos.tolist(),
        "rod_quat_wxyz": rod_quat.tolist(),
        "eef_xyz_m": eef.tolist(),
        "rod_minus_eef_m": (rod_pos - eef).tolist(),
        "settle_grasp_true_frames": int(sum(grasp_checks)),
        "settle_steps": int(settle_steps),
        "discarded_bddl_sampler_attempts_before_exact_initialization": int(reset_attempts - 1),
    }


def create_env_with_sampler_retries(
    case: LiberoPushBoxCase,
    *,
    seed: int,
    max_attempts: int = 8,
) -> tuple[LiberoPushBoxEnv, int]:
    """Retry only LIBERO's discarded pre-initialization placement sampler."""
    for attempt in range(int(max_attempts)):
        try:
            return LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed + attempt)), attempt
        except RandomizationError:
            if attempt + 1 >= int(max_attempts):
                raise
    raise RuntimeError("unreachable")


def rollout_episode(
    env: LiberoPushBoxEnv,
    *,
    beam_name: str,
    support_offset_m: float,
    beam_xy: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectory = config["trajectory"]
    obs, initialization = initialize_pregripped_episode(
        env,
        beam_name=beam_name,
        beam_xy=beam_xy,
        settle_steps=int(config["pregrip_settle_steps"]),
    )
    initial_beam_pos, initial_beam_quat, _ = object_state(env, beam_name)
    rod_pos, rod_quat, _ = object_state(env, ROD_NAME)
    beam_axis = quat_rotation_matrix(initial_beam_quat)[:, 0]
    rod_axis = quat_rotation_matrix(rod_quat)[:, 0]
    insertion_axis = rod_axis if float(np.dot(rod_axis, initial_beam_pos - rod_pos)) >= 0.0 else -rod_axis
    support_point = initial_beam_pos + beam_axis * float(support_offset_m)
    nominal_beam_z = float(config["nominal_beam_z_m"])
    under_z = float(nominal_beam_z + float(trajectory["under_beam_rod_center_z_offset_m"]))
    final_rod_center = support_point - insertion_axis * float(trajectory["gripper_to_support_lever_m"])
    final_rod_center[2] = under_z
    approach_rod_center = final_rod_center - insertion_axis * float(trajectory["staging_backoff_m"])
    approach_rod_center[2] = under_z
    lifted_rod_center = final_rod_center.copy()
    lifted_rod_center[2] = nominal_beam_z + float(trajectory["lifted_rod_center_z_offset_m"])
    rod_minus_eef = np.asarray(initialization["rod_minus_eef_m"], dtype=np.float64)
    rod_minus_eef[2] = float(config["nominal_rod_minus_eef_z_m"])

    def eef_for_rod_center(center: np.ndarray) -> np.ndarray:
        return np.asarray(center, dtype=np.float64) - rod_minus_eef

    gripper_ids = gripper_geom_ids(env)
    rod_ids = geom_ids(env, ROD_NAME)
    beam_ids = geom_ids(env, beam_name)
    records: list[dict[str, Any]] = []
    command_target = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()

    def run_phase(
        phase: str,
        target_end: np.ndarray,
        *,
        steps: int,
        max_delta_action: float,
        ramp_steps: int = 0,
    ) -> None:
        nonlocal obs, command_target
        end = np.asarray(target_end, dtype=np.float64)
        start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        ramp = min(max(int(ramp_steps), 0), int(steps))
        for step_index in range(int(steps)):
            if ramp > 0:
                alpha = min(float(step_index + 1) / float(ramp), 1.0)
                absolute_target = start + alpha * (end - start)
            else:
                absolute_target = end
            absolute_action = np.asarray(
                [absolute_target[0], absolute_target[1], absolute_target[2], 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            beam_pos, beam_quat, _ = object_state(env, beam_name)
            grasp = robosuite_grasping(env)
            gripper_rod = has_contact(env, gripper_ids, rod_ids)
            rod_beam = has_contact(env, rod_ids, beam_ids)
            direct_gripper_beam = has_contact(env, gripper_ids, beam_ids)
            eef_actual = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            delta_action = np.zeros(7, dtype=np.float64)
            delta_action[:3] = np.clip(4.0 * (absolute_target - eef_actual), -max_delta_action, max_delta_action)
            delta_action[-1] = 1.0
            records.append(
                {
                    "frame_index": len(records),
                    "phase": phase,
                    "obs": copy_required_obs(obs),
                    "absolute_action": absolute_action,
                    "controller_delta_action": delta_action.copy(),
                    "object_poses": object_pose_vector(env, beam_name),
                    "contact_state": np.asarray(
                        [float(grasp), float(gripper_rod), float(rod_beam), float(direct_gripper_beam)],
                        dtype=np.float32,
                    ),
                    "beam_lift_m": float(beam_pos[2] - initial_beam_pos[2]),
                    "beam_tilt_deg": float(beam_tilt_degrees(beam_quat)),
                    "action_state_position_error_m": float(np.linalg.norm(absolute_target - eef_actual)),
                }
            )
            obs, _, _, _ = env.step(delta_action)
        command_target = end.copy()

    run_phase(
        "direct_approach_to_beam_gap",
        eef_for_rod_center(approach_rod_center),
        steps=int(trajectory["approach_steps"]),
        max_delta_action=float(trajectory.get("approach_max_delta_action", 0.45)),
        ramp_steps=int(trajectory.get("approach_ramp_steps", 0)),
    )
    run_phase(
        "insert_under_beam",
        eef_for_rod_center(final_rod_center),
        steps=int(trajectory["insert_steps"]),
        max_delta_action=float(trajectory.get("insert_max_delta_action", 0.28)),
    )
    run_phase(
        "lift_at_sampled_support",
        eef_for_rod_center(lifted_rod_center),
        steps=int(trajectory["lift_steps"]),
        max_delta_action=float(trajectory.get("lift_max_delta_action", 0.16)),
    )
    run_phase(
        "hold_balance_result",
        eef_for_rod_center(lifted_rod_center),
        steps=int(trajectory["hold_steps"]),
        max_delta_action=float(trajectory.get("hold_max_delta_action", 0.16)),
    )

    phase_segments: list[dict[str, Any]] = []
    for row in records:
        if not phase_segments or phase_segments[-1]["name"] != row["phase"]:
            phase_segments.append(
                {
                    "phase_id": len(phase_segments),
                    "name": row["phase"],
                    "start_frame": int(row["frame_index"]),
                    "end_frame": int(row["frame_index"]),
                    "frame_count": 1,
                }
            )
        else:
            phase_segments[-1]["end_frame"] = int(row["frame_index"])
            phase_segments[-1]["frame_count"] += 1

    hold = [row for row in records if row["phase"] == "hold_balance_result"]
    action_state_errors = [float(row["action_state_position_error_m"]) for row in records]
    phase_final_errors = {
        segment["name"]: float(records[segment["end_frame"]]["action_state_position_error_m"])
        for segment in phase_segments
    }
    phase_final_poses = {
        segment["name"]: {
            "actual_eef_xyz_m": np.asarray(
                records[segment["end_frame"]]["obs"]["robot0_eef_pos"], dtype=np.float64
            ).tolist(),
            "target_eef_xyz_m": np.asarray(
                records[segment["end_frame"]]["absolute_action"][:3], dtype=np.float64
            ).tolist(),
            "beam_xyz_m": np.asarray(
                records[segment["end_frame"]]["object_poses"][:3], dtype=np.float64
            ).tolist(),
            "rod_xyz_m": np.asarray(
                records[segment["end_frame"]]["object_poses"][7:10], dtype=np.float64
            ).tolist(),
            "rod_quat_wxyz": np.asarray(
                records[segment["end_frame"]]["object_poses"][10:14], dtype=np.float64
            ).tolist(),
        }
        for segment in phase_segments
    }
    grasp_frames = int(sum(int(row["contact_state"][0]) for row in records))
    rod_beam_frames = int(sum(int(row["contact_state"][2]) for row in records))
    direct_lift_hold = int(
        sum(
            int(row["contact_state"][3])
            for row in records
            if row["phase"] in {"lift_at_sampled_support", "hold_balance_result"}
        )
    )
    max_lift = float(max(row["beam_lift_m"] for row in records))
    hold_max_tilt = float(max(row["beam_tilt_deg"] for row in hold))
    metrics = {
        "frame_count": len(records),
        "robosuite_grasp_frames": grasp_frames,
        "gripper_rod_contact_frames": int(sum(int(row["contact_state"][1]) for row in records)),
        "rod_beam_contact_frames": rod_beam_frames,
        "direct_gripper_beam_contact_frames_during_lift_hold": direct_lift_hold,
        "max_beam_lift_m": max_lift,
        "hold_max_beam_tilt_deg": hold_max_tilt,
        "hold_final_beam_tilt_deg": float(hold[-1]["beam_tilt_deg"]),
        "balance_outcome": "stable" if max_lift >= 0.02 and hold_max_tilt <= 5.0 else "unstable",
        "mean_action_state_position_error_m": float(np.mean(action_state_errors)),
        "max_action_state_position_error_m": float(max(action_state_errors)),
        "phase_final_action_state_position_error_m": phase_final_errors,
        "phase_final_poses": phase_final_poses,
        "initialization": initialization,
        "initial_beam_axis_world": beam_axis.tolist(),
        "initial_rod_axis_world": rod_axis.tolist(),
        "insertion_axis_world": insertion_axis.tolist(),
        "support_point_world_m": support_point.tolist(),
        "nominal_action_beam_z_m": nominal_beam_z,
        "nominal_action_rod_minus_eef_z_m": float(config["nominal_rod_minus_eef_z_m"]),
        "phase_segments": phase_segments,
    }
    physical_valid = bool(
        grasp_frames == len(records)
        and rod_beam_frames >= 5
        and direct_lift_hold == 0
        and max_lift >= 0.02
        and metrics["max_action_state_position_error_m"] >= 0.005
    )
    metrics["valid_physical_episode"] = physical_valid
    if not physical_valid:
        raise RuntimeError(f"Invalid physical episode: {metrics}")
    return records, metrics


def write_episode(
    dataset: LeRobotDataset,
    records: list[dict[str, Any]],
    *,
    fps: int,
    jpeg_quality: int,
) -> int:
    episode_index = int(dataset.meta.total_episodes)
    for frame_index, record in enumerate(records):
        agent, wrist = _obs_to_images(record["obs"])
        frame = {
            "observation.images.image": agent,
            "observation.images.wrist_image": wrist,
            "observation.state": _obs_to_state(record["obs"]),
            "observation.object_poses": np.asarray(record["object_poses"], dtype=np.float32),
            "observation.contact_state": np.asarray(record["contact_state"], dtype=np.float32),
            "action": np.asarray(record["absolute_action"], dtype=np.float32),
        }
        dataset.add_frame(frame, task=TASK, timestamp=float(frame_index) / float(fps))
        _write_image_for_last_frame(
            dataset,
            "observation.images.image",
            frame_index,
            agent,
            jpeg_quality=jpeg_quality,
        )
        _write_image_for_last_frame(
            dataset,
            "observation.images.wrist_image",
            frame_index,
            wrist,
            jpeg_quality=jpeg_quality,
        )
    dataset.save_episode()
    return episode_index


def create_episode_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    ratios = [float(value) for value in config["right_to_left_mass_ratios"]]
    bin_count = int(config["support_bin_count"])
    low, high = [float(value) for value in config["support_center_range_m"]]
    edges = np.linspace(low, high, bin_count + 1, dtype=np.float64)
    support_rng = np.random.default_rng(int(config["support_seed"]))
    placement_rng = np.random.default_rng(int(config["placement_seed"]))
    x_low, x_high = [float(value) for value in config["beam_front_back_x_offset_range_m"]]
    y_low, y_high = [float(value) for value in config["beam_left_right_y_offset_range_m"]]
    plan = []
    for ratio_index, ratio in enumerate(ratios):
        for support_bin in range(bin_count):
            plan.append(
                {
                    "ratio_index": int(ratio_index),
                    "right_to_left_mass_ratio": float(ratio),
                    "support_bin_index": int(support_bin),
                    "support_bin_low_m": float(edges[support_bin]),
                    "support_bin_high_m": float(edges[support_bin + 1]),
                    "sampled_support_offset_m": float(support_rng.uniform(edges[support_bin], edges[support_bin + 1])),
                    "beam_front_back_x_offset_m": float(placement_rng.uniform(x_low, x_high)),
                    "beam_left_right_y_offset_m": float(placement_rng.uniform(y_low, y_high)),
                }
            )
    return plan


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (args.output or Path(config["output"])).resolve()
    ratio_count = len(config["right_to_left_mass_ratios"])
    support_bin_count = int(config["support_bin_count"])
    training_ratio_count = int(config["training_mass_ratio_count"])
    test_ratio_count = ratio_count - training_ratio_count
    expected_total = ratio_count * support_bin_count
    configured_total = int(config.get("expected_episode_count", expected_total))
    if expected_total != configured_total:
        raise ValueError(
            f"Configured grid is {ratio_count}x{support_bin_count}={expected_total}, "
            f"not expected_episode_count={configured_total}"
        )
    if training_ratio_count != 20 or test_ratio_count != 10:
        raise ValueError(
            f"Expected combined split 20 training + 10 test ratios, got "
            f"{training_ratio_count} + {test_ratio_count}"
        )
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite only when replacement is intended: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    patch_lerobot_video_crf(int(config["video_crf"]))
    dataset = LeRobotDataset.create(
        repo_id=str(config["repo_id"]),
        root=output,
        fps=int(config["fps"]),
        features=build_features(int(config["camera_resolution"])),
        use_videos=True,
        video_codec=str(args.video_codec),
        is_compute_episode_stats_image=False,
    )
    write_json(output / "resolved_generation_config.json", config)

    specs = [
        mass_spec(config, index, float(ratio))
        for index, ratio in enumerate(config["right_to_left_mass_ratios"])
    ]
    for spec in specs:
        xml_path = output / "assets" / f"{spec['category']}.xml"
        write_ratio_xml(xml_path, spec)
        register_ratio_beam(spec, xml_path)
        bddl_path = output / "bddl" / f"{spec['category']}.bddl"
        write_bddl(bddl_path, spec)
        spec["xml_file"] = str(xml_path)
        spec["bddl_file"] = str(bddl_path)

    plan = create_episode_plan(config)
    if args.max_episodes is not None:
        plan = plan[: int(args.max_episodes)]
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": config["dataset_type"],
        "variant": config["variant"],
        "episodes_expected_full_grid": expected_total,
        "episodes_requested_this_run": len(plan),
        "grid": {"mass_ratio_count": ratio_count, "support_bin_count": support_bin_count},
        "split_definition": {
            "training_ratio_count": training_ratio_count,
            "test_ratio_count": test_ratio_count,
            "training_right_to_left_mass_ratios": config["training_right_to_left_mass_ratios"],
            "test_right_to_left_mass_ratios": config["test_right_to_left_mass_ratios"],
            "episode_split_field": "dataset_split",
        },
        "mass_specs": specs,
        "support_sampling": {
            "method": "one independent uniform sample inside each of 15 equal-width support intervals for every mass ratio",
            "range_m": config["support_center_range_m"],
            "support_seed": int(config["support_seed"]),
            "outcome_filtering": False,
        },
        "placement_sampling": {
            "base_xy_m": config["beam_base_xy_m"],
            "front_back_x_offset_range_m": config["beam_front_back_x_offset_range_m"],
            "left_right_y_offset_range_m": config["beam_left_right_y_offset_range_m"],
            "placement_seed": int(config["placement_seed"]),
            "outcome_filtering": False,
        },
        "action_definition": {
            "type": "absolute_eef_setpoint_not_state",
            "vector": ["target_x_m", "target_y_m", "target_z_m", "target_rx_rad", "target_ry_rad", "target_rz_rad", "gripper_open"],
            "rotation": "fixed [0,0,0] command",
            "gripper": "fixed 0.0 closed in dataset convention",
            "simulator_boundary": "absolute target minus actual observation is converted to a clipped LIBERO delta OSC action; this delta is not stored as the training action",
            "z_definition": "two mass-independent nominal absolute EEF heights; no settled beam height enters action z",
            "nominal_beam_z_m": float(config["nominal_beam_z_m"]),
            "nominal_rod_minus_eef_z_m": float(config["nominal_rod_minus_eef_z_m"]),
        },
        "state_definition": "actual pre-action LIBERO observation: EEF position, observed EEF quaternion converted to axis-angle, and two gripper joint positions",
        "physics_guarantees": {
            "initial_condition": "beam, gripper joints, and free rod are placed once before unrecorded settling so frame 0 begins physically pregripped",
            "after_first_recorded_frame": "no qpos, qvel, body pose, model mass, weld, suction, or attachment writes",
            "mass_variation": "30 separately compiled MuJoCo XML objects; first 20 are training ratios and last 10 are held-out test ratios",
            "required": "robosuite grasp every recorded frame, real rod-beam contact, no direct gripper-beam contact during lift/hold",
            "sampler_retry_scope": "only discarded BDDL temporary placement before exact initialization; never based on recorded outcome",
        },
        "camera_resolution": int(config["camera_resolution"]),
        "fps": int(config["fps"]),
        "video_codec": str(args.video_codec),
        "video_crf": int(config["video_crf"]),
        "video_gop": 2,
        "background_domain_randomization": False,
        "episodes": rows,
    }

    def autosave() -> None:
        metadata["episodes_collected"] = len(rows)
        metadata["episodes_by_split"] = {
            "training": sum(row["dataset_split"] == "training" for row in rows),
            "test": sum(row["dataset_split"] == "test" for row in rows),
        }
        write_json(output / "mass_balance_generation_metadata.json", metadata)
        write_jsonl(output / "meta" / "mass_balance_episode_metadata.jsonl", rows)

    base_xy = np.asarray(config["beam_base_xy_m"], dtype=np.float64)
    collected = 0
    for ratio_index, spec in enumerate(specs):
        ratio_plan = [entry for entry in plan if int(entry["ratio_index"]) == ratio_index]
        if not ratio_plan:
            continue
        case = LiberoPushBoxCase(
            case_id=f"mass_balance_ratio{ratio_index:02d}",
            friction_mu=float(config["fixed_friction_mu"]),
            bddl_file=str(spec["bddl_file"]),
            box_name=str(spec["beam_name"]),
            max_steps=240,
            camera_resolution=int(config["camera_resolution"]),
            control_freq=float(config["fps"]),
        )
        env, constructor_sampler_retries = create_env_with_sampler_retries(
            case,
            seed=int(config["support_seed"]) + ratio_index,
        )
        try:
            obj = object_instance(env, str(spec["beam_name"]))
            body_id = int(env.inner_env.sim.model.body_name2id(obj.root_body))
            compiled_mass = float(env.inner_env.sim.model.body_mass[body_id])
            if abs(compiled_mass - float(spec["total_mass_kg"])) > 1e-6:
                raise RuntimeError(f"Compiled mass mismatch for ratio {ratio_index}: {compiled_mass} vs {spec['total_mass_kg']}")
            for entry in ratio_plan:
                beam_xy = base_xy + np.asarray(
                    [entry["beam_front_back_x_offset_m"], entry["beam_left_right_y_offset_m"]],
                    dtype=np.float64,
                )
                records, metrics = rollout_episode(
                    env,
                    beam_name=str(spec["beam_name"]),
                    support_offset_m=float(entry["sampled_support_offset_m"]),
                    beam_xy=beam_xy,
                    config=config,
                )
                episode_index = write_episode(
                    dataset,
                    records,
                    fps=int(config["fps"]),
                    jpeg_quality=int(config["jpeg_quality"]),
                )
                row = {
                    "episode_index": int(episode_index),
                    "dataset_split": str(spec["dataset_split"]),
                    "ratio_index": int(ratio_index),
                    "right_to_left_mass_ratio": float(spec["right_to_left_mass_ratio"]),
                    "left_end_mass_kg": float(spec["left_end_mass_kg"]),
                    "right_end_mass_kg": float(spec["right_end_mass_kg"]),
                    "crossbar_mass_kg": float(spec["crossbar_mass_kg"]),
                    "total_mass_kg": float(spec["total_mass_kg"]),
                    "theoretical_com_offset_m": float(spec["theoretical_com_offset_m"]),
                    "support_bin_index": int(entry["support_bin_index"]),
                    "support_bin_low_m": float(entry["support_bin_low_m"]),
                    "support_bin_high_m": float(entry["support_bin_high_m"]),
                    "sampled_support_offset_m": float(entry["sampled_support_offset_m"]),
                    "support_minus_com_m": float(entry["sampled_support_offset_m"] - spec["theoretical_com_offset_m"]),
                    "beam_base_xy_m": base_xy.tolist(),
                    "beam_front_back_x_offset_m": float(entry["beam_front_back_x_offset_m"]),
                    "beam_left_right_y_offset_m": float(entry["beam_left_right_y_offset_m"]),
                    "beam_initialized_xy_m": beam_xy.tolist(),
                    "action_type": "absolute_eef_setpoint_not_state",
                    "phase_segments": metrics["phase_segments"],
                    "discarded_constructor_sampler_attempts": int(constructor_sampler_retries),
                    "metrics": metrics,
                }
                rows.append(row)
                collected += 1
                autosave()
                print(
                    f"collect {collected:03d}/{len(plan):03d} ratio={spec['right_to_left_mass_ratio']:.4f} "
                    f"bin={entry['support_bin_index']:02d} support={entry['sampled_support_offset_m']:+.4f}m "
                    f"xy=({beam_xy[0]:+.3f},{beam_xy[1]:+.3f}) "
                    f"lift={metrics['max_beam_lift_m'] * 100:.1f}cm tilt={metrics['hold_max_beam_tilt_deg']:.1f}deg "
                    f"action_state_max={metrics['max_action_state_position_error_m'] * 100:.1f}cm",
                    flush=True,
                )
        finally:
            env.close()
    autosave()
    if args.max_episodes is None and collected != expected_total:
        raise RuntimeError(f"Collected {collected}/{expected_total} episodes")
    print(f"dataset={output}")
    print(f"episodes={collected}")


if __name__ == "__main__":
    main()

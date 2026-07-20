#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_UTILS_SCRIPT = REPO_ROOT / "scripts" / "render_libero_push_box_mass_oracle_force_and_lowkp_mu0040_2026-07-16_hai-machine.py"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pushbox" / "demos" / "libero_two_box_collision_mass_mu0020_2026-07-16_hai-machine"

FPS = 20
CAMERA_RESOLUTION = 224
FRICTION_MU = 0.02
PROJECTILE_NAME = "cream_cheese_1"
TARGET_NAME = "cream_cheese_2"
PROJECTILE_MASS_KG = 0.25
TARGET_MASSES_KG = (0.10, 0.25, 0.50, 1.00, 2.00)
PROJECTILE_INIT_XY = (-0.30, -0.035)
TARGET_INIT_XY = (0.05, -0.035)
CONTACT_Z = 0.915
BEHIND_OFFSET_M = 0.115
CONTROLLER_TRANSLATION_SCALE = 4.0
LAUNCH_PROFILE = np.asarray([0.5, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
BRAKE_PROFILE = np.asarray([-0.40, -0.40, 0.0, 0.0], dtype=np.float64)
CALIBRATION_AMPLITUDES = (0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.50)
TARGET_PREIMPACT_SPEED_MPS = 0.70


def load_pilot_utils() -> Any:
    spec = importlib.util.spec_from_file_location("two_box_collision_pilot_utils_hai_machine", PILOT_UTILS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot utilities: {PILOT_UTILS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = load_pilot_utils()
utils.FRICTION_MU = FRICTION_MU
utils.demo.FRICTION_MU = FRICTION_MU
demo = utils.demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fixed robot-action two-box collision demos across target masses.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def region(center: tuple[float, float], half_size: float = 0.002) -> tuple[float, float, float, float]:
    x, y = center
    return x - half_size, y - half_size, x + half_size, y + half_size


def write_two_box_bddl(output: Path) -> str:
    p = region(PROJECTILE_INIT_XY)
    t = region(TARGET_INIT_XY)
    g = region((0.34, -0.035), half_size=0.025)
    text = f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language launch the blue block into the orange block)
    (:regions
      (projectile_init_region
          (:target main_table)
          (:ranges (({p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]:.4f})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (mass_target_init_region
          (:target main_table)
          (:ranges (({t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {t[3]:.4f})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (dummy_goal_region
          (:target main_table)
          (:ranges (({g[0]:.4f} {g[1]:.4f} {g[2]:.4f} {g[3]:.4f})))
          (:rgba (0.0 0.0 0.0 0.0))
      )
    )

  (:fixtures
    main_table - table
  )

  (:objects
    cream_cheese_1 cream_cheese_2 - cream_cheese
  )

  (:obj_of_interest
    cream_cheese_1
    cream_cheese_2
  )

  (:init
    (On cream_cheese_1 main_table_projectile_init_region)
    (On cream_cheese_2 main_table_mass_target_init_region)
  )

  (:goal
    (And (On cream_cheese_2 main_table_dummy_goal_region))
  )
)
"""
    path = output / "bddl" / "two_box_collision_mass_mu0020.bddl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def make_case(*, bddl_file: str) -> Any:
    case = utils.make_case(amplitude=0.0, bddl_file=bddl_file)
    return replace(
        case,
        case_id="two_box_collision_mass_mu0020",
        friction_mu=FRICTION_MU,
        pusher_approach_steps=0,
        pusher_descend_steps=0,
        pusher_push_steps=10000,
        pusher_retreat_steps=0,
        pusher_settle_steps=0,
        pusher_push_controller_scale=1.0,
        pusher_max_push_controller_scale=1.0,
        pusher_push_controller_scale_ramp_steps=1,
        max_steps=10000,
    )


def object_instance(env: Any, name: str) -> Any:
    return env.inner_env.get_object(name)


def object_state(env: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = object_instance(env, name)
    qpos = np.asarray(env.inner_env.sim.data.get_joint_qpos(obj.joints[-1]), dtype=np.float64)
    qvel = np.asarray(env.inner_env.sim.data.get_joint_qvel(obj.joints[-1]), dtype=np.float64)
    return qpos[:3].copy(), qpos[3:7].copy(), qvel.copy()


def object_body_ids(env: Any, name: str) -> list[int]:
    model = env.inner_env.sim.model
    ids = []
    for body_id in range(int(model.nbody)):
        body_name = model.body_id2name(body_id) or ""
        if body_name == name or body_name.startswith(f"{name}_"):
            ids.append(int(body_id))
    if not ids:
        raise RuntimeError(f"No bodies found for {name}")
    return ids


def object_geom_ids(env: Any, name: str) -> set[int]:
    model = env.inner_env.sim.model
    ids: set[int] = set()
    for geom_id in range(int(model.ngeom)):
        geom_name = model.geom_id2name(geom_id) or ""
        if geom_name == name or geom_name.startswith(f"{name}_"):
            ids.add(int(geom_id))
    if not ids:
        raise RuntimeError(f"No geoms found for {name}")
    return ids


def set_object_mass(env: Any, name: str, mass_kg: float) -> dict[str, Any]:
    model = env.inner_env.sim.model
    ids = object_body_ids(env, name)
    native = float(sum(float(model.body_mass[body_id]) for body_id in ids))
    scale = float(mass_kg) / native
    for body_id in ids:
        model.body_mass[body_id] *= scale
        model.body_inertia[body_id] *= scale
    env.inner_env.sim.forward()
    return {"name": name, "mass_kg": float(mass_kg), "native_mass_kg": native, "mass_scale": scale}


def set_object_contact_properties(env: Any, name: str, *, rgba: tuple[float, float, float, float]) -> None:
    model = env.inner_env.sim.model
    for geom_id in object_geom_ids(env, name):
        model.geom_friction[geom_id] = np.asarray(
            [FRICTION_MU, float(env.case.geom_friction_spin), float(env.case.geom_friction_roll)],
            dtype=np.float64,
        )
        model.geom_rgba[geom_id] = np.asarray(rgba, dtype=np.float64)
        if hasattr(model, "geom_matid"):
            model.geom_matid[geom_id] = -1
    obj = object_instance(env, name)
    for joint_name in obj.joints:
        joint_id = model.joint_name2id(joint_name)
        start = int(model.jnt_dofadr[joint_id])
        model.dof_damping[start : start + 6] = float(env.case.joint_damping)


def contact_between(env: Any, first: set[int], second: set[int]) -> bool:
    sim = env.inner_env.sim
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & first and pair & second:
            return True
    return False


def robot_projectile_contact(env: Any, projectile_geoms: set[int]) -> bool:
    sim = env.inner_env.sim
    model = sim.model
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in projectile_geoms and geom2 not in projectile_geoms:
            continue
        other = geom2 if geom1 in projectile_geoms else geom1
        other_name = (model.geom_id2name(other) or "").lower()
        if "gripper" in other_name or "finger" in other_name or "robot" in other_name:
            return True
    return False


def move_to(env: Any, target_xyz: np.ndarray, *, steps: int, max_action: float) -> None:
    for _ in range(int(steps)):
        obs = env._last_obs
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        action = env._cartesian_action(
            eef,
            np.asarray(target_xyz, dtype=np.float64),
            float(env.case.pusher_gripper),
            max_action=float(max_action),
            position_gain=4.0,
        )
        env.step(action)


def establish_projectile_contact(env: Any, projectile_geoms: set[int]) -> int:
    projectile_xyz, _, _ = object_state(env, PROJECTILE_NAME)
    behind = np.asarray([projectile_xyz[0] - BEHIND_OFFSET_M, projectile_xyz[1], CONTACT_Z], dtype=np.float64)
    move_to(env, np.asarray([behind[0], behind[1], 1.04]), steps=35, max_action=0.20)
    move_to(env, behind, steps=70, max_action=0.30)
    target = np.asarray([projectile_xyz[0] + 0.08, projectile_xyz[1], CONTACT_Z], dtype=np.float64)
    for index in range(160):
        eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
        action = env._cartesian_action(
            eef,
            target,
            float(env.case.pusher_gripper),
            max_action=0.12,
            position_gain=4.0,
        )
        env.step(action)
        if robot_projectile_contact(env, projectile_geoms):
            return index + 1
    raise RuntimeError("Robot failed to establish contact with projectile block.")


def controller_translation_limits(env: Any) -> tuple[np.ndarray, np.ndarray]:
    controller = env._controller()
    return np.asarray(controller.output_min, dtype=np.float64).copy(), np.asarray(controller.output_max, dtype=np.float64).copy()


def set_controller_translation_scale(env: Any, base_limits: tuple[np.ndarray, np.ndarray], scale: float) -> None:
    controller = env._controller()
    output_min, output_max = (value.copy() for value in base_limits)
    output_min[:3] *= float(scale)
    output_max[:3] *= float(scale)
    controller.output_min = output_min
    controller.output_max = output_max
    controller.action_scale = None
    controller.action_output_transform = None
    controller.action_input_transform = None


def fixed_launch_actions(amplitude: float) -> np.ndarray:
    rows = []
    for scale in LAUNCH_PROFILE:
        action = np.zeros(7, dtype=np.float64)
        action[0] = float(amplitude * scale)
        action[-1] = 1.0
        rows.append(action)
    for value in BRAKE_PROFILE:
        action = np.zeros(7, dtype=np.float64)
        action[0] = float(value)
        action[-1] = 1.0
        rows.append(action)
    return np.asarray(rows, dtype=np.float64)


def action_hash(actions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(actions, dtype="<f8").tobytes()).hexdigest()


def render_frame(
    obs: dict[str, Any],
    *,
    target_mass_kg: float,
    phase: str,
    projectile_x_m: float,
    target_x_m: float,
    projectile_vx_mps: float,
    target_vx_mps: float,
) -> np.ndarray:
    return utils.frame_with_lines(
        obs,
        [
            f"blue projectile=250g | orange target={target_mass_kg * 1000.0:.0f}g | mu={FRICTION_MU:.2f}",
            f"same robot velocity action | {phase}",
            f"x=[{projectile_x_m:+.2f},{target_x_m:+.2f}]m vx=[{projectile_vx_mps:+.2f},{target_vx_mps:+.2f}]m/s",
        ],
    )


def rollout(
    *,
    target_mass_kg: float,
    amplitude: float,
    bddl_file: str,
    seed: int,
    capture_frames: bool,
) -> dict[str, Any]:
    env = demo.base.LiberoPushBoxEnv(make_case(bddl_file=bddl_file), repo_root=REPO_ROOT, seed=int(seed))
    frames: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    try:
        env.reset()
        projectile_mass = set_object_mass(env, PROJECTILE_NAME, PROJECTILE_MASS_KG)
        target_mass = set_object_mass(env, TARGET_NAME, float(target_mass_kg))
        set_object_contact_properties(env, PROJECTILE_NAME, rgba=(0.10, 0.35, 0.95, 1.0))
        set_object_contact_properties(env, TARGET_NAME, rgba=(0.95, 0.25, 0.05, 1.0))
        target_obj = object_instance(env, TARGET_NAME)
        env.inner_env.sim.data.set_joint_qvel(target_obj.joints[-1], np.zeros(6, dtype=np.float64))
        env.inner_env.sim.forward()

        projectile_geoms = object_geom_ids(env, PROJECTILE_NAME)
        target_geoms = object_geom_ids(env, TARGET_NAME)
        creep_steps = establish_projectile_contact(env, projectile_geoms)
        projectile_initial, _, _ = object_state(env, PROJECTILE_NAME)
        target_initial, _, _ = object_state(env, TARGET_NAME)
        base_limits = controller_translation_limits(env)
        set_controller_translation_scale(env, base_limits, CONTROLLER_TRANSLATION_SCALE)
        actions = fixed_launch_actions(amplitude)

        first_block_collision_frame = None
        last_robot_contact_frame = None
        preimpact_projectile_vx = None
        previous_projectile_vx = 0.0
        for frame_index, action in enumerate(actions):
            obs, _, _, _ = env.step(action)
            projectile_xyz, _, projectile_qvel = object_state(env, PROJECTILE_NAME)
            target_xyz, _, target_qvel = object_state(env, TARGET_NAME)
            robot_contact = robot_projectile_contact(env, projectile_geoms)
            blocks_contact = contact_between(env, projectile_geoms, target_geoms)
            if robot_contact:
                last_robot_contact_frame = frame_index
            if blocks_contact and first_block_collision_frame is None:
                first_block_collision_frame = frame_index
                preimpact_projectile_vx = previous_projectile_vx
            previous_projectile_vx = float(projectile_qvel[0])
            phase = "launch" if frame_index < len(LAUNCH_PROFILE) else "robot brake"
            timeline.append(
                {
                    "frame": frame_index,
                    "phase": phase,
                    "action": action,
                    "robot_projectile_contact": robot_contact,
                    "projectile_target_contact": blocks_contact,
                    "projectile_xyz_m": projectile_xyz,
                    "target_xyz_m": target_xyz,
                    "projectile_qvel": projectile_qvel,
                    "target_qvel": target_qvel,
                }
            )
            if capture_frames:
                frames.append(
                    render_frame(
                        obs,
                        target_mass_kg=target_mass_kg,
                        phase=phase,
                        projectile_x_m=float(projectile_xyz[0]),
                        target_x_m=float(target_xyz[0]),
                        projectile_vx_mps=float(projectile_qvel[0]),
                        target_vx_mps=float(target_qvel[0]),
                    )
                )

        set_controller_translation_scale(env, base_limits, 1.0)
        current_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
        lift_target = current_eef.copy()
        lift_target[2] += 0.14
        zero_action = np.zeros(7, dtype=np.float64)
        zero_action[-1] = 1.0
        for settle_index in range(150):
            if settle_index < 25:
                eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
                action = env._cartesian_action(eef, lift_target, 1.0, max_action=0.18, position_gain=4.0)
            else:
                action = zero_action
            obs, _, _, _ = env.step(action)
            projectile_xyz, _, projectile_qvel = object_state(env, PROJECTILE_NAME)
            target_xyz, _, target_qvel = object_state(env, TARGET_NAME)
            global_frame = len(actions) + settle_index
            robot_contact = robot_projectile_contact(env, projectile_geoms)
            blocks_contact = contact_between(env, projectile_geoms, target_geoms)
            if robot_contact:
                last_robot_contact_frame = global_frame
            if blocks_contact and first_block_collision_frame is None:
                first_block_collision_frame = global_frame
                preimpact_projectile_vx = previous_projectile_vx
            previous_projectile_vx = float(projectile_qvel[0])
            phase = "robot lift" if settle_index < 25 else "collision / settle"
            timeline.append(
                {
                    "frame": global_frame,
                    "phase": phase,
                    "action": action,
                    "robot_projectile_contact": robot_contact,
                    "projectile_target_contact": blocks_contact,
                    "projectile_xyz_m": projectile_xyz,
                    "target_xyz_m": target_xyz,
                    "projectile_qvel": projectile_qvel,
                    "target_qvel": target_qvel,
                }
            )
            if capture_frames:
                frames.append(
                    render_frame(
                        obs,
                        target_mass_kg=target_mass_kg,
                        phase=phase,
                        projectile_x_m=float(projectile_xyz[0]),
                        target_x_m=float(target_xyz[0]),
                        projectile_vx_mps=float(projectile_qvel[0]),
                        target_vx_mps=float(target_qvel[0]),
                    )
                )
        projectile_final, _, projectile_final_qvel = object_state(env, PROJECTILE_NAME)
        target_final, _, target_final_qvel = object_state(env, TARGET_NAME)
    finally:
        env.close()

    target_vx = np.asarray([float(row["target_qvel"][0]) for row in timeline], dtype=np.float64)
    projectile_vx = np.asarray([float(row["projectile_qvel"][0]) for row in timeline], dtype=np.float64)
    collision_after_robot_release = (
        first_block_collision_frame is not None
        and last_robot_contact_frame is not None
        and first_block_collision_frame > last_robot_contact_frame
    )
    return {
        "friction_mu": FRICTION_MU,
        "projectile_mass": projectile_mass,
        "target_mass": target_mass,
        "amplitude": float(amplitude),
        "controller_translation_scale": CONTROLLER_TRANSLATION_SCALE,
        "fixed_action_sequence": actions,
        "fixed_action_hash_sha256": action_hash(actions),
        "creep_setup_steps": creep_steps,
        "first_block_collision_frame": first_block_collision_frame,
        "last_robot_projectile_contact_frame": last_robot_contact_frame,
        "collision_after_robot_release": collision_after_robot_release,
        "preimpact_projectile_vx_mps": preimpact_projectile_vx,
        "max_projectile_vx_mps": float(np.max(projectile_vx)),
        "max_target_vx_mps": float(np.max(target_vx)),
        "projectile_initial_xyz_m": projectile_initial,
        "target_initial_xyz_m": target_initial,
        "projectile_final_xyz_m": projectile_final,
        "target_final_xyz_m": target_final,
        "projectile_displacement_m": float(np.linalg.norm(projectile_final[:2] - projectile_initial[:2])),
        "target_displacement_m": float(np.linalg.norm(target_final[:2] - target_initial[:2])),
        "projectile_final_qvel": projectile_final_qvel,
        "target_final_qvel": target_final_qvel,
        "timeline": timeline,
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
        utils.shutil.rmtree(output)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    bddl_file = write_two_box_bddl(output)

    calibration = []
    for index, amplitude in enumerate(CALIBRATION_AMPLITUDES):
        result = rollout(
            target_mass_kg=PROJECTILE_MASS_KG,
            amplitude=float(amplitude),
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=False,
        )
        calibration.append(utils.without_frames(result))
        print(
            f"[calibration {index + 1}/{len(CALIBRATION_AMPLITUDES)}] A={amplitude:.2f} "
            f"pre_v={result['preimpact_projectile_vx_mps']} "
            f"collision={result['first_block_collision_frame']} "
            f"robot_end={result['last_robot_projectile_contact_frame']} "
            f"clean={result['collision_after_robot_release']}",
            flush=True,
        )

    clean = [
        result
        for result in calibration
        if result["collision_after_robot_release"] and result["preimpact_projectile_vx_mps"] is not None
    ]
    if not clean:
        raise RuntimeError("No calibration action produced a clean block collision after robot release.")
    chosen = min(clean, key=lambda item: abs(float(item["preimpact_projectile_vx_mps"]) - TARGET_PREIMPACT_SPEED_MPS))
    chosen_amplitude = float(chosen["amplitude"])
    expected_actions = fixed_launch_actions(chosen_amplitude)
    expected_hash = action_hash(expected_actions)

    results = []
    for index, mass_kg in enumerate(TARGET_MASSES_KG):
        result = rollout(
            target_mass_kg=float(mass_kg),
            amplitude=chosen_amplitude,
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=True,
        )
        if result["fixed_action_hash_sha256"] != expected_hash or not np.array_equal(
            np.asarray(result["fixed_action_sequence"]), expected_actions
        ):
            raise RuntimeError(f"Robot action mismatch at target mass {mass_kg}kg")
        video = output / "videos" / f"{index + 1:02d}_target_mass_{int(round(mass_kg * 1000)):04d}g.mp4"
        utils.write_video(video, result["frames"])
        result["video"] = str(video)
        results.append(result)
        print(
            f"[mass {index + 1}/5] target={mass_kg:.2f}kg "
            f"pre_v={result['preimpact_projectile_vx_mps']} "
            f"target_vmax={result['max_target_vx_mps']:.3f}m/s "
            f"target_dx={result['target_displacement_m'] * 100.0:.1f}cm "
            f"clean={result['collision_after_robot_release']}",
            flush=True,
        )

    comparison = output / "videos" / "00_same_robot_action_two_box_mass_collision_comparison.mp4"
    utils.write_comparison(comparison, results)
    preimpact_speeds = [float(result["preimpact_projectile_vx_mps"]) for result in results]
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "artifact_type": "video_demo_only_not_a_training_dataset",
        "experiment": "fixed robot velocity action launches a fixed-mass projectile block into a variable-mass target block",
        "friction_mu": FRICTION_MU,
        "projectile_mass_kg": PROJECTILE_MASS_KG,
        "target_masses_kg": TARGET_MASSES_KG,
        "chosen_amplitude": chosen_amplitude,
        "controller_translation_scale": CONTROLLER_TRANSLATION_SCALE,
        "fixed_action_hash_sha256": expected_hash,
        "fixed_action_sequence": expected_actions,
        "all_action_sequences_exactly_equal": True,
        "preimpact_speed_range_mps": [min(preimpact_speeds), max(preimpact_speeds)],
        "preimpact_speed_span_mps": max(preimpact_speeds) - min(preimpact_speeds),
        "calibration": calibration,
        "results": [utils.without_frames(result) for result in results],
        "comparison_video": str(comparison),
    }
    utils.write_json(output / "summary.json", summary)
    print(f"chosen_A={chosen_amplitude:.2f}", flush=True)
    print(f"action_hash={expected_hash}", flush=True)
    print(f"preimpact_speed_span={summary['preimpact_speed_span_mps']:.6f}m/s", flush=True)
    print(f"comparison={comparison}", flush=True)


if __name__ == "__main__":
    main()

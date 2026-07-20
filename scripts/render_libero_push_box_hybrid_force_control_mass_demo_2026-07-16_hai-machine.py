#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
import importlib.util
from pathlib import Path
import sys
from types import MethodType
from typing import Any

import mujoco
import numpy as np
from robosuite.utils.control_utils import nullspace_torques, opspace_matrices, orientation_error


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_UTILS_SCRIPT = REPO_ROOT / "scripts" / "render_libero_push_box_mass_oracle_force_and_lowkp_mu0040_2026-07-16_hai-machine.py"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pushbox" / "demos" / "libero_push_box_hybrid_force_control_mass_mu0040_2026-07-16_hai-machine"

FRICTION_MU = 0.04
FPS = 20
MASS_TARGETS_KG = (0.25, 0.35, 0.45, 0.60, 0.75)
REFERENCE_MASS_KG = 0.45
TARGET_DISPLACEMENT_M = 0.40
FORCE_CALIBRATION_N = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
FORCE_PROFILE = np.asarray([0.5, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)

CONTACT_Z = 0.915
BEHIND_OFFSET_M = 0.115
CONTACT_CREEP_ACTION = 0.12
CONTACT_CREEP_MAX_STEPS = 140
CONTACT_FORCE_TRIGGER_N = 0.01
FORCE_FEEDBACK_GAIN = 12.0
FORCE_INTEGRAL_GAIN = 30.0
FORCE_FILTER_ALPHA = 0.20
FORCE_ERROR_INTEGRAL_LIMIT_NS = 0.50
FORCE_COMMAND_LIMIT_N = 80.0
HOLD_KP_YZ = 180.0
HOLD_KD_YZ = 2.0 * np.sqrt(HOLD_KP_YZ)
HOLD_KP_ORI = 25.0
HOLD_KD_ORI = 2.0 * np.sqrt(HOLD_KP_ORI)
BRAKE_DAMPING_X = 12.0
BRAKE_FORCE_LIMIT_N = 3.0


def load_pilot_utils() -> Any:
    spec = importlib.util.spec_from_file_location("mass_oracle_lowkp_utils_hai_machine", PILOT_UTILS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot utilities: {PILOT_UTILS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = load_pilot_utils()
demo = utils.demo
demo.FRICTION_MU = FRICTION_MU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render same-force, different-mass LIBERO robot pushing demos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_manual_case(*, bddl_file: str) -> Any:
    case = utils.make_case(amplitude=0.0, bddl_file=bddl_file)
    return replace(
        case,
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


def box_geom_ids(env: Any) -> set[int]:
    model = env.inner_env.sim.model
    obj = env.inner_env.get_object(env.case.box_name)
    ids: set[int] = set()
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or ""
        if name == obj.name or name.startswith(f"{obj.name}_"):
            ids.add(int(geom_id))
    if not ids:
        raise RuntimeError(f"No collision geoms found for {obj.name!r}")
    return ids


def robot_box_contact_force_N(env: Any, geom_ids: set[int]) -> float:
    sim = env.inner_env.sim
    model = sim.model
    total = 0.0
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in geom_ids and geom2 not in geom_ids:
            continue
        other = geom2 if geom1 in geom_ids else geom1
        other_name = (model.geom_id2name(other) or "").lower()
        if "table" in other_name or "target" in other_name or other in geom_ids:
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model._model, sim.data._data, contact_index, wrench)
        total += abs(float(wrench[0]))
    return total


def move_to(
    env: Any,
    target_xyz: np.ndarray,
    *,
    steps: int,
    max_action: float,
    frames: list[np.ndarray] | None,
    label: str,
    mass_kg: float,
) -> None:
    for _ in range(int(steps)):
        obs = env._last_obs if env._last_obs is not None else env._refresh_obs()
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        action = env._cartesian_action(
            eef,
            np.asarray(target_xyz, dtype=np.float64),
            float(env.case.pusher_gripper),
            max_action=float(max_action),
            position_gain=4.0,
        )
        obs, _, _, _ = env.step(action)
        if frames is not None:
            frames.append(
                utils.frame_with_lines(
                    obs,
                    [
                        f"hybrid force demo | mu={FRICTION_MU:.2f} mass={mass_kg * 1000.0:.0f}g",
                        label,
                    ],
                )
            )


def install_hybrid_force_controller(env: Any, *, geom_ids: set[int]) -> tuple[Any, dict[str, Any]]:
    controller = env._controller()
    if controller is None:
        raise RuntimeError("Robot controller is unavailable.")
    controller.update(force=True)
    original_run_controller = controller.run_controller
    state = {
        "mode": "force",
        "target_force_N": 0.0,
        "hold_yz": np.asarray(controller.ee_pos[1:3], dtype=np.float64).copy(),
        "hold_ori": np.asarray(controller.ee_ori_mat, dtype=np.float64).copy(),
        "null_joint": np.asarray(controller.joint_pos, dtype=np.float64).copy(),
        "contact_force_samples_N": [],
        "cartesian_force_command_samples_N": [],
        "eef_vx_samples_mps": [],
        "filtered_contact_force_N": 0.0,
        "force_error_integral_Ns": 0.0,
    }

    def run_hybrid(self: Any) -> np.ndarray:
        self.update()
        measured_force = robot_box_contact_force_N(env, geom_ids)
        if state["mode"] == "force":
            target_force = float(state["target_force_N"])
            filtered_force = (
                (1.0 - FORCE_FILTER_ALPHA) * float(state["filtered_contact_force_N"])
                + FORCE_FILTER_ALPHA * measured_force
            )
            state["filtered_contact_force_N"] = filtered_force
            force_error = target_force - filtered_force
            integral = float(state["force_error_integral_Ns"]) + force_error * float(self.model_timestep)
            integral = float(np.clip(integral, -FORCE_ERROR_INTEGRAL_LIMIT_NS, FORCE_ERROR_INTEGRAL_LIMIT_NS))
            state["force_error_integral_Ns"] = integral
            command_x = target_force + FORCE_FEEDBACK_GAIN * force_error + FORCE_INTEGRAL_GAIN * integral
            command_x = float(np.clip(command_x, 0.0, FORCE_COMMAND_LIMIT_N))
        else:
            state["force_error_integral_Ns"] = 0.0
            command_x = float(np.clip(-BRAKE_DAMPING_X * float(self.ee_pos_vel[0]), -BRAKE_FORCE_LIMIT_N, 0.0))

        yz_error = np.asarray(state["hold_yz"], dtype=np.float64) - np.asarray(self.ee_pos[1:3], dtype=np.float64)
        yz_force = HOLD_KP_YZ * yz_error - HOLD_KD_YZ * np.asarray(self.ee_pos_vel[1:3], dtype=np.float64)
        cartesian_force = np.asarray([command_x, yz_force[0], yz_force[1]], dtype=np.float64)
        ori_error = orientation_error(np.asarray(state["hold_ori"]), np.asarray(self.ee_ori_mat))
        cartesian_torque = HOLD_KP_ORI * ori_error - HOLD_KD_ORI * np.asarray(self.ee_ori_vel, dtype=np.float64)

        _, _, _, nullspace_matrix = opspace_matrices(
            self.mass_matrix,
            self.J_full,
            self.J_pos,
            self.J_ori,
        )
        torques = self.J_pos.T @ cartesian_force + self.J_ori.T @ cartesian_torque + self.torque_compensation
        torques += nullspace_torques(
            self.mass_matrix,
            nullspace_matrix,
            np.asarray(state["null_joint"]),
            self.joint_pos,
            self.joint_vel,
        )
        self.torques = np.asarray(torques, dtype=np.float64)
        self.new_update = True

        state["contact_force_samples_N"].append(measured_force)
        state["cartesian_force_command_samples_N"].append(command_x)
        state["eef_vx_samples_mps"].append(float(self.ee_pos_vel[0]))
        return self.torques

    controller.run_controller = MethodType(run_hybrid, controller)
    return original_run_controller, state


def restore_controller(env: Any, original_run_controller: Any) -> None:
    controller = env._controller()
    controller.run_controller = original_run_controller
    controller.update(force=True)
    controller.update_initial_joints(controller.joint_pos.copy())
    controller.reset_goal()


def clear_step_samples(state: dict[str, Any]) -> None:
    state["contact_force_samples_N"].clear()
    state["cartesian_force_command_samples_N"].clear()
    state["eef_vx_samples_mps"].clear()


def force_rollout(
    *,
    mass_kg: float,
    peak_force_N: float,
    bddl_file: str,
    seed: int,
    capture_frames: bool,
) -> dict[str, Any]:
    case = make_manual_case(bddl_file=bddl_file)
    env = demo.base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(seed))
    frames: list[np.ndarray] = []
    force_rows: list[dict[str, Any]] = []
    original_run_controller = None
    contact_found = False
    try:
        obs = env.reset()
        mass_info = demo.set_box_mass_kg(env, float(mass_kg))
        initial_scene_xyz, _ = env.box_pose()
        geom_ids = box_geom_ids(env)
        box_xyz = np.asarray(initial_scene_xyz, dtype=np.float64)
        behind = np.asarray([box_xyz[0] - BEHIND_OFFSET_M, box_xyz[1], CONTACT_Z], dtype=np.float64)
        move_to(
            env,
            np.asarray([behind[0], behind[1], 1.04]),
            steps=30,
            max_action=0.18,
            frames=frames if capture_frames else None,
            label="position approach above",
            mass_kg=mass_kg,
        )
        move_to(
            env,
            behind,
            steps=70,
            max_action=0.30,
            frames=frames if capture_frames else None,
            label="position descend behind box",
            mass_kg=mass_kg,
        )

        creep_target = np.asarray([box_xyz[0] + 0.08, box_xyz[1], CONTACT_Z], dtype=np.float64)
        for creep_index in range(CONTACT_CREEP_MAX_STEPS):
            current_obs = env._last_obs
            eef = np.asarray(current_obs["robot0_eef_pos"], dtype=np.float64)
            action = env._cartesian_action(
                eef,
                creep_target,
                float(case.pusher_gripper),
                max_action=CONTACT_CREEP_ACTION,
                position_gain=4.0,
            )
            obs, _, _, _ = env.step(action)
            contact_force = robot_box_contact_force_N(env, geom_ids)
            if capture_frames:
                frames.append(
                    utils.frame_with_lines(
                        obs,
                        [
                            f"hybrid force demo | mu={FRICTION_MU:.2f} mass={mass_kg * 1000.0:.0f}g",
                            f"slow contact approach | contact={contact_force:.3f}N",
                        ],
                    )
                )
            if contact_force >= CONTACT_FORCE_TRIGGER_N:
                contact_found = True
                break
        if not contact_found:
            raise RuntimeError(f"No robot-box contact for mass={mass_kg}kg")

        zero_action = np.zeros(7, dtype=np.float64)
        zero_action[-1] = float(case.pusher_gripper)
        obs, _, _, _ = env.step(zero_action)
        push_initial_xyz, _ = env.box_pose()
        controller = env._controller()
        controller.update(force=True)
        original_run_controller, force_state = install_hybrid_force_controller(env, geom_ids=geom_ids)

        for local_step, scale in enumerate(FORCE_PROFILE):
            clear_step_samples(force_state)
            force_state["mode"] = "force"
            force_state["target_force_N"] = float(scale * peak_force_N)
            obs, _, _, _ = env.step(zero_action)
            measured = np.asarray(force_state["contact_force_samples_N"], dtype=np.float64)
            commanded = np.asarray(force_state["cartesian_force_command_samples_N"], dtype=np.float64)
            eef_vx = np.asarray(force_state["eef_vx_samples_mps"], dtype=np.float64)
            current_box_xyz, _ = env.box_pose()
            displacement = float(np.linalg.norm(current_box_xyz[:2] - push_initial_xyz[:2]))
            row = {
                "local_step": local_step,
                "profile_scale": float(scale),
                "target_force_N": float(scale * peak_force_N),
                "measured_contact_force_mean_N": float(np.mean(measured)) if measured.size else 0.0,
                "measured_contact_force_max_N": float(np.max(measured)) if measured.size else 0.0,
                "cartesian_force_command_mean_N": float(np.mean(commanded)) if commanded.size else 0.0,
                "eef_vx_mean_mps": float(np.mean(eef_vx)) if eef_vx.size else 0.0,
                "box_displacement_m": displacement,
            }
            force_rows.append(row)
            if capture_frames:
                frames.append(
                    utils.frame_with_lines(
                        obs,
                        [
                            f"same robot force | mu={FRICTION_MU:.2f} mass={mass_kg * 1000.0:.0f}g",
                            f"F_target={row['target_force_N']:.2f}N F_contact={row['measured_contact_force_mean_N']:.2f}N",
                            f"force step={local_step + 1}/{len(FORCE_PROFILE)} box={displacement * 100.0:.1f}cm",
                        ],
                    )
                )

        for brake_step in range(4):
            clear_step_samples(force_state)
            force_state["mode"] = "brake"
            force_state["target_force_N"] = 0.0
            obs, _, _, _ = env.step(zero_action)
            if capture_frames:
                current_box_xyz, _ = env.box_pose()
                displacement = float(np.linalg.norm(current_box_xyz[:2] - push_initial_xyz[:2]))
                frames.append(
                    utils.frame_with_lines(
                        obs,
                        [
                            f"same robot force | mu={FRICTION_MU:.2f} mass={mass_kg * 1000.0:.0f}g",
                            f"brake and release {brake_step + 1}/4 | box={displacement * 100.0:.1f}cm",
                        ],
                    )
                )

        restore_controller(env, original_run_controller)
        original_run_controller = None
        current_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
        lift_target = current_eef.copy()
        lift_target[2] += 0.13
        move_to(
            env,
            lift_target,
            steps=20,
            max_action=0.16,
            frames=frames if capture_frames else None,
            label="position lift after force pulse",
            mass_kg=mass_kg,
        )
        for _ in range(100):
            obs, _, _, _ = env.step(zero_action)
            if capture_frames:
                current_box_xyz, _ = env.box_pose()
                displacement = float(np.linalg.norm(current_box_xyz[:2] - push_initial_xyz[:2]))
                frames.append(
                    utils.frame_with_lines(
                        obs,
                        [
                            f"same robot force | mu={FRICTION_MU:.2f} mass={mass_kg * 1000.0:.0f}g",
                            f"free slide and settle | box={displacement * 100.0:.1f}cm",
                        ],
                    )
                )
        final_xyz, _ = env.box_pose()
        final_qvel = env.box_velocity()
    finally:
        if original_run_controller is not None:
            try:
                restore_controller(env, original_run_controller)
            except Exception:
                pass
        env.close()

    target_impulse = float(peak_force_N * np.sum(FORCE_PROFILE) / FPS)
    measured_impulse = float(sum(row["measured_contact_force_mean_N"] / FPS for row in force_rows))
    return {
        "mode": "robot_hybrid_cartesian_force_control",
        "friction_mu": FRICTION_MU,
        "mass": mass_info,
        "peak_target_force_N": float(peak_force_N),
        "force_profile": FORCE_PROFILE,
        "target_impulse_Ns": target_impulse,
        "measured_contact_impulse_Ns": measured_impulse,
        "contact_found": contact_found,
        "precontact_box_motion_m": float(np.linalg.norm(push_initial_xyz[:2] - initial_scene_xyz[:2])),
        "push_initial_box_xyz_m": np.asarray(push_initial_xyz, dtype=np.float64),
        "final_box_xyz_m": np.asarray(final_xyz, dtype=np.float64),
        "final_displacement_m": float(np.linalg.norm(final_xyz[:2] - push_initial_xyz[:2])),
        "final_box_qvel": np.asarray(final_qvel, dtype=np.float64),
        "force_steps": force_rows,
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
    bddl_file = demo.base.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="hybrid_force_mass_mu0040_hidden",
        init_xy=demo.base.INIT_XY,
        target_xy=(
            float(demo.base.INIT_XY[0] + demo.base.DUMMY_TARGET_DISTANCE),
            float(demo.base.INIT_XY[1]),
        ),
        init_half_size=0.002,
        target_radius=demo.base.TARGET_RADIUS,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )

    calibration = []
    for index, force_N in enumerate(FORCE_CALIBRATION_N):
        result = force_rollout(
            mass_kg=REFERENCE_MASS_KG,
            peak_force_N=float(force_N),
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=False,
        )
        calibration.append(utils.without_frames(result))
        print(
            f"[force calibration {index + 1}/{len(FORCE_CALIBRATION_N)}] "
            f"F={force_N:.2f}N -> {result['final_displacement_m'] * 100.0:.1f}cm, "
            f"J_contact={result['measured_contact_impulse_Ns']:.3f}Ns",
            flush=True,
        )
    chosen = min(calibration, key=lambda item: abs(float(item["final_displacement_m"]) - TARGET_DISPLACEMENT_M))
    chosen_force_N = float(chosen["peak_target_force_N"])

    results = []
    for index, mass_kg in enumerate(MASS_TARGETS_KG):
        result = force_rollout(
            mass_kg=float(mass_kg),
            peak_force_N=chosen_force_N,
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=True,
        )
        video = output / "videos" / f"{index + 1:02d}_same_force_mass_{int(round(mass_kg * 1000)):04d}g.mp4"
        utils.write_video(video, result["frames"])
        result["video"] = str(video)
        results.append(result)
        print(
            f"[same force {index + 1}/5] mass={mass_kg:.2f}kg -> "
            f"{result['final_displacement_m'] * 100.0:.1f}cm, "
            f"J_target={result['target_impulse_Ns']:.3f}Ns, "
            f"J_contact={result['measured_contact_impulse_Ns']:.3f}Ns",
            flush=True,
        )

    comparison = output / "videos" / "00_same_robot_force_different_mass_comparison.mp4"
    utils.write_comparison(comparison, results)
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "artifact_type": "video_demo_only_not_a_training_dataset",
        "controller": "hybrid Cartesian force along world +x; position/impedance hold in y,z,orientation",
        "friction_mu": FRICTION_MU,
        "mass_targets_kg": MASS_TARGETS_KG,
        "reference_mass_kg": REFERENCE_MASS_KG,
        "target_calibration_displacement_m": TARGET_DISPLACEMENT_M,
        "force_calibration_N": FORCE_CALIBRATION_N,
        "chosen_peak_force_N": chosen_force_N,
        "force_profile": FORCE_PROFILE,
        "calibration": calibration,
        "same_force_results": [utils.without_frames(result) for result in results],
        "comparison_video": str(comparison),
    }
    utils.write_json(output / "summary.json", summary)
    print(f"chosen_peak_force_N={chosen_force_N:.2f}", flush=True)
    print(f"comparison={comparison}", flush=True)


if __name__ == "__main__":
    main()

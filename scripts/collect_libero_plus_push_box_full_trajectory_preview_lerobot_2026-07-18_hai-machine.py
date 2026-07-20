#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_plus_push_box_native_gripper_preview_lerobot_2026-07-18_hai-machine.py"
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
    / "libero_plus_official_scenes_full_trajectory_preview"
    / "libero_plus_push_box_official_scenes_full_trajectory_9eps_lerobot_2026-07-18_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module(SOURCE_SCRIPT, "libero_plus_native_preview_source_20260718_hai_machine")
legacy.CONFIG_PATH = CONFIG_PATH
legacy.DEFAULT_OUTPUT = DEFAULT_OUTPUT
legacy._NATIVE_PRESETS = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))[
    "background_randomization"
]["presets"]

base = legacy.base
ramp = legacy.ramp
touch = legacy.touch


def full_trajectory_rollout(
    case: Any,
    *,
    dataset: Any,
    action_cfg: dict[str, Any],
    variant: dict[str, Any],
    experiment: dict[str, Any],
    seed: int,
) -> tuple[int, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    angle_rad = math.radians(float(action_cfg["angle_deg"]))
    direction = np.asarray([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)

    env = touch.make_env(case, seed=int(seed))
    rows: list[dict[str, Any]] = []
    push_rows: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    contact_frames: list[int] = []
    contact_episode_count = 0
    contact_active = False
    frame_index = 0

    try:
        base_box_xyz, _ = env.box_pose()
        background = legacy.randomize_background(
            env,
            rng,
            experiment["background_randomization"],
        )
        object_state = legacy.set_object_variant_and_pose(
            env,
            variant=variant,
            rng=rng,
            base_init_xy=np.asarray(base_box_xyz[:2], dtype=np.float64),
            position_cfg=experiment["initial_position"],
            table_rgb=np.asarray(
                background["material_rgb"]["table_texture"], dtype=np.float64
            ),
            minimum_contrast=float(
                experiment["background_randomization"][
                    "minimum_object_table_rgb_distance"
                ]
            ),
        )
        env._last_obs = env._refresh_obs()
        settled_box_xyz, _ = env.box_pose()
        pusher_state = legacy.configure_directional_board(
            env,
            direction,
            experiment["pusher"],
        )

        base.remove_current_episode_images(dataset)
        episode_index = int(dataset.meta.total_episodes)
        task = base.prompt_for_case("observation", str(action_cfg["kind"]))
        fps = float(experiment["fps"])
        action_scale_m = float(
            experiment["absolute_action"][
                "translation_scale_m_per_normalized_unit"
            ]
        )
        recording = experiment["recording"]
        trajectory_cfg = experiment["trajectory_recording"]
        initial_eef_xyz = np.asarray(
            env._last_obs["robot0_eef_pos"], dtype=np.float64
        ).copy()

        def record_step(
            action: np.ndarray,
            *,
            phase_name: str,
            scripted_step_count: int,
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            nonlocal frame_index
            obs_for_frame = ramp.copy_obs(env._last_obs)
            state = np.asarray(base._obs_to_state(obs_for_frame), dtype=np.float32)
            stored_action = legacy.absolute_action(state, action, action_scale_m)
            eef_before = np.asarray(
                obs_for_frame["robot0_eef_pos"], dtype=np.float64
            ).copy()
            agent, wrist = base._obs_to_images(obs_for_frame)

            env.step_count = int(scripted_step_count)
            _, _, _, info = env.step(action)
            row = dict(info["push_box"])
            row.update(
                {
                    "frame_index": int(frame_index),
                    "phase": str(phase_name),
                    "eef_xyz_before_m": eef_before.astype(float).tolist(),
                    "env_action": np.asarray(action, dtype=float).tolist(),
                }
            )
            if extra:
                row.update(extra)
            rows.append(row)
            phase_counts[str(phase_name)] += 1

            frame = {
                "observation.images.image": agent,
                "observation.images.wrist_image": wrist,
                "observation.state": state,
                "action": stored_action,
            }
            dataset.add_frame(
                frame,
                task=task,
                timestamp=float(frame_index) / fps,
            )
            base.write_image_for_last_frame(
                dataset,
                "observation.images.image",
                frame_index,
                agent,
                jpeg_quality=int(recording["jpeg_quality"]),
            )
            base.write_image_for_last_frame(
                dataset,
                "observation.images.wrist_image",
                frame_index,
                wrist,
                jpeg_quality=int(recording["jpeg_quality"]),
            )
            frame_index += 1
            return row

        def move_to(
            target_xyz: np.ndarray,
            *,
            steps: int,
            max_action: float,
            gain: float,
            phase_name: str,
        ) -> float:
            for phase_step in range(int(steps)):
                eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
                action = np.zeros(7, dtype=np.float64)
                action[:3] = np.clip(
                    float(gain) * (target_xyz - eef),
                    -float(max_action),
                    float(max_action),
                )
                action[-1] = float(experiment["pusher"]["gripper_command"])
                record_step(
                    action,
                    phase_name=phase_name,
                    scripted_step_count=0,
                    extra={
                        "phase_step": int(phase_step),
                        "target_eef_xyz_m": target_xyz.astype(float).tolist(),
                    },
                )
            final_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
            return float(np.linalg.norm(target_xyz - final_eef))

        def calibrated_contact_z(reference_z: float) -> float:
            candidates = []
            for candidate_z in np.linspace(
                float(reference_z) - 0.16,
                float(reference_z) + 0.16,
                641,
                dtype=np.float64,
            ):
                try:
                    footprint = legacy._native_contact_footprint(
                        env,
                        direction,
                        float(candidate_z),
                    )
                except RuntimeError:
                    continue
                box_center_z = float(np.asarray(footprint["box_center"])[2])
                leading_center_z = float(
                    np.mean(
                        [
                            float(item["predicted_center_z_m"])
                            for item in footprint["leading_geoms"]
                        ]
                    )
                )
                candidates.append(
                    (
                        abs(float(candidate_z) - float(reference_z)),
                        abs(leading_center_z - box_center_z),
                        float(candidate_z),
                    )
                )
            if not candidates:
                raise RuntimeError(
                    f"No geometry-valid contact EEF height around z={reference_z:.4f} m"
                )
            return min(candidates, key=lambda item: item[:2])[2]

        pusher_cfg = dict(experiment["pusher"])
        target_z = float(case.pusher_contact_z)
        desired, _ = legacy._native_desired_eef(
            env,
            direction,
            lateral_offset_m=float(action_cfg["contact_lateral_offset_m"]),
            standoff_m=float(pusher_cfg["standoff_m"]),
            target_eef_z=target_z,
        )
        above = desired.copy()
        above[2] += float(pusher_cfg["approach_height_m"])
        approach_error_m = move_to(
            above,
            steps=int(pusher_cfg["approach_steps"]),
            max_action=float(trajectory_cfg["approach_max_action"]),
            gain=float(trajectory_cfg["approach_gain"]),
            phase_name="approach_fast",
        )

        target_z = calibrated_contact_z(target_z)
        desired, _ = legacy._native_desired_eef(
            env,
            direction,
            lateral_offset_m=float(action_cfg["contact_lateral_offset_m"]),
            standoff_m=float(pusher_cfg["standoff_m"]),
            target_eef_z=target_z,
        )
        descend_error_m = move_to(
            desired,
            steps=int(pusher_cfg["descend_steps"]),
            max_action=float(trajectory_cfg["descend_max_action"]),
            gain=float(trajectory_cfg["descend_gain"]),
            phase_name="descend_fast",
        )

        target_z = calibrated_contact_z(target_z)
        desired, _ = legacy._native_desired_eef(
            env,
            direction,
            lateral_offset_m=float(action_cfg["contact_lateral_offset_m"]),
            standoff_m=float(pusher_cfg["standoff_m"]),
            target_eef_z=target_z,
        )
        alignment_error_m = move_to(
            desired,
            steps=int(pusher_cfg["alignment_steps"]),
            max_action=float(trajectory_cfg["alignment_max_action"]),
            gain=float(trajectory_cfg["alignment_gain"]),
            phase_name="align",
        )

        final_target_z = calibrated_contact_z(target_z)
        if abs(final_target_z - target_z) > 0.00025:
            target_z = final_target_z
            desired, _ = legacy._native_desired_eef(
                env,
                direction,
                lateral_offset_m=float(action_cfg["contact_lateral_offset_m"]),
                standoff_m=float(pusher_cfg["standoff_m"]),
                target_eef_z=target_z,
            )
            alignment_error_m = move_to(
                desired,
                steps=max(4, int(pusher_cfg["alignment_steps"]) // 2),
                max_action=float(trajectory_cfg["alignment_max_action"]),
                gain=float(trajectory_cfg["alignment_gain"]),
                phase_name="align_height",
            )

        touch_start_eef = np.asarray(
            env._last_obs["robot0_eef_pos"], dtype=np.float64
        ).copy()
        touch_start_box, _ = env.box_pose()
        target_lateral = float(np.dot(desired[:2], lateral))
        touch_trigger = ""
        projected_box_motion_m = 0.0
        projected_box_speed_mps = 0.0
        touch_steps = 0
        for touch_steps in range(1, int(pusher_cfg["touch_max_steps"]) + 1):
            if legacy.board_box_contact(env):
                touch_trigger = "contact_before_step"
                break
            eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
            action = np.zeros(7, dtype=np.float64)
            lateral_error = target_lateral - float(np.dot(eef[:2], lateral))
            action[:2] = direction * float(pusher_cfg["touch_action"]) + lateral * float(
                np.clip(6.0 * lateral_error, -0.08, 0.08)
            )
            action[2] = float(np.clip(6.0 * (target_z - eef[2]), -0.08, 0.08))
            action[-1] = float(pusher_cfg["gripper_command"])
            record_step(
                action,
                phase_name="touch",
                scripted_step_count=0,
                extra={"phase_step": int(touch_steps - 1)},
            )
            box_xyz, _ = env.box_pose()
            projected_box_motion_m = float(
                np.dot(box_xyz[:2] - touch_start_box[:2], direction)
            )
            projected_box_speed_mps = float(np.dot(env.box_velocity()[:2], direction))
            if legacy.board_box_contact(env):
                touch_trigger = "contact_after_step"
                break
            if projected_box_motion_m >= float(
                pusher_cfg["contact_motion_threshold_m"]
            ):
                touch_trigger = "box_motion"
                break
            if projected_box_speed_mps >= float(
                pusher_cfg["contact_speed_threshold_mps"]
            ):
                touch_trigger = "box_speed"
                break
            touch_travel = float(
                np.dot(
                    np.asarray(env._last_obs["robot0_eef_pos"])[:2]
                    - touch_start_eef[:2],
                    direction,
                )
            )
            if touch_travel > float(pusher_cfg["touch_max_travel_m"]):
                break
        if not touch_trigger:
            raw_sim = env.inner_env.sim
            raw_box_id = legacy._native_box_geom_id(env)
            raw_eef = np.asarray(env._last_obs["robot0_eef_pos"], dtype=np.float64)
            raw_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            raw_geometry = {
                "target_z": float(target_z),
                "eef": raw_eef.astype(float).tolist(),
                "box_center": np.asarray(raw_sim.data.geom_xpos[raw_box_id], dtype=np.float64).tolist(),
                "box_z_support": legacy._geom_support(env, raw_box_id, raw_up),
                "gripper": [
                    {
                        "name": raw_sim.model.geom_id2name(geom_id),
                        "center": np.asarray(raw_sim.data.geom_xpos[geom_id], dtype=np.float64).tolist(),
                        "z_support": legacy._geom_support(env, geom_id, raw_up),
                    }
                    for geom_id in legacy._native_gripper_collision_ids(env)
                ],
            }
            raise RuntimeError(
                f"Native gripper touch failed after {touch_steps} steps; "
                f"geometry={raw_geometry}"
            )

        sim = env.inner_env.sim
        obj = env.inner_env.get_object(env.case.box_name)
        joint_name = obj.joints[-1]
        qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        latch_shift_m = 0.0
        search_step = float(pusher_cfg["latch_search_step_m"])
        search_count = int(
            round(float(pusher_cfg["latch_search_max_m"]) / search_step)
        )
        sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
        sim.forward()
        for _ in range(search_count + 1):
            if legacy.board_box_contact(env):
                break
            qpos[:2] -= direction * search_step
            latch_shift_m += search_step
            sim.data.set_joint_qpos(joint_name, qpos)
            sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
            sim.forward()
        if not legacy.board_box_contact(env):
            try:
                latch_geometry = legacy._native_touch_geometry(env, direction)
            except RuntimeError as error:
                latch_geometry = {"error": str(error)}
            latch_contacts = []
            for contact_index in range(int(sim.data.ncon)):
                contact = sim.data.contact[contact_index]
                latch_contacts.append(
                    [
                        sim.model.geom_id2name(int(contact.geom1)) or "",
                        sim.model.geom_id2name(int(contact.geom2)) or "",
                    ]
                )
            raise RuntimeError(
                f"Native gripper contact latch failed after shifting box "
                f"{latch_shift_m:.6f} m; touch_trigger={touch_trigger}; "
                f"projected_box_motion_m={projected_box_motion_m:.6f}; "
                f"projected_box_speed_mps={projected_box_speed_mps:.6f}; "
                f"geometry={latch_geometry}; contacts={latch_contacts}"
            )
        sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
        sim.forward()
        env._last_obs = env._refresh_obs()
        touch_state = {
            "touch_steps": int(touch_steps),
            "touch_trigger": touch_trigger,
            "projected_box_motion_before_latch_m": float(projected_box_motion_m),
            "projected_box_speed_before_latch_mps": float(
                projected_box_speed_mps
            ),
            "latch_box_shift_toward_gripper_m": float(latch_shift_m),
            "geometry": legacy._native_touch_geometry(env, direction),
            "launch_box_speed_mps": float(np.linalg.norm(env.box_velocity()[:2])),
        }

        launch_box_xyz, _ = env.box_pose()
        launch_eef = np.asarray(
            env._last_obs["robot0_eef_pos"], dtype=np.float64
        ).copy()
        previous_eef = launch_eef.copy()
        hold_z = float(launch_eef[2])
        hold_perpendicular = float(np.dot(launch_eef[:2], lateral))
        amplitude = float(action_cfg["A"])
        controller = experiment["controller"]
        target_projected_travel_m = float(
            action_cfg.get(
                "target_projected_travel_m",
                controller["target_projected_travel_m"],
            )
        )
        push_start_frame = int(frame_index)
        push_phase = "drive"
        brake_trigger_frame = None
        brake_frames = 0
        stop_mode = str(
            controller.get("stop_mode", "fixed_travel_latched_brake")
        )
        event_cfg = dict(controller.get("event_stop", {}))
        event_motion_frame = None
        event_stop_frame = None
        event_stop_reason = None

        for push_index in range(int(controller["recorded_steps"])):
            eef_before = np.asarray(
                env._last_obs["robot0_eef_pos"], dtype=np.float64
            ).copy()
            velocity = (
                (eef_before - previous_eef) * fps
                if push_index
                else np.zeros(3, dtype=np.float64)
            )
            previous_eef = eef_before.copy()
            projected_travel = float(
                np.dot(eef_before[:2] - launch_eef[:2], direction)
            )
            projected_v = float(np.dot(velocity[:2], direction))
            remaining = float(target_projected_travel_m - projected_travel)

            if (
                stop_mode == "fixed_travel_latched_brake"
                and push_phase == "drive"
                and push_index > 0
            ):
                lookahead = (
                    float(controller["brake_trigger_lookahead_frames"])
                    * max(0.0, projected_v)
                    / fps
                )
                if remaining <= lookahead:
                    push_phase = "brake"
                    brake_trigger_frame = int(push_index)
            if push_phase == "drive":
                command = (
                    amplitude * float(controller["first_frame_fraction"])
                    if push_index == 0
                    else amplitude
                )
            elif push_phase == "brake":
                if brake_frames >= int(controller["maximum_brake_frames"]) or (
                    brake_frames > 0
                    and projected_v <= float(controller["stop_speed_mps"])
                ):
                    push_phase = "locked_zero"
                    command = 0.0
                else:
                    command = float(
                        np.clip(
                            -float(controller["brake_gain_action_per_mps"])
                            * max(0.0, projected_v),
                            -amplitude,
                            0.0,
                        )
                    )
                    brake_frames += 1
            else:
                command = 0.0

            action = np.zeros(7, dtype=np.float64)
            if push_phase != "locked_zero":
                perpendicular_error = hold_perpendicular - float(
                    np.dot(eef_before[:2], lateral)
                )
                hold_limit = float(controller["hold_max_action"])
                perpendicular_action = float(
                    np.clip(
                        float(controller["perpendicular_hold_gain"])
                        * perpendicular_error,
                        -hold_limit,
                        hold_limit,
                    )
                )
                action[:2] = (
                    command * direction + perpendicular_action * lateral
                )
                action[2] = float(
                    np.clip(
                        float(controller["height_hold_gain"])
                        * (hold_z - eef_before[2]),
                        -hold_limit,
                        hold_limit,
                    )
                )
            action[:3] = np.clip(
                action[:3],
                -float(controller["pusher_max_pos_action"]),
                float(controller["pusher_max_pos_action"]),
            )
            action[-1] = float(pusher_cfg["gripper_command"])
            row = record_step(
                action,
                phase_name=f"push_{push_phase}",
                scripted_step_count=int(case.pusher_approach_steps)
                + int(case.pusher_descend_steps)
                + min(push_index, 15),
                extra={
                    "push_index": int(push_index),
                    "projected_eef_travel_before_m": projected_travel,
                    "projected_eef_velocity_before_mps": projected_v,
                    "remaining_before_m": remaining,
                    "target_projected_travel_m": target_projected_travel_m,
                    "command_along_direction": command,
                },
            )
            push_rows.append(row)
            has_contact = legacy.board_box_contact(env)
            if has_contact:
                contact_frames.append(int(push_index))
                if not contact_active:
                    contact_episode_count += 1
            contact_active = has_contact

            if stop_mode == "event_latched_zero" and push_phase == "drive":
                box_xyz_after = np.asarray(row["box_xyz"], dtype=np.float64)
                box_vxy_after = np.asarray(row["box_vxy"], dtype=np.float64)
                projected_box_motion = float(
                    np.dot(box_xyz_after[:2] - launch_box_xyz[:2], direction)
                )
                projected_box_velocity = float(np.dot(box_vxy_after, direction))
                motion_threshold = float(
                    event_cfg.get("contact_motion_threshold_m", 0.001)
                )
                speed_threshold = float(
                    event_cfg.get("contact_speed_threshold_mps", 0.03)
                )
                if event_motion_frame is None and (
                    projected_box_motion >= motion_threshold
                    or projected_box_velocity >= speed_threshold
                ):
                    event_motion_frame = int(push_index)
                if event_motion_frame is not None:
                    frames_after_motion = int(push_index) - int(event_motion_frame)
                    source_action = abs(
                        float(getattr(case, "pusher_push_action_end", amplitude))
                    )
                    release_speed = max(
                        speed_threshold,
                        float(event_cfg.get("trigger_velocity_ratio", 2.2))
                        * source_action,
                    )
                    velocity_ready = projected_box_velocity >= release_speed
                    timeout_ready = frames_after_motion >= int(
                        event_cfg.get("maximum_contact_hold_frames", 3)
                    )
                    minimum_hold_ready = frames_after_motion >= int(
                        event_cfg.get("minimum_contact_hold_frames", 0)
                    )
                    if minimum_hold_ready and (velocity_ready or timeout_ready):
                        push_phase = "locked_zero"
                        event_stop_frame = int(push_index + 1)
                        event_stop_reason = (
                            "velocity_ready" if velocity_ready else "hold_timeout"
                        )

        final_eef = np.asarray(
            env._last_obs["robot0_eef_pos"], dtype=np.float64
        ).copy()
        final_box_xyz, _ = env.box_pose()
        dataset.save_episode()
    finally:
        env.close()

    push_eef_positions = np.asarray(
        [row["eef_xyz_before_m"] for row in push_rows]
        + [final_eef.astype(float).tolist()],
        dtype=np.float64,
    )
    push_eef_delta = push_eef_positions[:, :2] - launch_eef[:2]
    projected = push_eef_delta @ direction
    perpendicular = push_eef_delta @ lateral
    box_delta = np.asarray(final_box_xyz[:2]) - np.asarray(launch_box_xyz[:2])
    box_velocity = np.asarray([row["box_vxy"] for row in push_rows], dtype=np.float64)
    projected_eef_velocity = np.asarray(
        [float(row["projected_eef_velocity_before_mps"]) for row in push_rows],
        dtype=np.float64,
    )
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
        "peak_box_projected_velocity_mps": float(
            np.max(box_velocity @ direction)
        ),
        "peak_projected_eef_velocity_mps": float(
            np.max(projected_eef_velocity)
        ),
        "drive_frame_count": int(
            sum(str(row["phase"]) == "push_drive" for row in push_rows)
        ),
        "maximum_projected_eef_travel_m": float(np.max(projected)),
        "final_projected_eef_travel_m": float(
            np.dot(final_eef[:2] - launch_eef[:2], direction)
        ),
        "maximum_absolute_perpendicular_eef_travel_m": float(
            np.max(np.abs(perpendicular))
        ),
        "brake_trigger_frame": brake_trigger_frame,
        "brake_frames": int(brake_frames),
        "stop_mode": stop_mode,
        "event_motion_frame": event_motion_frame,
        "event_stop_frame": event_stop_frame,
        "event_stop_reason": event_stop_reason,
        "contact_frames": contact_frames,
        "contact_episode_count": int(contact_episode_count),
        "background": background,
        "object": object_state,
        "board": pusher_state,
        "touch_preparation": touch_state,
        "full_trajectory": {
            "recorded_from_environment_reset": True,
            "total_recorded_frames": int(frame_index),
            "push_start_frame": int(push_start_frame),
            "phase_frame_counts": dict(phase_counts),
            "initial_eef_xyz_m": initial_eef_xyz.astype(float).tolist(),
            "approach_target_error_m": float(approach_error_m),
            "descend_target_error_m": float(descend_error_m),
            "alignment_target_error_m": float(alignment_error_m),
        },
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
            float(pusher_state["normal_alignment_error_deg"])
            <= float(gates["maximum_board_normal_error_deg"])
        ),
        "touch_reached": bool(
            touch_state["touch_steps"] <= int(pusher_cfg["touch_max_steps"])
        ),
        "approach_precision": bool(
            float(alignment_error_m)
            <= float(gates["maximum_alignment_target_error_m"])
        ),
    }
    if stop_mode == "event_latched_zero":
        checks["event_stop_triggered"] = event_stop_frame is not None
        checks["no_reverse_brake"] = int(brake_frames) == 0
    metrics["quality_checks"] = checks
    metrics["quality_pass"] = bool(all(checks.values()))
    return episode_index, metrics


legacy.rollout_preview = full_trajectory_rollout


if __name__ == "__main__":
    legacy.main()

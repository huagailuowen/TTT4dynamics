#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
LIBERO_REPO = REPO_ROOT.parent / "LIBERO"
for path in (REPO_ROOT, SCRIPTS_DIR, LIBERO_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_libero_push_box_adaptation_dataset import build_case, write_geometry_bddl  # noqa: E402
from ttt4dynamics.push_box_libero import LiberoPushBoxEnv  # noqa: E402


FPS = 20
CAMERA_RESOLUTION = 224
VIDEO_CRF = 18
FRICTION_MU = 0.1
CONTROLLER_SCALE = 4.0
INIT_XY = (-0.245, -0.035)
TARGET_XY = (0.255, -0.035)
CONTACT_Z = 0.915
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "pushbox" / "demos"
    / "libero_action_world_model_diverse_mu0100_2026-07-15_hai-machine"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render diverse fixed-friction LIBERO actions for action-conditioned world-model inspection."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def env_action_to_fastwam(action: np.ndarray) -> np.ndarray:
    converted = np.asarray(action, dtype=np.float32).copy()
    converted[-1] = (1.0 - converted[-1]) / 2.0
    return converted


def obs_image(obs: dict[str, Any], key: str) -> np.ndarray:
    return np.ascontiguousarray(obs[key][::-1, ::-1]).astype(np.uint8)


def normalized(direction_xy: np.ndarray | tuple[float, float]) -> np.ndarray:
    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("Direction must be non-zero.")
    return direction / norm


def smootherstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def make_writer(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", str(VIDEO_CRF)],
    )


def grasping(env: LiberoPushBoxEnv) -> bool:
    obj = env.inner_env.get_object(env.case.box_name)
    try:
        return bool(env.inner_env._check_grasp(gripper=env.inner_env.robots[0].gripper, object_geoms=obj))
    except Exception:
        return False


def robot_box_contact(env: LiberoPushBoxEnv) -> bool:
    model = env.inner_env.sim.model
    obj = env.inner_env.get_object(env.case.box_name)
    box_geom_ids: set[int] = set()
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or ""
        if name.startswith(f"{obj.name}_") or env.case.box_name in name:
            box_geom_ids.add(int(geom_id))
    for contact_index in range(int(env.inner_env.sim.data.ncon)):
        contact = env.inner_env.sim.data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in box_geom_ids and geom2 not in box_geom_ids:
            continue
        other = geom2 if geom1 in box_geom_ids else geom1
        if other in box_geom_ids:
            continue
        other_name = (model.geom_id2name(other) or "").lower()
        if "table" not in other_name and "target" not in other_name:
            return True
    return False


class DemoRecorder:
    def __init__(
        self,
        *,
        env: LiberoPushBoxEnv,
        demo_name: str,
        description: str,
        writer: Any,
        combined_writer: Any,
    ):
        self.env = env
        self.demo_name = demo_name
        self.description = description
        self.writer = writer
        self.combined_writer = combined_writer
        self.obs = env.reset()
        self.initial_box_xyz, _ = env.box_pose()
        self.rows: list[dict[str, Any]] = []
        self.frame_index = 0

    def _render(self, action: np.ndarray, phase: str, *, terminal: bool = False) -> np.ndarray:
        agent = obs_image(self.obs, "agentview_image")
        wrist = obs_image(self.obs, "robot0_eye_in_hand_image")
        canvas = Image.new("RGB", (CAMERA_RESOLUTION * 2, CAMERA_RESOLUTION + 32), (18, 18, 18))
        canvas.paste(Image.fromarray(agent), (0, 32))
        canvas.paste(Image.fromarray(wrist), (CAMERA_RESOLUTION, 32))
        box_xyz, _ = self.env.box_pose()
        displacement_cm = 100.0 * float(np.linalg.norm(box_xyz[:2] - self.initial_box_xyz[:2]))
        draw = ImageDraw.Draw(canvas)
        suffix = " terminal" if terminal else ""
        draw.text(
            (4, 2),
            f"{self.demo_name} | {phase}{suffix} | t={self.frame_index / FPS:5.2f}s",
            fill=(245, 245, 245),
        )
        draw.text(
            (4, 17),
            "cmd xyz=[{:+.3f},{:+.3f},{:+.3f}] grip={:+.0f} box_d={:.1f}cm".format(
                float(action[0]),
                float(action[1]),
                float(action[2]),
                float(action[-1]),
                displacement_cm,
            ),
            fill=(210, 225, 240),
        )
        return np.asarray(canvas, dtype=np.uint8)

    def step(self, action: np.ndarray, phase: str) -> None:
        command = np.asarray(action, dtype=np.float64).copy()
        command[:6] = np.clip(command[:6], -1.0, 1.0)
        command[-1] = float(np.clip(command[-1], -1.0, 1.0))

        eef_t = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
        quat_t = np.asarray(self.obs["robot0_eef_quat"], dtype=np.float64)
        gripper_t = np.asarray(self.obs["robot0_gripper_qpos"], dtype=np.float64)
        box_xyz_t, box_quat_t = self.env.box_pose()
        box_qvel_t = self.env.box_velocity()
        frame = self._render(command, phase)

        obs_tp1, _, _, _ = self.env.step(command)
        eef_tp1 = np.asarray(obs_tp1["robot0_eef_pos"], dtype=np.float64)
        quat_tp1 = np.asarray(obs_tp1["robot0_eef_quat"], dtype=np.float64)
        gripper_tp1 = np.asarray(obs_tp1["robot0_gripper_qpos"], dtype=np.float64)
        box_xyz_tp1, box_quat_tp1 = self.env.box_pose()
        box_qvel_tp1 = self.env.box_velocity()

        self.writer.append_data(frame)
        self.combined_writer.append_data(frame)
        self.rows.append(
            {
                "frame_index": int(self.frame_index),
                "timestamp_s": float(self.frame_index / FPS),
                "phase": phase,
                "alignment": "observation_t and action_t are recorded before env.step(action_t); observation_tp1 is recorded immediately after",
                "action_env_t": command,
                "action_fastwam_t": env_action_to_fastwam(command),
                "observation_t": {
                    "eef_xyz_m": eef_t,
                    "eef_quat_wxyz": quat_t,
                    "gripper_qpos": gripper_t,
                    "box_xyz_m": box_xyz_t,
                    "box_quat_wxyz": box_quat_t,
                    "box_qvel": box_qvel_t,
                },
                "observation_tp1": {
                    "eef_xyz_m": eef_tp1,
                    "eef_quat_wxyz": quat_tp1,
                    "gripper_qpos": gripper_tp1,
                    "box_xyz_m": box_xyz_tp1,
                    "box_quat_wxyz": box_quat_tp1,
                    "box_qvel": box_qvel_tp1,
                    "robot_box_contact": robot_box_contact(self.env),
                    "robosuite_grasping": grasping(self.env),
                },
                "realized_eef_delta_m": eef_tp1 - eef_t,
                "realized_eef_velocity_mps": (eef_tp1 - eef_t) * FPS,
                "realized_box_delta_m": box_xyz_tp1 - box_xyz_t,
            }
        )
        self.obs = obs_tp1
        self.frame_index += 1

    def terminal_frames(self, count: int = 8) -> None:
        action = np.zeros(7, dtype=np.float64)
        for _ in range(int(count)):
            frame = self._render(action, "final_observation", terminal=True)
            self.writer.append_data(frame)
            self.combined_writer.append_data(frame)

    def hold(self, steps: int, *, gripper: float, phase: str) -> None:
        for _ in range(int(steps)):
            action = np.zeros(7, dtype=np.float64)
            action[-1] = float(gripper)
            self.step(action, phase)

    def move_to(
        self,
        target_xyz: np.ndarray | tuple[float, float, float],
        *,
        steps: int,
        gripper: float,
        phase: str,
        gain: float = 3.5,
        max_action: float = 0.20,
    ) -> None:
        target = np.asarray(target_xyz, dtype=np.float64)
        for _ in range(int(steps)):
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
            action = np.zeros(7, dtype=np.float64)
            action[:3] = np.clip(float(gain) * (target - eef), -float(max_action), float(max_action))
            action[-1] = float(gripper)
            self.step(action, phase)

    def track_line(
        self,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        *,
        steps: int,
        gripper: float,
        phase: str,
        gain: float,
        max_action: float,
    ) -> None:
        start = np.asarray(start_xyz, dtype=np.float64)
        end = np.asarray(end_xyz, dtype=np.float64)
        for index in range(int(steps)):
            alpha = smootherstep(float(index + 1) / float(max(1, steps)))
            target = (1.0 - alpha) * start + alpha * end
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
            action = np.zeros(7, dtype=np.float64)
            action[:3] = np.clip(float(gain) * (target - eef), -float(max_action), float(max_action))
            action[-1] = float(gripper)
            self.step(action, phase)

    def directional_pulse(
        self,
        direction_xy: np.ndarray | tuple[float, float],
        amplitudes: list[float],
        *,
        line_xy: np.ndarray,
        contact_z: float,
        phase: str,
    ) -> None:
        direction = normalized(direction_xy)
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        for amplitude in amplitudes:
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64)
            lateral_error = float(np.dot(np.asarray(line_xy, dtype=np.float64) - eef[:2], lateral))
            action = np.zeros(7, dtype=np.float64)
            action[:2] = float(amplitude) * direction
            action[:2] += float(np.clip(3.0 * lateral_error, -0.10, 0.10)) * lateral
            action[2] = float(np.clip(3.0 * (float(contact_z) - eef[2]), -0.12, 0.12))
            action[-1] = 1.0
            self.step(action, phase)

    def contact_triggered_pulse(
        self,
        direction_xy: np.ndarray | tuple[float, float],
        *,
        peak: float,
        line_xy: np.ndarray,
        contact_z: float,
        max_precontact_steps: int,
        hold_after_contact: int,
        phase: str,
    ) -> None:
        contact_seen = False
        hold_remaining = int(hold_after_contact)
        for index in range(int(max_precontact_steps) + int(hold_after_contact)):
            amplitude = 0.5 * float(peak) if index == 0 else float(peak)
            self.directional_pulse(
                direction_xy,
                [amplitude],
                line_xy=line_xy,
                contact_z=contact_z,
                phase=phase,
            )
            row = self.rows[-1]
            box_delta = float(np.linalg.norm(row["observation_tp1"]["box_xyz_m"][:2] - self.initial_box_xyz[:2]))
            box_speed = float(np.linalg.norm(row["observation_tp1"]["box_qvel"][:2]))
            contact_now = bool(row["observation_tp1"]["robot_box_contact"])
            if not contact_seen and (contact_now or box_delta > 0.001 or box_speed > 0.03):
                contact_seen = True
            elif contact_seen:
                hold_remaining -= 1
                if hold_remaining <= 0:
                    break
        self.directional_pulse(
            direction_xy,
            [0.5 * float(peak), 0.0],
            line_xy=line_xy,
            contact_z=contact_z,
            phase=f"{phase}_brake",
        )

    def summary(self) -> dict[str, Any]:
        final_box_xyz, _ = self.env.box_pose()
        box_speeds = [float(np.linalg.norm(row["observation_tp1"]["box_qvel"][:2])) for row in self.rows]
        eef_speeds = [float(np.linalg.norm(row["realized_eef_velocity_mps"])) for row in self.rows]
        return {
            "demo_name": self.demo_name,
            "description": self.description,
            "transition_count": len(self.rows),
            "video_frame_count": len(self.rows) + 8,
            "initial_box_xyz_m": self.initial_box_xyz,
            "final_box_xyz_m": final_box_xyz,
            "final_box_displacement_m": float(np.linalg.norm(final_box_xyz[:2] - self.initial_box_xyz[:2])),
            "max_box_planar_speed_mps": max(box_speeds, default=0.0),
            "max_realized_eef_speed_mps": max(eef_speeds, default=0.0),
            "robot_box_contact_steps": int(
                sum(bool(row["observation_tp1"]["robot_box_contact"]) for row in self.rows)
            ),
            "robosuite_grasping_steps": int(
                sum(bool(row["observation_tp1"]["robosuite_grasping"]) for row in self.rows)
            ),
        }


def initial_hold(recorder: DemoRecorder, *, gripper: float = -1.0) -> None:
    recorder.hold(8, gripper=gripper, phase="initial_hold")


def approach_for_push(
    recorder: DemoRecorder,
    direction_xy: np.ndarray | tuple[float, float],
    *,
    offset_m: float,
) -> np.ndarray:
    direction = normalized(direction_xy)
    line_xy = recorder.initial_box_xyz[:2].copy()
    behind_xy = line_xy - direction * float(offset_m)
    recorder.move_to(
        (float(behind_xy[0]), float(behind_xy[1]), 1.04),
        steps=28,
        gripper=1.0,
        phase="approach_above",
        max_action=0.18,
    )
    recorder.move_to(
        (float(behind_xy[0]), float(behind_xy[1]), CONTACT_Z),
        steps=24,
        gripper=1.0,
        phase="descend_behind_box",
        max_action=0.13,
    )
    return behind_xy


def finish_push(recorder: DemoRecorder) -> None:
    recorder.hold(18, gripper=1.0, phase="observe_free_motion")
    eef = np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64)
    recorder.move_to(
        (float(eef[0]), float(eef[1]), 1.05),
        steps=18,
        gripper=1.0,
        phase="lift_clear",
        max_action=0.14,
    )
    recorder.hold(8, gripper=1.0, phase="final_settle")


def demo_hover_square(recorder: DemoRecorder) -> None:
    initial_hold(recorder)
    x, y = recorder.initial_box_xyz[:2]
    points = [
        (x - 0.12, y - 0.11, 1.10),
        (x + 0.08, y - 0.11, 1.10),
        (x + 0.08, y + 0.11, 1.10),
        (x - 0.12, y + 0.11, 1.10),
        (x - 0.12, y - 0.11, 1.10),
    ]
    for index, point in enumerate(points):
        recorder.move_to(point, steps=18, gripper=-1.0, phase=f"hover_square_edge_{index}", max_action=0.16)
    recorder.hold(8, gripper=-1.0, phase="hover_stop")


def demo_slow_position_push(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=1.0)
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    behind_xy = approach_for_push(recorder, direction, offset_m=0.115)
    start = np.asarray([behind_xy[0], behind_xy[1], CONTACT_Z], dtype=np.float64)
    end = start.copy()
    end[:2] += direction * 0.16
    recorder.track_line(
        start,
        end,
        steps=38,
        gripper=1.0,
        phase="slow_position_push",
        gain=2.8,
        max_action=0.075,
    )
    finish_push(recorder)


def demo_medium_tap(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=1.0)
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    approach_for_push(recorder, direction, offset_m=0.145)
    recorder.contact_triggered_pulse(
        direction,
        peak=0.28,
        line_xy=recorder.initial_box_xyz[:2],
        contact_z=CONTACT_Z,
        max_precontact_steps=24,
        hold_after_contact=3,
        phase="medium_tap",
    )
    finish_push(recorder)


def demo_fast_ram(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=1.0)
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    approach_for_push(recorder, direction, offset_m=0.16)
    recorder.contact_triggered_pulse(
        direction,
        peak=0.50,
        line_xy=recorder.initial_box_xyz[:2],
        contact_z=CONTACT_Z,
        max_precontact_steps=20,
        hold_after_contact=3,
        phase="fast_ram",
    )
    finish_push(recorder)


def demo_sustained_shove(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=1.0)
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    approach_for_push(recorder, direction, offset_m=0.14)
    recorder.directional_pulse(
        direction,
        [0.08, 0.16] + [0.22] * 16 + [0.14, 0.07, 0.0],
        line_xy=recorder.initial_box_xyz[:2],
        contact_z=CONTACT_Z,
        phase="sustained_shove",
    )
    finish_push(recorder)


def demo_lateral_push(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=1.0)
    direction = np.asarray([0.0, 1.0], dtype=np.float64)
    behind_xy = approach_for_push(recorder, direction, offset_m=0.14)
    start = np.asarray([behind_xy[0], behind_xy[1], CONTACT_Z], dtype=np.float64)
    end = start.copy()
    end[:2] += direction * 0.20
    recorder.track_line(
        start,
        end,
        steps=38,
        gripper=1.0,
        phase="lateral_push_positive_y",
        gain=3.0,
        max_action=0.11,
    )
    finish_push(recorder)


def demo_diagonal_push(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=1.0)
    angle = math.radians(30.0)
    direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    behind_xy = approach_for_push(recorder, direction, offset_m=0.15)
    start = np.asarray([behind_xy[0], behind_xy[1], CONTACT_Z], dtype=np.float64)
    end = start.copy()
    end[:2] += direction * 0.21
    recorder.track_line(
        start,
        end,
        steps=40,
        gripper=1.0,
        phase="diagonal_push_30deg",
        gain=3.0,
        max_action=0.12,
    )
    finish_push(recorder)


def demo_top_press(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=1.0)
    box = recorder.initial_box_xyz
    recorder.move_to(
        (float(box[0]), float(box[1]), 1.12),
        steps=32,
        gripper=1.0,
        phase="approach_above_box",
        max_action=0.18,
    )
    recorder.move_to(
        (float(box[0]), float(box[1]), float(box[2] + 0.055)),
        steps=28,
        gripper=1.0,
        phase="descend_to_top",
        max_action=0.12,
    )
    recorder.move_to(
        (float(box[0]), float(box[1]), float(box[2] - 0.020)),
        steps=24,
        gripper=1.0,
        phase="press_down",
        gain=2.5,
        max_action=0.08,
    )
    recorder.hold(12, gripper=1.0, phase="hold_pressure")
    eef = np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64)
    recorder.move_to((float(eef[0]), float(eef[1]), 1.12), steps=28, gripper=1.0, phase="release_press")


def demo_grasp_lift_place(recorder: DemoRecorder) -> None:
    initial_hold(recorder, gripper=-1.0)
    box = recorder.initial_box_xyz.copy()
    recorder.move_to(
        (float(box[0]), float(box[1]), 1.12),
        steps=35,
        gripper=-1.0,
        phase="grasp_approach",
        max_action=0.18,
    )
    recorder.move_to(
        (float(box[0]), float(box[1]), float(box[2] - 0.035)),
        steps=35,
        gripper=-1.0,
        phase="grasp_descend",
        max_action=0.12,
    )
    recorder.hold(28, gripper=1.0, phase="close_gripper")
    recorder.move_to(
        (float(box[0]), float(box[1]), 1.12),
        steps=35,
        gripper=1.0,
        phase="lift_box",
        max_action=0.16,
    )
    place_xy = np.asarray([float(box[0] + 0.10), float(box[1] + 0.10)], dtype=np.float64)
    recorder.move_to(
        (float(place_xy[0]), float(place_xy[1]), 1.12),
        steps=32,
        gripper=1.0,
        phase="carry_box",
        max_action=0.14,
    )
    recorder.move_to(
        (float(place_xy[0]), float(place_xy[1]), 0.99),
        steps=28,
        gripper=1.0,
        phase="lower_box",
        max_action=0.10,
    )
    recorder.hold(24, gripper=-1.0, phase="open_gripper_drop")
    eef = np.asarray(recorder.obs["robot0_eef_pos"], dtype=np.float64)
    recorder.move_to(
        (float(eef[0]), float(eef[1]), 1.12),
        steps=28,
        gripper=-1.0,
        phase="clear_after_drop",
        max_action=0.14,
    )


DEMO_SPECS: list[tuple[str, str, Callable[[DemoRecorder], None]]] = [
    ("hover_square", "No-contact square trajectory above and around the box.", demo_hover_square),
    ("slow_position_push", "Slow position-tracked push along +x.", demo_slow_position_push),
    ("medium_tap", "Short trapezoidal +x tap with a medium peak command.", demo_medium_tap),
    ("fast_ram", "Short high-peak +x ram capped at action amplitude 0.50.", demo_fast_ram),
    ("sustained_shove", "Longer constant-command shove to expose contact-rich dynamics.", demo_sustained_shove),
    ("lateral_push", "Push along +y from the side of the box.", demo_lateral_push),
    ("diagonal_push", "Push in the table plane at +30 degrees.", demo_diagonal_push),
    ("top_press", "Approach from above, press downward, hold, and release.", demo_top_press),
    ("grasp_lift_place", "Open, grasp, lift, translate, lower, and release the box.", demo_grasp_lift_place),
]


def title_card(name: str, description: str) -> np.ndarray:
    canvas = Image.new("RGB", (CAMERA_RESOLUTION * 2, CAMERA_RESOLUTION + 32), (14, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 108), name, fill=(245, 245, 245))
    draw.text((16, 128), description, fill=(190, 210, 230))
    return np.asarray(canvas, dtype=np.uint8)


def build_demo_case(bddl_file: str) -> Any:
    base = build_case(
        case_id="action_world_model_diverse_mu0100_demo",
        domain="action_world_model_demo",
        friction_group="mu0100",
        friction_mu=FRICTION_MU,
        geometry_id="fixed_classic_scene",
        init_xy=INIT_XY,
        target_distance=float(TARGET_XY[0] - INIT_XY[0]),
        bddl_file=bddl_file,
        target_radius=0.025,
        push_distance_x=0.10,
        max_steps=10000,
        camera_resolution=CAMERA_RESOLUTION,
    )
    return replace(
        base,
        target_xy=TARGET_XY,
        pusher_approach_steps=0,
        pusher_descend_steps=0,
        pusher_push_steps=10000,
        pusher_retreat_steps=0,
        pusher_settle_steps=0,
        pusher_push_controller_scale=CONTROLLER_SCALE,
        pusher_max_push_controller_scale=CONTROLLER_SCALE,
        pusher_push_controller_scale_ramp_steps=1,
        enable_controller_output_scaling=False,
        controller_output_scale=1.0,
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(parents=True, exist_ok=True)
    bddl_file = write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="action_world_model_diverse_mu0100_hidden",
        init_xy=INIT_XY,
        target_xy=TARGET_XY,
        init_half_size=0.002,
        target_radius=0.025,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )
    case = build_demo_case(bddl_file)
    combined_path = output / "videos" / "00_all_diverse_actions_mu0100.mp4"
    combined_writer = make_writer(combined_path)
    episodes: list[dict[str, Any]] = []
    try:
        for demo_index, (name, description, policy) in enumerate(DEMO_SPECS, start=1):
            card = title_card(name, description)
            for _ in range(FPS):
                combined_writer.append_data(card)
            video_path = output / "videos" / f"{demo_index:02d}_{name}.mp4"
            trajectory_path = output / "trajectories" / f"{demo_index:02d}_{name}.json"
            env = LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=int(args.seed))
            writer = make_writer(video_path)
            try:
                recorder = DemoRecorder(
                    env=env,
                    demo_name=name,
                    description=description,
                    writer=writer,
                    combined_writer=combined_writer,
                )
                policy(recorder)
                recorder.terminal_frames()
                summary = recorder.summary()
                trajectory = {
                    "demo_index": demo_index,
                    "demo_name": name,
                    "description": description,
                    "friction_mu": FRICTION_MU,
                    "controller_scale": CONTROLLER_SCALE,
                    "fps": FPS,
                    "camera_resolution_per_view": CAMERA_RESOLUTION,
                    "action_names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"],
                    "action_rotation_policy": "dax=day=daz=0 for this pilot",
                    "summary": summary,
                    "transitions": recorder.rows,
                }
                write_json(trajectory_path, trajectory)
                episode = {
                    **summary,
                    "video": str(video_path),
                    "trajectory": str(trajectory_path),
                }
                episodes.append(episode)
                print(
                    f"[{demo_index:02d}/{len(DEMO_SPECS):02d}] {name}: "
                    f"box={summary['final_box_displacement_m'] * 100.0:.1f}cm "
                    f"contact={summary['robot_box_contact_steps']} "
                    f"grasp={summary['robosuite_grasping_steps']} "
                    f"frames={summary['video_frame_count']}",
                    flush=True,
                )
            finally:
                writer.close()
                env.close()
    finally:
        combined_writer.close()

    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "artifact_type": "video_demo_only_not_a_training_dataset",
        "purpose": "Inspect diverse action-to-visual-transition coverage before collecting an action-conditioned world-model dataset.",
        "output_root": str(output),
        "combined_video": str(combined_path),
        "scene": {
            "simulator": "LIBERO / robosuite / MuJoCo",
            "object": "cream_cheese_1",
            "object_init_xy_m": INIT_XY,
            "friction_mu": FRICTION_MU,
            "target_visible": False,
            "camera_resolution_per_view": [CAMERA_RESOLUTION, CAMERA_RESOLUTION],
            "video_layout": "agentview and wrist view concatenated horizontally with a 32-pixel telemetry header",
            "fps": FPS,
            "video_crf": VIDEO_CRF,
        },
        "controller": {
            "action_space": "LIBERO OSC_POSE normalized 7D command",
            "action_names_env": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_command"],
            "action_names_fastwam": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"],
            "fixed_controller_scale": CONTROLLER_SCALE,
            "rotation_commands": [0.0, 0.0, 0.0],
            "gripper_conversion": "fastwam_gripper_open=(1-env_gripper_command)/2",
        },
        "alignment": {
            "transition": "observation_t + action_t -> observation_tp1",
            "record_order": [
                "read and render observation_t",
                "record exact action_t",
                "call env.step(action_t) once",
                "record observation_tp1 and realized EEF/object deltas",
            ],
            "future_lerobot_contract": "At a sampled start frame, FastWAM consumes 33 observations and the first 32 actions; action_t must be the command applied between observation_t and observation_t+1.",
        },
        "episodes": episodes,
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "demo_count": len(episodes),
            "combined_video": str(combined_path),
            "friction_mu": FRICTION_MU,
            "episodes": episodes,
        },
    )
    print(f"output={output}", flush=True)
    print(f"combined_video={combined_path}", flush=True)


if __name__ == "__main__":
    main()

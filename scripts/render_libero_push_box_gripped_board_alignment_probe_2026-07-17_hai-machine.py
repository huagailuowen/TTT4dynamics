#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_20fric_30peak_fixed16_event_tap_lerobot_2026-07-16_hai-machine.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "libero_push_box_gripped_board_alignment_probe_mu015_2026-07-17_hai-machine"
)
SOURCE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "pushbox"
    / "libero_push_box_20fric_30peak_fixed16_event_tap_hidden_lerobot_2026-07-16_hai-machine"
    / "manifest.json"
)

FRICTION_MU = 0.15
AMPLITUDES = (0.370, 0.400, 0.413, 0.428, 0.442, 0.457, 0.471, 0.486, 0.500)
PUSH_STEPS = 16
CAMERA_RESOLUTION = 224
FPS = 20
VIDEO_CRF = 18

# Coordinates are in the Panda EEF frame. At the calibrated push pose, EEF +y
# is world +x and EEF +x is world +y. The relative x-axis rotation removes the
# measured 3.28 degree pitch so the board face normal is world +x.
BOARD_POS_EEF = (0.0, 0.0470, 0.0030)
BOARD_QUAT_EEF_WXYZ = (0.999589, -0.028649, 0.0, 0.0)
BOARD_HALF_SIZE = (0.050, 0.004, 0.008)
BOARD_MASS_KG = 0.080
BOARD_RGBA = (0.95, 0.38, 0.08, 1.0)
HANDLE_POS_BOARD = (0.0, -0.0235, -0.0140)
HANDLE_HALF_SIZE = (0.004, 0.0195, 0.006)
HANDLE_RGBA = (0.32, 0.12, 0.035, 1.0)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source_dataset = load_module(SOURCE_SCRIPT, "board_probe_source_dataset_hai_machine")
source = source_dataset.source


def values(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def install_board_gripper() -> Any:
    import robosuite.models.grippers as grippers
    from robosuite.models.grippers.panda_gripper import PandaGripper

    original = grippers.GRIPPER_MAPPING["PandaGripper"]

    class PandaGripperWithBoard(PandaGripper):
        def __init__(self, idn: int | str = 0):
            super().__init__(idn=idn)
            eef_name = f"{self.naming_prefix}eef"
            eef_body = next(
                (body for body in self.worldbody.iter("body") if body.get("name") == eef_name),
                None,
            )
            if eef_body is None:
                raise RuntimeError(f"Unable to find Panda EEF body {eef_name}")
            board_body = ET.SubElement(
                eef_body,
                "body",
                {
                    "name": f"{self.naming_prefix}board_tool",
                    "pos": values(BOARD_POS_EEF),
                    "quat": values(BOARD_QUAT_EEF_WXYZ),
                },
            )
            ET.SubElement(
                board_body,
                "geom",
                {
                    "name": f"{self.naming_prefix}board_tool_collision",
                    "type": "box",
                    "size": values(BOARD_HALF_SIZE),
                    "mass": f"{BOARD_MASS_KG:.9g}",
                    "rgba": values(BOARD_RGBA),
                    "group": "0",
                    "contype": "2",
                    "conaffinity": "0",
                    "condim": "3",
                    "friction": "0.6 0.005 0.0001",
                    "solref": "0.01 1",
                    "solimp": "0.95 0.99 0.001",
                },
            )
            ET.SubElement(
                board_body,
                "geom",
                {
                    "name": f"{self.naming_prefix}board_tool_visual",
                    "type": "box",
                    "size": values(BOARD_HALF_SIZE),
                    "rgba": values(BOARD_RGBA),
                    "group": "1",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
            ET.SubElement(
                board_body,
                "geom",
                {
                    "name": f"{self.naming_prefix}board_handle_visual",
                    "type": "box",
                    "pos": values(HANDLE_POS_BOARD),
                    "size": values(HANDLE_HALF_SIZE),
                    "rgba": values(HANDLE_RGBA),
                    "group": "1",
                    "contype": "0",
                    "conaffinity": "0",
                    "mass": "0.001",
                },
            )

        @property
        def init_qpos(self) -> np.ndarray:
            # The episode starts with the fingers already closed around the
            # rigid visual handle; there is no grasp-acquisition phase.
            return np.asarray([0.0, 0.0], dtype=np.float64)

    grippers.GRIPPER_MAPPING["PandaGripper"] = PandaGripperWithBoard
    return original


def restore_board_gripper(original: Any) -> None:
    import robosuite.models.grippers as grippers

    grippers.GRIPPER_MAPPING["PandaGripper"] = original


def configure_board_collision(env: Any) -> None:
    """Make the rigid tool collide with the box, not its gripper or table."""
    sim = env.inner_env.sim
    model = sim.model
    board_id = model.geom_name2id("gripper0_board_tool_collision")
    model.geom_contype[board_id] = 2
    model.geom_conaffinity[board_id] = 0
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or ""
        if "cream_cheese" not in name:
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        model.geom_conaffinity[geom_id] = int(model.geom_conaffinity[geom_id]) | 2
    sim.forward()


def make_case(config: dict[str, Any], *, amplitude: float, bddl_file: str, action_id: int) -> Any:
    action_cfg = {
        "action_id": int(action_id),
        "A": float(amplitude),
        "push_steps": PUSH_STEPS,
    }
    case = source.base.build_fixed_case(
        config,
        mu=FRICTION_MU,
        action_cfg=action_cfg,
        case_id=f"board_mu1500_a{action_id:02d}_A{int(round(amplitude * 1000)):03d}",
        bddl_file=bddl_file,
        camera_resolution=CAMERA_RESOLUTION,
    )
    updated = replace(
        case,
        pusher_push_yz_hold_gain=8.0,
        pusher_push_yz_max_action=0.25,
        pusher_gripper=1.0,
    )
    # build_fixed_case stores the fixed action template as dynamic attributes;
    # dataclasses.replace only carries declared fields, so preserve them here.
    object.__setattr__(updated, "hai_action_id", getattr(case, "hai_action_id"))
    object.__setattr__(updated, "hai_action_profile", getattr(case, "hai_action_profile"))
    return updated


def board_state(env: Any, box_xyz: np.ndarray) -> dict[str, Any]:
    sim = env.inner_env.sim
    geom_id = sim.model.geom_name2id("gripper0_board_tool_collision")
    center = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    normal = rotation[:, 1].copy()
    if normal[0] < 0.0:
        normal *= -1.0
    normal /= np.linalg.norm(normal)
    alignment_deg = float(np.degrees(np.arccos(np.clip(normal[0], -1.0, 1.0))))
    return {
        "center_xyz": center,
        "normal_xyz": normal,
        "normal_error_deg": alignment_deg,
        "center_y_error_m": float(center[1] - box_xyz[1]),
        "center_z_error_m": float(center[2] - box_xyz[2]),
    }


def contact_state(env: Any) -> dict[str, Any]:
    sim = env.inner_env.sim
    model = sim.model
    pairs = []
    points = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        name1 = model.geom_id2name(int(contact.geom1)) or ""
        name2 = model.geom_id2name(int(contact.geom2)) or ""
        names = {name1, name2}
        if "gripper0_board_tool_collision" not in names:
            continue
        if not any("cream_cheese" in name for name in names):
            continue
        pairs.append([name1, name2])
        points.append(np.asarray(contact.pos, dtype=np.float64).astype(float).tolist())
    return {"pairs": pairs, "points": points}


def label_frame(frame: np.ndarray, *, amplitude: float, phase: str, displacement_m: float) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 222, 35), fill=(0, 0, 0))
    draw.text((6, 5), f"board pusher  mu={FRICTION_MU:.3f}  A={amplitude:.3f}", fill=(255, 255, 255))
    draw.text((6, 20), f"{phase}  displacement={displacement_m * 100.0:.1f}cm", fill=(255, 220, 170))
    return np.asarray(image, dtype=np.uint8)


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def rollout(config: dict[str, Any], *, amplitude: float, action_id: int, bddl_file: str, seed: int) -> dict[str, Any]:
    case = make_case(config, amplitude=amplitude, bddl_file=bddl_file, action_id=action_id)
    original_gripper = install_board_gripper()
    try:
        env = source.base.LiberoPushBoxEnv(case, repo_root=REPO_ROOT, seed=seed)
        # LIBERO creates and compiles the robosuite model lazily on reset, so
        # keep the custom gripper registered until the first reset completes.
        obs = env.reset()
        configure_board_collision(env)
    finally:
        restore_board_gripper(original_gripper)

    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    contact_frame = None
    forced_stop = False
    forced_stop_frame = None
    stop_reason = None
    alignment_before_contact = None
    board_contact_frames = []
    board_contact_points = []
    try:
        source.base.preposition_fixed_start(env)
        env.step_count = 0
        env._last_scripted_action = np.zeros(7, dtype=np.float64)
        env._last_scripted_phase = None
        obs = env._last_obs
        initial_xyz, _ = env.box_pose()
        init_x = float(initial_xyz[0])
        push_start = int(case.pusher_approach_steps) + int(case.pusher_descend_steps)
        for frame_index in range(int(case.max_steps)):
            if forced_stop:
                obs, _, _, info = env.step(source.brake_action(case))
                info["push_box"]["phase"] = "event_stop"
            else:
                obs, _, _, info = env.step()
            row = dict(info["push_box"])
            row["frame_index"] = frame_index
            rows.append(row)
            box_xyz = np.asarray(row["box_xyz"], dtype=np.float64)
            board = board_state(env, box_xyz)
            contacts = contact_state(env)
            if contacts["pairs"]:
                board_contact_frames.append(frame_index)
                board_contact_points.extend(contacts["points"])

            box_x = float(box_xyz[0])
            vx = float(row["box_vxy"][0])
            if contact_frame is None and ((box_x - init_x) > source.CONTACT_MOVE_M or abs(vx) > source.CONTACT_SPEED_MPS):
                contact_frame = frame_index
                alignment_before_contact = dict(board)
            if contact_frame is not None and not forced_stop:
                frames_after_contact = frame_index - int(contact_frame)
                target_vx = max(source.CONTACT_SPEED_MPS, source.TRIGGER_VX_RATIO * float(case.pusher_push_action_end))
                vx_ready = abs(vx) >= target_vx
                timeout_ready = frames_after_contact >= source.MAX_CONTACT_HOLD
                if frames_after_contact >= source.HOLD_AFTER_CONTACT and (vx_ready or timeout_ready):
                    forced_stop = True
                    forced_stop_frame = frame_index + 1
                    stop_reason = "vx_ready" if vx_ready else "timeout"

            agent, _ = source.base._obs_to_images(obs)
            phase = "event_stop" if forced_stop else str(row["phase"])
            displacement = float(np.linalg.norm(box_xyz[:2] - initial_xyz[:2]))
            frames.append(label_frame(agent, amplitude=amplitude, phase=phase, displacement_m=displacement))
    finally:
        env.close()

    final_xyz = np.asarray(rows[-1]["box_xyz"], dtype=np.float64)
    velocities = np.asarray([row["box_vxy"] for row in rows], dtype=np.float64)
    contact_points = np.asarray(board_contact_points, dtype=np.float64) if board_contact_points else np.zeros((0, 3))
    result = {
        "action_id": int(action_id),
        "A": float(amplitude),
        "friction_mu": FRICTION_MU,
        "push_steps": PUSH_STEPS,
        "initial_box_xyz_m": initial_xyz.astype(float).tolist(),
        "final_box_xyz_m": final_xyz.astype(float).tolist(),
        "final_displacement_m": float(np.linalg.norm(final_xyz[:2] - initial_xyz[:2])),
        "final_forward_m": float(final_xyz[0] - initial_xyz[0]),
        "final_lateral_m": float(final_xyz[1] - initial_xyz[1]),
        "peak_box_vx_mps": float(np.max(velocities[:, 0])),
        "contact_frame": contact_frame,
        "contact_local": None if contact_frame is None else int(contact_frame - push_start),
        "forced_stop_frame": forced_stop_frame,
        "stop_reason": stop_reason,
        "alignment_before_contact": {
            key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value
            for key, value in (alignment_before_contact or {}).items()
        },
        "sampled_board_contact_frames": board_contact_frames,
        "sampled_contact_y_error_m": None
        if not len(contact_points)
        else float(np.mean(contact_points[:, 1]) - initial_xyz[1]),
        "board": {
            "position_eef": list(BOARD_POS_EEF),
            "quaternion_eef_wxyz": list(BOARD_QUAT_EEF_WXYZ),
            "half_size_m": list(BOARD_HALF_SIZE),
            "mass_kg": BOARD_MASS_KG,
            "handle_position_board": list(HANDLE_POS_BOARD),
            "handle_half_size_m": list(HANDLE_HALF_SIZE),
            "initial_gripper_state": "closed around the handle at environment reset",
        },
        "frames": frames,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an aligned rigid board held by the Panda gripper.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = source_dataset.configure_dataset()
    prepare = dict(config["prepare_config"])
    prepare["descend_steps"] = 45
    prepare["prepare_position_gain"] = 8.0
    config["prepare_config"] = prepare
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    bddl_file = next(
        row["bddl_file"]
        for row in source_manifest["episodes"]
        if abs(float(row["mu"]) - FRICTION_MU) < 1e-12
    )

    summaries = []
    for action_id, amplitude in enumerate(AMPLITUDES):
        result = rollout(
            config,
            amplitude=float(amplitude),
            action_id=action_id,
            bddl_file=bddl_file,
            seed=int(args.seed),
        )
        frames = result.pop("frames")
        video_path = output_root / f"board_mu1500_a{action_id:02d}_A{int(round(amplitude * 1000)):03d}.mp4"
        write_video(video_path, frames)
        result["video"] = str(video_path)
        summaries.append(result)
        alignment = result["alignment_before_contact"]
        print(
            f"board {action_id + 1:02d}/{len(AMPLITUDES):02d} A={amplitude:.3f} "
            f"disp={result['final_displacement_m'] * 100.0:.1f}cm "
            f"peak_vx={result['peak_box_vx_mps']:.3f} "
            f"lateral={result['final_lateral_m'] * 100.0:.2f}cm "
            f"normal_err={alignment.get('normal_error_deg')}deg "
            f"y_err={alignment.get('center_y_error_m')}",
            flush=True,
        )

    payload = {
        "experiment": "rigid rectangular board visually held by closed Panda gripper",
        "source_script": str(SOURCE_SCRIPT),
        "friction_mu": FRICTION_MU,
        "amplitudes": list(AMPLITUDES),
        "results": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(source.base.to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"summary={output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

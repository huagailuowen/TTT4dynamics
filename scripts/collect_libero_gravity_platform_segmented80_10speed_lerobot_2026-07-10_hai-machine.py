#!/usr/bin/env python3
"""Calibrate and collect the 80-gravity x 10-speed LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = REPO_ROOT.parent / "FastWAM-TTT"
FASTWAM_SRC = FASTWAM_ROOT / "src"
DEMO_SCRIPT = REPO_ROOT / "scripts" / "render_libero_gravity_platform_demos_hai-machine.py"
DEFAULT_OUTPUT = (
    FASTWAM_ROOT
    / "data"
    / "libero_gravity"
    / "libero_gravity_platform_segmented80_10speed_hidden_lerobot_2026-07-10_hai-machine"
)
for path in (REPO_ROOT, FASTWAM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.lerobot.lerobot import lerobot_dataset as lerobot_dataset_module
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
)


GRAVITY_COUNT = 80
SPEED_COUNT = 10
GRAVITY_MIN_MPS2 = 1.0
GRAVITY_MAX_MPS2 = 50.0
SPEED_MIN_MPS = 0.48
SPEED_MAX_MPS = 0.96
FPS = 20
CAMERA_RESOLUTION = 224
VIDEO_CODEC = "h264"
VIDEO_CRF = 18
JPEG_QUALITY = 98
PRE_ROLL_S = 0.40
POST_CONTACT_FRAMES = 6
MAX_EPISODE_S = 3.20
MIN_LANDING_X_M = 0.040
MAX_LANDING_X_M = 0.690
MAX_ABS_Y_M = 0.005
EDGE_SPEED_RATIO_RANGE = (0.97, 1.03)
MAX_THEORY_ERROR_M = 0.030


def _load_demo_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "libero_gravity_platform_demo_hai_machine", DEMO_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


demo = _load_demo_module()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonable(row), separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def gravity_grid() -> np.ndarray:
    inverse_sqrt = np.linspace(
        1.0 / np.sqrt(GRAVITY_MIN_MPS2),
        1.0 / np.sqrt(GRAVITY_MAX_MPS2),
        GRAVITY_COUNT,
        dtype=np.float64,
    )
    return 1.0 / np.square(inverse_sqrt)


def speed_grid() -> np.ndarray:
    return np.linspace(
        SPEED_MIN_MPS, SPEED_MAX_MPS, SPEED_COUNT, dtype=np.float64
    )


def _scene_ids(model: mujoco.MjModel) -> dict[str, int]:
    names = {
        "cube_joint": (mujoco.mjtObj.mjOBJ_JOINT, "cube_free"),
        "cube_body": (mujoco.mjtObj.mjOBJ_BODY, "cube"),
        "cube_geom": (mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
        "table_geom": (mujoco.mjtObj.mjOBJ_GEOM, "tabletop"),
    }
    result = {
        key: int(mujoco.mj_name2id(model, object_type, name))
        for key, (object_type, name) in names.items()
    }
    if any(value < 0 for value in result.values()):
        raise RuntimeError(f"Missing MuJoCo scene ids: {result}")
    return result


def _has_contact(data: mujoco.MjData, geom_a: int, geom_b: int) -> bool:
    target = {int(geom_a), int(geom_b)}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} == target:
            return True
    return False


def _theoretical_landing_x(gravity_mps2: float, speed_mps: float) -> float:
    fall_height = float(demo.PLATFORM_TOP_Z_M - demo.TABLE_TOP_Z_M)
    return float(
        demo.PLATFORM_EDGE_X_M
        + speed_mps * np.sqrt(2.0 * fall_height / gravity_mps2)
    )


def simulate_case(model: mujoco.MjModel, gravity_mps2: float, speed_mps: float) -> dict[str, Any]:
    data = mujoco.MjData(model)
    ids = _scene_ids(model)
    dof_address = int(model.jnt_dofadr[ids["cube_joint"]])
    mujoco.mj_forward(model, data)

    while data.time < PRE_ROLL_S:
        mujoco.mj_step(model, data)
    data.qvel[dof_address] = float(speed_mps)
    launch_time = float(data.time)
    edge_time = None
    edge_speed = None
    contact_time = None
    contact_x = None
    max_abs_y = 0.0
    while data.time - launch_time <= MAX_EPISODE_S:
        mujoco.mj_step(model, data)
        cube_position = np.asarray(data.xpos[ids["cube_body"]], dtype=np.float64)
        max_abs_y = max(max_abs_y, abs(float(cube_position[1])))
        if edge_time is None and float(cube_position[0]) >= demo.PLATFORM_EDGE_X_M:
            edge_time = float(data.time)
            edge_speed = float(data.qvel[dof_address])
        if _has_contact(data, ids["cube_geom"], ids["table_geom"]):
            contact_time = float(data.time)
            contact_x = float(cube_position[0])
            break

    theory_x = _theoretical_landing_x(gravity_mps2, speed_mps)
    edge_ratio = None if edge_speed is None else float(edge_speed / speed_mps)
    theory_error = None if contact_x is None else float(contact_x - theory_x)
    checks = {
        "crossed_platform_edge": edge_time is not None,
        "landed_on_table": contact_time is not None,
        "landing_in_safe_bounds": bool(
            contact_x is not None and MIN_LANDING_X_M <= contact_x <= MAX_LANDING_X_M
        ),
        "edge_speed_preserved": bool(
            edge_ratio is not None
            and EDGE_SPEED_RATIO_RANGE[0] <= edge_ratio <= EDGE_SPEED_RATIO_RANGE[1]
        ),
        "lateral_drift_small": bool(max_abs_y <= MAX_ABS_Y_M),
        "theory_residual_small": bool(
            theory_error is not None and abs(theory_error) <= MAX_THEORY_ERROR_M
        ),
    }
    return {
        "gravity_mps2": float(gravity_mps2),
        "initial_speed_mps": float(speed_mps),
        "launch_time_s": launch_time,
        "edge_crossing_time_after_launch_s": None
        if edge_time is None
        else float(edge_time - launch_time),
        "edge_speed_mps": edge_speed,
        "edge_speed_ratio": edge_ratio,
        "first_table_contact_time_after_launch_s": None
        if contact_time is None
        else float(contact_time - launch_time),
        "first_table_contact_x_m": contact_x,
        "theoretical_landing_x_m": theory_x,
        "theory_error_m": theory_error,
        "max_abs_y_m": float(max_abs_y),
        "checks": checks,
        "quality_pass": bool(all(checks.values())),
    }


def _monotonicity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = np.full((GRAVITY_COUNT, SPEED_COUNT), np.nan, dtype=np.float64)
    for row in rows:
        matrix[int(row["gravity_index"]), int(row["speed_index"])] = float(
            row["first_table_contact_x_m"]
        )
    speed_diffs = np.diff(matrix, axis=1)
    gravity_diffs = np.diff(matrix, axis=0)
    return {
        "landing_increases_with_speed": bool(np.all(speed_diffs > 0.0)),
        "landing_decreases_with_gravity": bool(np.all(gravity_diffs < 0.0)),
        "minimum_adjacent_speed_delta_m": float(np.min(speed_diffs)),
        "maximum_adjacent_speed_delta_m": float(np.max(speed_diffs)),
        "minimum_adjacent_gravity_drop_m": float(np.min(-gravity_diffs)),
        "maximum_adjacent_gravity_drop_m": float(np.max(-gravity_diffs)),
    }


def run_calibration(output_root: Path) -> dict[str, Any]:
    gravities = gravity_grid()
    speeds = speed_grid()
    rows = []
    for gravity_index, gravity in enumerate(gravities):
        model = mujoco.MjModel.from_xml_string(demo._mjcf(float(gravity)))
        for speed_index, speed in enumerate(speeds):
            row = simulate_case(model, float(gravity), float(speed))
            row["gravity_index"] = gravity_index
            row["speed_index"] = speed_index
            row["case_id"] = f"gravity_g{gravity_index:02d}_v{speed_index:02d}"
            rows.append(row)
        if (gravity_index + 1) % 10 == 0:
            print(f"calibrate gravity {gravity_index + 1:02d}/{GRAVITY_COUNT}", flush=True)

    monotonicity = _monotonicity(rows)
    failed = [row for row in rows if not row["quality_pass"]]
    quality_pass = bool(not failed and all(value for key, value in monotonicity.items() if isinstance(value, bool)))
    contact_x = np.asarray([row["first_table_contact_x_m"] for row in rows], dtype=np.float64)
    theory_error = np.asarray([row["theory_error_m"] for row in rows], dtype=np.float64)
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "calibration_type": "libero_gravity_platform_segmented80_10speed_hai-machine",
        "quality_pass": quality_pass,
        "gravity_sampling": "80 values uniformly spaced in 1/sqrt(g), producing approximately uniform ballistic-range changes",
        "gravity_count": GRAVITY_COUNT,
        "gravity_min_mps2": float(gravities[0]),
        "gravity_max_mps2": float(gravities[-1]),
        "speed_count": SPEED_COUNT,
        "speed_min_mps": float(speeds[0]),
        "speed_max_mps": float(speeds[-1]),
        "combination_count": len(rows),
        "failed_combination_count": len(failed),
        "landing_x_range_m": [float(np.min(contact_x)), float(np.max(contact_x))],
        "max_abs_theory_error_m": float(np.max(np.abs(theory_error))),
        "monotonicity": monotonicity,
        "quality_limits": {
            "landing_x_m": [MIN_LANDING_X_M, MAX_LANDING_X_M],
            "max_abs_y_m": MAX_ABS_Y_M,
            "edge_speed_ratio": list(EDGE_SPEED_RATIO_RANGE),
            "max_abs_theory_error_m": MAX_THEORY_ERROR_M,
        },
        "failed_cases": failed,
    }
    calibration_dir = output_root / "calibration"
    _write_json(calibration_dir / "calibration_summary.json", summary)
    _write_json(calibration_dir / "gravity_speed_parameter_table.json", rows)
    csv_path = calibration_dir / "gravity_speed_parameter_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "case_id",
            "gravity_index",
            "gravity_mps2",
            "speed_index",
            "initial_speed_mps",
            "edge_speed_mps",
            "edge_speed_ratio",
            "edge_crossing_time_after_launch_s",
            "first_table_contact_time_after_launch_s",
            "first_table_contact_x_m",
            "theoretical_landing_x_m",
            "theory_error_m",
            "max_abs_y_m",
            "quality_pass",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    config = {
        "dataset_name": "libero_gravity_platform_segmented80_10speed_hidden_lerobot_2026-07-10_hai-machine",
        "lerobot_codebase_version": CODEBASE_VERSION,
        "gravities_mps2": gravities.tolist(),
        "initial_speeds_mps": speeds.tolist(),
        "gravity_count": GRAVITY_COUNT,
        "speed_count": SPEED_COUNT,
        "expected_episode_count": GRAVITY_COUNT * SPEED_COUNT,
        "fps": FPS,
        "camera_resolution": CAMERA_RESOLUTION,
        "pre_roll_s": PRE_ROLL_S,
        "post_contact_frames": POST_CONTACT_FRAMES,
        "max_episode_s": MAX_EPISODE_S,
        "scene": {
            "cube_color": "blue",
            "cube_mass_kg": demo.CUBE_MASS_KG,
            "cube_half_size_m": demo.CUBE_HALF_SIZE_M,
            "platform_top_z_m": demo.PLATFORM_TOP_Z_M,
            "table_top_z_m": demo.TABLE_TOP_Z_M,
            "platform_edge_x_m": demo.PLATFORM_EDGE_X_M,
            "cube_start_x_m": demo.CUBE_START_X_M,
            "platform_friction": [0.0002, 0.0001, 0.0001],
        },
        "calibration_summary": summary,
    }
    _write_json(calibration_dir / "selected_config.json", config)
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    if not quality_pass:
        raise RuntimeError(
            f"Calibration rejected {len(failed)} combinations or monotonicity failed; refusing collection"
        )
    return config


def _camera_main() -> mujoco.MjvCamera:
    return demo._camera()


def _camera_auxiliary() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([0.04, 0.0, 0.12], dtype=np.float64)
    camera.distance = 1.20
    camera.azimuth = 92.0
    camera.elevation = -20.0
    return camera


def _cube_state(model: mujoco.MjModel, data: mujoco.MjData, ids: dict[str, int]) -> np.ndarray:
    qpos_address = int(model.jnt_qposadr[ids["cube_joint"]])
    dof_address = int(model.jnt_dofadr[ids["cube_joint"]])
    return np.concatenate(
        [
            np.asarray(data.qpos[qpos_address : qpos_address + 7], dtype=np.float32),
            np.asarray(data.qvel[dof_address : dof_address + 6], dtype=np.float32),
        ]
    ).astype(np.float32)


def _write_episode_image(
    dataset: LeRobotDataset, key: str, frame_index: int, image: np.ndarray
) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=frame_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=JPEG_QUALITY)


def _patch_video_crf() -> None:
    original = lerobot_dataset_module.encode_video_frames

    def encode_with_crf(*args: Any, **kwargs: Any) -> None:
        kwargs["crf"] = VIDEO_CRF
        return original(*args, **kwargs)

    lerobot_dataset_module.encode_video_frames = encode_with_crf


def _features() -> dict[str, dict[str, Any]]:
    image_shape = (3, CAMERA_RESOLUTION, CAMERA_RESOLUTION)
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
            "shape": (13,),
            "names": [
                "cube_x",
                "cube_y",
                "cube_z",
                "cube_qw",
                "cube_qx",
                "cube_qy",
                "cube_qz",
                "cube_vx",
                "cube_vy",
                "cube_vz",
                "cube_wx",
                "cube_wy",
                "cube_wz",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["commanded_initial_speed_mps", "launch_trigger"],
        },
    }


def collect_episode(
    *,
    dataset: LeRobotDataset,
    model: mujoco.MjModel,
    renderer_main: mujoco.Renderer,
    renderer_aux: mujoco.Renderer,
    gravity_index: int,
    gravity_mps2: float,
    speed_index: int,
    speed_mps: float,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    ids = _scene_ids(model)
    dof_address = int(model.jnt_dofadr[ids["cube_joint"]])
    camera_main = _camera_main()
    camera_aux = _camera_auxiliary()
    task = [
        "observe a blue cube launched from a smooth elevated platform",
        "predict its ballistic trajectory and first landing point",
        "complete calibrated physics rollout",
        "quality pass",
    ]
    launched = False
    launch_time = None
    contact_time = None
    contact_x = None
    contact_frame = None
    edge_time = None
    edge_speed = None
    max_abs_y = 0.0
    frame_index = 0
    mujoco.mj_forward(model, data)

    while frame_index / FPS <= MAX_EPISODE_S + PRE_ROLL_S:
        target_time = frame_index / float(FPS)
        while data.time + 0.5 * demo.SIM_TIMESTEP < target_time:
            if not launched and data.time >= PRE_ROLL_S - 0.5 * demo.SIM_TIMESTEP:
                data.qvel[dof_address] = float(speed_mps)
                launched = True
                launch_time = float(data.time)
            mujoco.mj_step(model, data)
            cube_position = np.asarray(data.xpos[ids["cube_body"]], dtype=np.float64)
            max_abs_y = max(max_abs_y, abs(float(cube_position[1])))
            if launched and edge_time is None and float(cube_position[0]) >= demo.PLATFORM_EDGE_X_M:
                edge_time = float(data.time)
                edge_speed = float(data.qvel[dof_address])
            if launched and contact_time is None and _has_contact(
                data, ids["cube_geom"], ids["table_geom"]
            ):
                contact_time = float(data.time)
                contact_x = float(cube_position[0])

        renderer_main.update_scene(data, camera=camera_main)
        main_image = renderer_main.render().copy()
        renderer_aux.update_scene(data, camera=camera_aux)
        auxiliary_image = renderer_aux.render().copy()
        launch_trigger = 1.0 if frame_index == int(round(PRE_ROLL_S * FPS)) else 0.0
        frame = {
            "observation.images.image": main_image,
            "observation.images.wrist_image": auxiliary_image,
            "observation.state": _cube_state(model, data, ids),
            "action": np.asarray([speed_mps, launch_trigger], dtype=np.float32),
        }
        dataset.add_frame(frame, task=task, timestamp=frame_index / float(FPS))
        _write_episode_image(dataset, "observation.images.image", frame_index, main_image)
        _write_episode_image(
            dataset, "observation.images.wrist_image", frame_index, auxiliary_image
        )
        if contact_time is not None and contact_frame is None:
            contact_frame = frame_index
        frame_index += 1
        if contact_frame is not None and frame_index > contact_frame + POST_CONTACT_FRAMES:
            break

    theory_x = _theoretical_landing_x(gravity_mps2, speed_mps)
    edge_ratio = None if edge_speed is None else float(edge_speed / speed_mps)
    theory_error = None if contact_x is None else float(contact_x - theory_x)
    checks = {
        "crossed_platform_edge": edge_time is not None,
        "landed_on_table": contact_time is not None,
        "landing_in_safe_bounds": bool(
            contact_x is not None and MIN_LANDING_X_M <= contact_x <= MAX_LANDING_X_M
        ),
        "edge_speed_preserved": bool(
            edge_ratio is not None
            and EDGE_SPEED_RATIO_RANGE[0] <= edge_ratio <= EDGE_SPEED_RATIO_RANGE[1]
        ),
        "lateral_drift_small": bool(max_abs_y <= MAX_ABS_Y_M),
        "theory_residual_small": bool(
            theory_error is not None and abs(theory_error) <= MAX_THEORY_ERROR_M
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Collection quality failure g={gravity_mps2} speed={speed_mps}: {checks}"
        )
    dataset.save_episode()
    return {
        "episode_index": int(dataset.meta.total_episodes - 1),
        "case_id": f"gravity_g{gravity_index:02d}_v{speed_index:02d}",
        "gravity_index": gravity_index,
        "gravity_mps2": gravity_mps2,
        "speed_index": speed_index,
        "initial_speed_mps": speed_mps,
        "frame_count": frame_index,
        "launch_frame": int(round(PRE_ROLL_S * FPS)),
        "edge_crossing_time_after_launch_s": None
        if edge_time is None or launch_time is None
        else float(edge_time - launch_time),
        "edge_speed_mps": edge_speed,
        "edge_speed_ratio": edge_ratio,
        "first_table_contact_frame": contact_frame,
        "first_table_contact_time_after_launch_s": None
        if contact_time is None or launch_time is None
        else float(contact_time - launch_time),
        "first_table_contact_x_m": contact_x,
        "theoretical_landing_x_m": theory_x,
        "theory_error_m": theory_error,
        "max_abs_y_m": max_abs_y,
        "checks": checks,
        "quality_pass": True,
    }


def run_collection(output_root: Path, config: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"Expected LeRobot v2.1, got {CODEBASE_VERSION}")
    if not config["calibration_summary"]["quality_pass"]:
        raise RuntimeError("Calibration did not pass; refusing collection")
    dataset_root = output_root / "hidden_straight_lerobot"
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_root} exists; pass --overwrite")
        shutil.rmtree(dataset_root)
    _patch_video_crf()
    dataset = LeRobotDataset.create(
        repo_id="libero_gravity_platform_segmented80_10speed_hidden_hai_machine",
        root=dataset_root,
        fps=FPS,
        features=_features(),
        use_videos=True,
        video_codec=VIDEO_CODEC,
        is_compute_episode_stats_image=False,
    )
    rows = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_gravity_platform_segmented80_10speed_hidden_lerobot_hai-machine",
        "lerobot_codebase_version": CODEBASE_VERSION,
        "storage_layout": "v2.1 per-episode parquet and per-camera MP4",
        "target_visible": False,
        "gravity_hidden_from_visual_and_task_text": True,
        "fps": FPS,
        "camera_resolution": CAMERA_RESOLUTION,
        "video_codec": VIDEO_CODEC,
        "video_crf": VIDEO_CRF,
        "jpeg_quality": JPEG_QUALITY,
        "camera_keys": {
            "observation.images.image": "main oblique camera used in demos",
            "observation.images.wrist_image": "static auxiliary side camera; compatibility key, no robot is present",
        },
        "config": config,
        "episodes": rows,
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": metadata["dataset_type"],
        "output_root": str(output_root),
        "hidden_straight_lerobot": str(dataset_root),
        "config": config,
        "episodes": rows,
    }

    def autosave() -> None:
        _write_json(output_root / "manifest.json", manifest)
        _write_json(dataset_root / "gravity_dataset_metadata.json", metadata)
        _write_jsonl(dataset_root / "meta" / "gravity_episode_metadata.jsonl", rows)

    gravities = [float(value) for value in config["gravities_mps2"]]
    speeds = [float(value) for value in config["initial_speeds_mps"]]
    total = len(gravities) * len(speeds)
    count = 0
    for gravity_index, gravity in enumerate(gravities):
        model = mujoco.MjModel.from_xml_string(demo._mjcf(gravity))
        renderer_main = mujoco.Renderer(
            model, height=CAMERA_RESOLUTION, width=CAMERA_RESOLUTION
        )
        renderer_aux = mujoco.Renderer(
            model, height=CAMERA_RESOLUTION, width=CAMERA_RESOLUTION
        )
        try:
            for speed_index, speed in enumerate(speeds):
                row = collect_episode(
                    dataset=dataset,
                    model=model,
                    renderer_main=renderer_main,
                    renderer_aux=renderer_aux,
                    gravity_index=gravity_index,
                    gravity_mps2=gravity,
                    speed_index=speed_index,
                    speed_mps=speed,
                )
                rows.append(row)
                count += 1
                print(
                    f"collect {count:03d}/{total:03d} {row['case_id']} "
                    f"g={gravity:.4f} v={speed:.3f} "
                    f"landing={row['first_table_contact_x_m']:.4f} "
                    f"frames={row['frame_count']}",
                    flush=True,
                )
                autosave()
        finally:
            demo._close_renderer(renderer_aux)
            demo._close_renderer(renderer_main)

    count_by_gravity = Counter(row["gravity_index"] for row in rows)
    count_by_speed = Counter(row["speed_index"] for row in rows)
    summary = {
        "episode_count": len(rows),
        "expected_episode_count": total,
        "quality_pass_count": sum(int(row["quality_pass"]) for row in rows),
        "hidden_straight_lerobot": str(dataset_root),
        "count_by_gravity_index": dict(sorted(count_by_gravity.items())),
        "count_by_speed_index": dict(sorted(count_by_speed.items())),
        "lerobot_codebase_version": CODEBASE_VERSION,
    }
    _write_json(output_root / "summary.json", summary)
    autosave()
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=["calibrate", "collect", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.stage in {"calibrate", "all"}:
        if output_root.exists() and args.overwrite:
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        config = run_calibration(output_root)
    else:
        config_path = output_root / "calibration" / "selected_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Calibration config missing: {config_path}; run --stage calibrate first"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.stage in {"collect", "all"}:
        run_collection(output_root, config, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()

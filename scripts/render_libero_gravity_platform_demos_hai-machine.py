#!/usr/bin/env python3
"""Render clean MuJoCo gravity-comparison demos for the LIBERO pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_OUTPUT = Path(
    "/home/yininghong/chenyuan/TTT-physics/repos/FastWAM-TTT/data/libero_gravity/demos"
)
DEFAULT_GRAVITIES = (1.00, 9.81, 50.00)
FPS = 30
WIDTH = 640
HEIGHT = 480
SIM_TIMESTEP = 0.002
PRE_ROLL_S = 0.50
POST_LAUNCH_S = 1.65
CUBE_HALF_SIZE_M = 0.025
CUBE_MASS_KG = 0.20
PLATFORM_EDGE_X_M = 0.0
PLATFORM_TOP_Z_M = 0.225
TABLE_TOP_Z_M = 0.0
CUBE_START_X_M = -0.36
INITIAL_SPEED_MPS = 0.80


@dataclass
class DemoResult:
    gravity_mps2: float
    video: str
    frames: list[np.ndarray]
    launch_time_s: float
    edge_crossing_time_s: float | None
    edge_speed_mps: float | None
    first_table_contact_time_s: float | None
    first_table_contact_x_m: float | None
    theoretical_landing_x_m: float
    final_cube_position_m: list[float]

    def metadata(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop("frames")
        return payload


def _mjcf(gravity_mps2: float) -> str:
    ruler_geoms = []
    for index, x in enumerate(np.arange(0.05, 0.451, 0.05)):
        alpha = 0.80 if index % 2 else 0.45
        ruler_geoms.append(
            f'<geom name="ruler_{index}" type="box" pos="{x:.3f} 0 0.002" '
            f'size="0.002 0.39 0.002" rgba="0.12 0.25 0.30 {alpha:.2f}" '
            'contype="0" conaffinity="0"/>'
        )
    ruler_xml = "\n".join(ruler_geoms)
    return f"""
<mujoco model="libero_gravity_platform_demo">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="{SIM_TIMESTEP}" integrator="RK4" gravity="0 0 -{gravity_mps2:.8f}"/>
  <visual>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.01" zfar="10"/>
    <rgba haze="0.25 0.28 0.30 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.74 0.82 0.84" rgb2="0.12 0.18 0.22" width="512" height="3072"/>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.52 0.31 0.17" rgb2="0.44 0.25 0.13" width="512" height="512"/>
    <material name="wood_mat" texture="wood" texrepeat="5 3" reflectance="0.08" shininess="0.30"/>
    <material name="platform_mat" rgba="0.74 0.79 0.80 1" reflectance="0.32" shininess="0.85"/>
    <material name="cube_mat" rgba="0.02 0.22 0.95 1" reflectance="0.18" shininess="0.55"/>
  </asset>
  <worldbody>
    <light name="key" pos="-0.4 -1.1 2.2" dir="0.2 0.3 -1" diffuse="0.95 0.92 0.86" castshadow="true"/>
    <light name="fill" pos="0.8 1.0 1.3" dir="-0.4 -0.3 -1" diffuse="0.45 0.55 0.62"/>
    <geom name="floor" type="plane" pos="0 0 -0.48" size="3 3 0.1" rgba="0.14 0.17 0.18 1"/>

    <body name="table">
      <geom name="tabletop" type="box" pos="0 0 -0.04" size="0.75 0.45 0.04"
            material="wood_mat" friction="0.65 0.01 0.001" solref="0.008 1"/>
      <geom type="box" pos="-0.62 -0.34 -0.27" size="0.045 0.045 0.23" rgba="0.24 0.16 0.11 1"/>
      <geom type="box" pos="-0.62 0.34 -0.27" size="0.045 0.045 0.23" rgba="0.24 0.16 0.11 1"/>
      <geom type="box" pos="0.62 -0.34 -0.27" size="0.045 0.045 0.23" rgba="0.24 0.16 0.11 1"/>
      <geom type="box" pos="0.62 0.34 -0.27" size="0.045 0.045 0.23" rgba="0.24 0.16 0.11 1"/>
    </body>

    {ruler_xml}
    <geom name="platform" type="box" pos="-0.25 0 0.200" size="0.25 0.16 0.025"
          material="platform_mat" friction="0.0002 0.0001 0.0001" solref="0.006 1"/>
    <geom name="platform_support" type="box" pos="-0.38 0 0.095" size="0.055 0.12 0.08"
          rgba="0.34 0.39 0.40 1"/>
    <geom name="edge_marker" type="box" pos="0 0 0.229" size="0.004 0.165 0.004"
          rgba="0.96 0.75 0.16 1" contype="0" conaffinity="0"/>

    <body name="cube" pos="{CUBE_START_X_M} 0 {PLATFORM_TOP_Z_M + CUBE_HALF_SIZE_M}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="{CUBE_HALF_SIZE_M} {CUBE_HALF_SIZE_M} {CUBE_HALF_SIZE_M}"
            mass="{CUBE_MASS_KG}" material="cube_mat" friction="0.0002 0.0001 0.0001"
            solref="0.008 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _label_frame(
    frame: np.ndarray,
    *,
    gravity_mps2: float,
    elapsed_after_launch_s: float,
    landing_x_m: float | None,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    lines = [
        f"gravity = {gravity_mps2:.2f} m/s^2",
        f"same horizontal speed = {INITIAL_SPEED_MPS:.2f} m/s",
        f"time after launch = {max(0.0, elapsed_after_launch_s):.2f} s",
    ]
    if landing_x_m is not None:
        lines.append(f"first landing x = {landing_x_m * 100:.1f} cm")
    box_width = 284
    box_height = 24 + 21 * len(lines)
    draw.rounded_rectangle((12, 12, box_width, box_height), radius=8, fill=(8, 15, 18, 220))
    for index, line in enumerate(lines):
        draw.text((24, 24 + 21 * index), line, fill=(245, 247, 239))
    return np.asarray(image)


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([-0.02, 0.0, 0.11], dtype=np.float64)
    camera.distance = 1.48
    camera.azimuth = 132.0
    camera.elevation = -24.0
    return camera


def _close_renderer(renderer: mujoco.Renderer) -> None:
    renderer._mjr_context.free()
    renderer._gl_context.free()


def _has_contact(data: mujoco.MjData, geom_a: int, geom_b: int) -> bool:
    target = {int(geom_a), int(geom_b)}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} == target:
            return True
    return False


def _video_tag(gravity_mps2: float) -> str:
    return f"g{gravity_mps2:05.2f}".replace(".", "p")


def render_gravity_case(gravity_mps2: float, output: Path) -> DemoResult:
    model = mujoco.MjModel.from_xml_string(_mjcf(gravity_mps2))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = _camera()
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tabletop")
    dof_address = int(model.jnt_dofadr[cube_joint])

    mujoco.mj_forward(model, data)
    total_frames = int(round((PRE_ROLL_S + POST_LAUNCH_S) * FPS)) + 1
    frames: list[np.ndarray] = []
    launched = False
    launch_time = PRE_ROLL_S
    edge_time = None
    edge_speed = None
    contact_time = None
    contact_x = None
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        output,
        fps=FPS,
        codec="libx264",
        ffmpeg_params=["-crf", "18"],
        macro_block_size=None,
    ) as writer:
        for frame_index in range(total_frames):
            target_time = frame_index / float(FPS)
            while data.time + 0.5 * SIM_TIMESTEP < target_time:
                if not launched and data.time >= PRE_ROLL_S - 0.5 * SIM_TIMESTEP:
                    data.qvel[dof_address] = INITIAL_SPEED_MPS
                    launched = True
                    launch_time = float(data.time)
                mujoco.mj_step(model, data)

                cube_x = float(data.xpos[cube_body, 0])
                if launched and edge_time is None and cube_x >= PLATFORM_EDGE_X_M:
                    edge_time = float(data.time)
                    edge_speed = float(data.qvel[dof_address])
                if launched and contact_time is None and _has_contact(data, cube_geom, table_geom):
                    contact_time = float(data.time)
                    contact_x = cube_x

            renderer.update_scene(data, camera=camera)
            rendered = renderer.render().copy()
            frame = _label_frame(
                rendered,
                gravity_mps2=gravity_mps2,
                elapsed_after_launch_s=float(data.time - launch_time),
                landing_x_m=contact_x,
            )
            frames.append(frame)
            writer.append_data(frame)

    _close_renderer(renderer)
    fall_height = PLATFORM_TOP_Z_M - TABLE_TOP_Z_M
    theoretical = PLATFORM_EDGE_X_M + INITIAL_SPEED_MPS * np.sqrt(
        2.0 * fall_height / float(gravity_mps2)
    )
    return DemoResult(
        gravity_mps2=float(gravity_mps2),
        video=str(output),
        frames=frames,
        launch_time_s=float(launch_time),
        edge_crossing_time_s=edge_time,
        edge_speed_mps=edge_speed,
        first_table_contact_time_s=contact_time,
        first_table_contact_x_m=contact_x,
        theoretical_landing_x_m=float(theoretical),
        final_cube_position_m=np.asarray(data.xpos[cube_body], dtype=np.float64).tolist(),
    )


def write_comparison(results: list[DemoResult], output: Path) -> None:
    max_frames = max(len(result.frames) for result in results)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output,
        fps=FPS,
        codec="libx264",
        ffmpeg_params=["-crf", "18"],
        macro_block_size=None,
    ) as writer:
        for frame_index in range(max_frames):
            panels = [
                result.frames[min(frame_index, len(result.frames) - 1)]
                for result in results
            ]
            writer.append_data(np.concatenate(panels, axis=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gravities", type=float, nargs="+", default=list(DEFAULT_GRAVITIES)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for gravity in args.gravities:
        output = args.output_dir / f"gravity_platform_{_video_tag(gravity)}_hai-machine.mp4"
        result = render_gravity_case(float(gravity), output)
        results.append(result)
        print(
            f"g={gravity:.2f} landing_x={result.first_table_contact_x_m} "
            f"theory_x={result.theoretical_landing_x_m:.4f} video={output}",
            flush=True,
        )

    comparison = args.output_dir / "gravity_platform_comparison_hai-machine.mp4"
    write_comparison(results, comparison)
    metadata = {
        "engine": "MuJoCo 2.3.7 (LIBERO/robosuite backend)",
        "purpose": "physics-only gravity identifiability demo before LIBERO task integration",
        "fixed_parameters": {
            "fps": FPS,
            "resolution_hw": [HEIGHT, WIDTH],
            "simulation_timestep_s": SIM_TIMESTEP,
            "cube_half_size_m": CUBE_HALF_SIZE_M,
            "cube_mass_kg": CUBE_MASS_KG,
            "initial_horizontal_speed_mps": INITIAL_SPEED_MPS,
            "platform_top_z_m": PLATFORM_TOP_Z_M,
            "platform_edge_x_m": PLATFORM_EDGE_X_M,
            "cube_start_x_m": CUBE_START_X_M,
            "platform_friction": [0.0002, 0.0001, 0.0001],
        },
        "comparison_video": str(comparison),
        "cases": [result.metadata() for result in results],
    }
    metadata_path = args.output_dir / "gravity_platform_demo_metadata_hai-machine.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"comparison={comparison}", flush=True)
    print(f"metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()

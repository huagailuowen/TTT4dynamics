#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DEMO_SCRIPT = REPO_ROOT / "scripts" / "render_libero_push_box_mass_sweep_mu0080_mid40cm_2026-07-16_hai-machine.py"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pushbox" / "demos" / "libero_push_box_mass_sweep_mu0040_mid40cm_2026-07-16_hai-machine"
FRICTION_MU = 0.04
TARGET_DISPLACEMENT_M = 0.40
CALIBRATION_AMPLITUDES = np.arange(0.12, 0.301, 0.02, dtype=np.float64)


def load_demo_module() -> Any:
    spec = importlib.util.spec_from_file_location("mass_sweep_mu0080_demo_base_hai_machine", BASE_DEMO_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import demo base script: {BASE_DEMO_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = load_demo_module()
demo.FRICTION_MU = FRICTION_MU
demo.TARGET_DISPLACEMENT_M = TARGET_DISPLACEMENT_M


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a fixed-action LIBERO box-mass comparison at mu=0.04.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
        demo.shutil.rmtree(output)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    bddl_file = demo.base.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=output / "bddl",
        geometry_id="mass_sweep_mu0040_hidden",
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
    for index, amplitude in enumerate(CALIBRATION_AMPLITUDES):
        result = demo.rollout(
            amplitude=float(amplitude),
            mass_kg=None,
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=False,
        )
        calibration.append(demo.without_frames(result))
        print(
            f"[calibration {index + 1:02d}/{len(CALIBRATION_AMPLITUDES):02d}] "
            f"A={amplitude:.3f} displacement={result['final_displacement_m'] * 100.0:.1f}cm",
            flush=True,
        )
    chosen = min(calibration, key=lambda item: abs(float(item["final_displacement_m"]) - TARGET_DISPLACEMENT_M))
    chosen_amplitude = float(chosen["amplitude"])
    native_mass = float(chosen["mass"]["native_mass_kg"])

    results = []
    resolved_masses = [native_mass if mass is None else float(mass) for mass in demo.MASS_TARGETS_KG]
    for index, mass in enumerate(resolved_masses):
        result = demo.rollout(
            amplitude=chosen_amplitude,
            mass_kg=mass,
            bddl_file=bddl_file,
            seed=int(args.seed),
            capture_frames=True,
        )
        tag = f"mass_{mass * 1000.0:07.2f}g".replace(".", "p")
        video = output / "videos" / f"{index + 1:02d}_{tag}.mp4"
        demo.write_video(video, result["frames"])
        result["video"] = str(video)
        results.append(result)
        print(
            f"[mass {index + 1}/5] {mass * 1000.0:.2f}g -> "
            f"{result['final_displacement_m'] * 100.0:.1f}cm, "
            f"peak_v={result['max_box_planar_speed_mps']:.3f}m/s",
            flush=True,
        )

    comparison = output / "videos" / "00_mass_comparison_mu0040_fixed_action.mp4"
    demo.write_comparison(comparison, results)
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "artifact_type": "video_demo_only_not_a_training_dataset",
        "purpose": "Measure box-mass dependence under one fixed push action at mu=0.04.",
        "friction_mu": FRICTION_MU,
        "target_calibration_displacement_m": TARGET_DISPLACEMENT_M,
        "chosen_amplitude": chosen_amplitude,
        "native_box_mass_kg": native_mass,
        "mass_targets_kg": resolved_masses,
        "action_policy": "original formal-6friction event_hold profile: first command 0.5A, then A until contact, hold A for 3 steps, then zero",
        "inertia_policy": "scale all object body inertias by the same ratio as body mass",
        "comparison_video": str(comparison),
        "calibration": calibration,
        "mass_results": [demo.without_frames(result) for result in results],
    }
    demo.write_json(output / "summary.json", summary)
    print(f"chosen A={chosen_amplitude:.3f}; native mass={native_mass * 1000.0:.3f}g", flush=True)
    print(f"comparison={comparison}", flush=True)


if __name__ == "__main__":
    main()

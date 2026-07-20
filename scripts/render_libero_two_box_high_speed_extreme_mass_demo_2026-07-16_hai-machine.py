#!/usr/bin/env python3
"""High-speed two-block collision demo with identical robot actions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts" / "render_libero_two_box_collision_mass_demo_2026-07-16_hai-machine.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("two_box_collision_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    demo = load_base_module()

    # Keep the original experiment untouched and specialize it through globals
    # consumed by its BDDL writer, rollout, calibration, and reporting code.
    demo.DEFAULT_OUTPUT = (
        REPO_ROOT / "outputs" / "pushbox" / "demos"
        / "libero_two_box_high_speed_extreme_mass_mu0010_2026-07-16_hai-machine"
    )
    demo.FRICTION_MU = 0.01
    demo.PROJECTILE_MASS_KG = 0.25
    demo.TARGET_MASSES_KG = (0.010, 0.025, 0.10, 0.50, 2.0, 5.0)

    # The original center spacing was 0.35 m. This leaves 0.26 m center spacing:
    # close enough for a short free-flight, but with time to verify separation.
    demo.PROJECTILE_INIT_XY = (-0.30, -0.035)
    demo.TARGET_INIT_XY = (-0.04, -0.035)

    demo.CONTROLLER_TRANSLATION_SCALE = 4.0
    demo.LAUNCH_PROFILE = np.asarray([0.70, 1.0, 1.0, 1.0, 1.0, 0.70], dtype=np.float64)
    demo.BRAKE_PROFILE = np.asarray([-0.80, -0.80, -0.40, 0.0, 0.0], dtype=np.float64)
    demo.CALIBRATION_AMPLITUDES = (0.34, 0.36, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50)

    # Deliberately above the expected range: the existing clean-candidate selector
    # therefore retains the fastest collision that passes the separation check.
    demo.TARGET_PREIMPACT_SPEED_MPS = 1.50
    demo.main()


if __name__ == "__main__":
    main()

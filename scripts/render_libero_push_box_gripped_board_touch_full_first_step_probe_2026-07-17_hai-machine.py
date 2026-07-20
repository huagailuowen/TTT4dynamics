#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WIDE_SWEEP_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "render_libero_push_box_gripped_board_touch_wide_A_sweep_2026-07-17_hai-machine.py"
)
BASELINE_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "libero_push_box_gripped_board_touch_wide_A_mu015_2026-07-17_hai-machine"
    / "summary.json"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "libero_push_box_gripped_board_touch_full_first_step_mu015_2026-07-17_hai-machine"
)
AMPLITUDES = (
    0.020,
    0.100,
    0.200,
    0.250,
    0.300,
    0.325,
    0.350,
    0.375,
    0.400,
    0.425,
    0.450,
    0.475,
    0.500,
    0.550,
    0.600,
    0.700,
    0.800,
)
FULL_FIRST_STEP_PROFILE = np.asarray([1.0] * 8 + [0.5, 0.0], dtype=np.float64)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wide = load_module(WIDE_SWEEP_SCRIPT, "touch_full_first_step_wide_sweep_hai_machine")


def plot_comparison(path: Path, baseline: list[dict[str, Any]], full: list[dict[str, Any]]) -> None:
    baseline_by_a = {round(float(row["A"]), 6): row for row in baseline}
    full = sorted(full, key=lambda row: float(row["A"]))
    half = [baseline_by_a[round(float(row["A"]), 6)] for row in full]
    a = np.asarray([float(row["A"]) for row in full])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    series = (
        ("Final displacement (cm)", lambda row: float(row["final_displacement_from_launch_m"]) * 100.0),
        ("Release box vx (m/s)", lambda row: float(row["release_box_vxy_mps"][0])),
        ("Peak box vx (m/s)", lambda row: float(row["peak_box_vx_mps"])),
        ("Absolute lateral drift (cm)", lambda row: abs(float(row["final_lateral_from_launch_m"])) * 100.0),
    )
    for axis, (label, value) in zip(axes.flat, series):
        axis.plot(a, [value(row) for row in half], "o-", label="first step 0.5A", color="#c85b32")
        axis.plot(a, [value(row) for row in full], "s-", label="first step A", color="#147d78")
        axis.set_xlabel("Action amplitude A")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Touch start: half-speed versus full-speed first control step")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    wide.AMPLITUDES = AMPLITUDES
    wide.FIXED_ACTION_PROFILE = FULL_FIRST_STEP_PROFILE
    wide.DEFAULT_OUTPUT = OUTPUT_ROOT
    wide.main()

    summary_path = OUTPUT_ROOT / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    baseline_by_a = {round(float(row["A"]), 6): row for row in baseline["results"]}
    comparisons = []
    for row in payload["results"]:
        base = baseline_by_a[round(float(row["A"]), 6)]
        comparisons.append(
            {
                "A": float(row["A"]),
                "distance_delta_full_minus_half_cm": (
                    float(row["final_displacement_from_launch_m"])
                    - float(base["final_displacement_from_launch_m"])
                )
                * 100.0,
                "release_vx_delta_full_minus_half_mps": (
                    float(row["release_box_vxy_mps"][0]) - float(base["release_box_vxy_mps"][0])
                ),
                "peak_vx_delta_full_minus_half_mps": (
                    float(row["peak_box_vx_mps"]) - float(base["peak_box_vx_mps"])
                ),
            }
        )
    payload["experiment"] = "strict touch-start comparison with full-speed first control step"
    payload["baseline_profile"] = [0.5] + [1.0] * 7 + [0.5, 0.0]
    payload["profile_integral_ratio_to_baseline"] = float(
        np.sum(FULL_FIRST_STEP_PROFILE) / np.sum(np.asarray(payload["baseline_profile"], dtype=np.float64))
    )
    payload["comparison_to_half_first_step"] = comparisons
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_comparison(OUTPUT_ROOT / "half_vs_full_first_step.png", baseline["results"], payload["results"])
    print(f"comparison_plot={OUTPUT_ROOT / 'half_vs_full_first_step.png'}", flush=True)


if __name__ == "__main__":
    main()

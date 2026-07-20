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
FULL8_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "libero_push_box_gripped_board_touch_full_first_step_mu015_2026-07-17_hai-machine"
    / "summary.json"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "libero_push_box_gripped_board_touch_duration_4_5_6_8_mu015_2026-07-17_hai-machine"
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
NEW_DURATIONS = (4, 5, 6)
VIDEO_AMPLITUDES = {0.200, 0.350, 0.450, 0.600, 0.800}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wide = load_module(WIDE_SWEEP_SCRIPT, "touch_duration_probe_wide_sweep_hai_machine")


def duration_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["A"]))
    distance = np.asarray([float(row["final_displacement_from_launch_m"]) * 100.0 for row in ordered])
    release_vx = np.asarray([float(row["release_box_vxy_mps"][0]) for row in ordered])
    lateral = np.asarray([abs(float(row["final_lateral_from_launch_m"])) * 100.0 for row in ordered])
    amplitudes = np.asarray([float(row["A"]) for row in ordered])
    deltas = np.diff(distance)
    clean = np.flatnonzero(lateral <= 1.0)
    severe = np.flatnonzero(deltas < -2.0)
    return {
        "distance_range_cm": [float(np.min(distance)), float(np.max(distance))],
        "max_distance_with_lateral_le_1cm": None if not len(clean) else float(np.max(distance[clean])),
        "A_at_max_clean_distance": None if not len(clean) else float(amplitudes[clean[np.argmax(distance[clean])]]),
        "largest_adjacent_drop_cm": float(min(0.0, np.min(deltas))),
        "first_drop_over_2cm_after_A": None if not len(severe) else float(amplitudes[severe[0]]),
        "distance_decrease_count_over_1cm": int(np.sum(deltas < -1.0)),
        "release_vx_range_mps": [float(np.min(release_vx)), float(np.max(release_vx))],
        "max_abs_lateral_cm": float(np.max(lateral)),
    }


def plot_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.3), constrained_layout=True)
    colors = {4: "#176b87", 5: "#159270", 6: "#d07a2d", 8: "#b64234"}
    series = (
        ("Final displacement (cm)", lambda row: float(row["final_displacement_from_launch_m"]) * 100.0),
        ("Release box vx (m/s)", lambda row: float(row["release_box_vxy_mps"][0])),
        ("Peak box vx (m/s)", lambda row: float(row["peak_box_vx_mps"])),
        ("Absolute lateral drift (cm)", lambda row: abs(float(row["final_lateral_from_launch_m"])) * 100.0),
    )
    for duration in (4, 5, 6, 8):
        selected = sorted(
            (row for row in rows if int(row["full_speed_frames"]) == duration),
            key=lambda row: float(row["A"]),
        )
        a = [float(row["A"]) for row in selected]
        for axis, (label, value) in zip(axes.flat, series):
            axis.plot(a, [value(row) for row in selected], "o-", markersize=3.5, color=colors[duration], label=f"{duration} full-A frames")
            axis.set_xlabel("Action amplitude A")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Strict touch start: fixed full-speed duration comparison")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    videos_root = OUTPUT_ROOT / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    config = wide.comparison.board_probe.source_dataset.configure_dataset()
    prepare = dict(config["prepare_config"])
    prepare["descend_steps"] = 45
    prepare["prepare_position_gain"] = 8.0
    config["prepare_config"] = prepare
    manifest = json.loads(wide.comparison.board_probe.SOURCE_MANIFEST.read_text(encoding="utf-8"))
    bddl_file = next(
        row["bddl_file"]
        for row in manifest["episodes"]
        if abs(float(row["mu"]) - wide.comparison.board_probe.FRICTION_MU) < 1e-12
    )

    rows = []
    for duration in NEW_DURATIONS:
        wide.FIXED_ACTION_PROFILE = np.asarray([1.0] * duration + [0.5, 0.0], dtype=np.float64)
        for action_id, amplitude in enumerate(AMPLITUDES):
            result = wide.rollout_fixed_release(
                config,
                amplitude=float(amplitude),
                action_id=action_id,
                bddl_file=bddl_file,
                seed=0,
            )
            frames = result.pop("frames")
            result["full_speed_frames"] = int(duration)
            result["action_profile"] = wide.FIXED_ACTION_PROFILE.astype(float).tolist()
            if float(amplitude) in VIDEO_AMPLITUDES:
                video = videos_root / f"touch_full{duration:02d}_A{int(round(amplitude * 1000)):03d}.mp4"
                wide.comparison.board_probe.write_video(video, frames)
                result["video"] = str(video)
            else:
                result["video"] = None
            rows.append(result)
            print(
                f"duration={duration} A={amplitude:.3f} "
                f"distance={result['final_displacement_from_launch_m'] * 100.0:.2f}cm "
                f"release_vx={result['release_box_vxy_mps'][0]:.3f}m/s "
                f"lateral={result['final_lateral_from_launch_m'] * 100.0:+.2f}cm",
                flush=True,
            )

    full8_payload = json.loads(FULL8_SUMMARY.read_text(encoding="utf-8"))
    full8_by_a = {round(float(row["A"]), 6): row for row in full8_payload["results"]}
    for amplitude in AMPLITUDES:
        row = dict(full8_by_a[round(float(amplitude), 6)])
        row["full_speed_frames"] = 8
        row["action_profile"] = [1.0] * 8 + [0.5, 0.0]
        rows.append(row)

    analysis = {
        str(duration): duration_analysis([row for row in rows if int(row["full_speed_frames"]) == duration])
        for duration in (4, 5, 6, 8)
    }
    payload = {
        "experiment": "strict touch-start comparison of 4, 5, 6, and 8 full-A frames",
        "friction_mu": wide.comparison.board_probe.FRICTION_MU,
        "amplitudes": list(AMPLITUDES),
        "profiles": {str(duration): [1.0] * duration + [0.5, 0.0] for duration in (4, 5, 6, 8)},
        "analysis": analysis,
        "results": rows,
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(
        json.dumps(wide.comparison.board_probe.source.base.to_jsonable(payload), indent=2), encoding="utf-8"
    )
    plot_path = OUTPUT_ROOT / "duration_4_5_6_8_comparison.png"
    plot_results(plot_path, rows)
    print(f"summary={summary_path}", flush=True)
    print(f"plot={plot_path}", flush=True)


if __name__ == "__main__":
    main()

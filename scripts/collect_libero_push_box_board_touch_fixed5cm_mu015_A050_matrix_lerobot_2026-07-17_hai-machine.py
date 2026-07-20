#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_board_touch_fixed5cm_ramp_latched_brake_probe_lerobot_2026-07-17_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_board_touch_fixed5cm_mu015_A050_matrix_2026-07-17_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "fixed_travel"
    / "libero_push_box_board_touch_fixed5cm_mu015_A050_matrix_80eps_lerobot_2026-07-17_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = load_module(SOURCE_SCRIPT, "fixed5cm_mu015_A050_matrix_source_hai_machine")
collector = source.collector
base = source.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a fixed-5-cm action-friction LeRobot matrix.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_index",
        "mu",
        "A",
        "box_displacement_cm",
        "box_forward_cm",
        "box_lateral_cm",
        "peak_box_vx_mps",
        "max_eef_travel_cm",
        "final_eef_travel_cm",
        "eef_overshoot_cm",
        "contact_episode_count",
        "nonzero_x_action_frames",
        "x_action_sign_changes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "episode_index": row["episode_index"],
                    "mu": row["mu"],
                    "A": row["A"],
                    "box_displacement_cm": metrics["final_box_displacement_from_launch_m"] * 100.0,
                    "box_forward_cm": metrics["final_box_forward_from_launch_m"] * 100.0,
                    "box_lateral_cm": metrics["final_box_lateral_from_launch_m"] * 100.0,
                    "peak_box_vx_mps": metrics["peak_box_vx_mps"],
                    "max_eef_travel_cm": metrics["max_eef_travel_m"] * 100.0,
                    "final_eef_travel_cm": metrics["final_eef_travel_m"] * 100.0,
                    "eef_overshoot_cm": metrics["eef_overshoot_m"] * 100.0,
                    "contact_episode_count": metrics["sampled_contact_episode_count"],
                    "nonzero_x_action_frames": metrics["nonzero_x_action_frames"],
                    "x_action_sign_changes": metrics["significant_x_action_sign_changes"],
                }
            )


def matrix_values(
    rows: list[dict[str, Any]],
    frictions: list[float],
    amplitudes: list[float],
    value: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    indexed = {(float(row["mu"]), float(row["A"])): row for row in rows}
    return np.asarray(
        [[value(indexed[(float(mu), float(amplitude))]) for amplitude in amplitudes] for mu in frictions],
        dtype=np.float64,
    )


def markdown_matrix(title: str, values: np.ndarray, frictions: list[float], amplitudes: list[float], digits: int) -> str:
    header = "| mu \\ A | " + " | ".join(f"{amplitude:.2f}" for amplitude in amplitudes) + " |"
    divider = "|---:" + "|---:" * len(amplitudes) + "|"
    lines = [f"## {title}", "", header, divider]
    for mu, row in zip(frictions, values):
        rendered = " | ".join(f"{item:.{digits}f}" for item in row)
        lines.append(f"| {mu:g} | {rendered} |")
    return "\n".join(lines)


def write_tables(path: Path, rows: list[dict[str, Any]], frictions: list[float], amplitudes: list[float]) -> None:
    distance = matrix_values(
        rows,
        frictions,
        amplitudes,
        lambda row: float(row["metrics"]["final_box_displacement_from_launch_m"]) * 100.0,
    )
    velocity = matrix_values(
        rows,
        frictions,
        amplitudes,
        lambda row: float(row["metrics"]["peak_box_vx_mps"]),
    )
    eef_travel = matrix_values(
        rows,
        frictions,
        amplitudes,
        lambda row: float(row["metrics"]["max_eef_travel_m"]) * 100.0,
    )
    sections = [
        "# Fixed 5 cm action-friction rollout tables",
        "",
        "All values come from real MuJoCo rollouts recorded as LeRobot episodes.",
        "The first action frame is 0.7A, then A until braking, followed by a latched zero command.",
        "",
        markdown_matrix("Final box displacement (cm)", distance, frictions, amplitudes, 2),
        "",
        markdown_matrix("Peak box x velocity (m/s)", velocity, frictions, amplitudes, 3),
        "",
        markdown_matrix("Maximum EEF x travel (cm)", eef_travel, frictions, amplitudes, 2),
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def plot_heatmaps(path: Path, rows: list[dict[str, Any]], frictions: list[float], amplitudes: list[float]) -> None:
    matrices = [
        (
            "Box displacement (cm)",
            matrix_values(rows, frictions, amplitudes, lambda row: row["metrics"]["final_box_displacement_from_launch_m"] * 100.0),
        ),
        (
            "Peak box vx (m/s)",
            matrix_values(rows, frictions, amplitudes, lambda row: row["metrics"]["peak_box_vx_mps"]),
        ),
        (
            "Max EEF travel (cm)",
            matrix_values(rows, frictions, amplitudes, lambda row: row["metrics"]["max_eef_travel_m"] * 100.0),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.1), constrained_layout=True)
    for axis, (title, values) in zip(axes, matrices):
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Peak action A")
        axis.set_ylabel("Friction mu")
        axis.set_xticks(range(len(amplitudes)), [f"{value:.2f}" for value in amplitudes], rotation=45)
        axis.set_yticks(range(len(frictions)), [f"{value:g}" for value in frictions])
        fig.colorbar(image, ax=axis, shrink=0.85)
    fig.suptitle("Fixed 5 cm board-touch rollout matrix, first frame = 0.7A")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiment = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    formal_experiment = json.loads(collector.CONFIG_PATH.read_text(encoding="utf-8"))
    config = collector.configure_dataset(formal_experiment)
    config["camera_resolution"] = int(experiment["camera_resolution"])
    config["fps"] = int(experiment["fps"])
    frictions = [float(value) for value in experiment["frictions"]]
    amplitudes = [float(value) for value in experiment["peak_amplitudes"]]
    total = len(frictions) * len(amplitudes)
    if total != int(experiment["expected_episode_count"]):
        raise RuntimeError(f"Configured {total} episodes, expected {experiment['expected_episode_count']}")

    base.patch_lerobot_video_crf(int(experiment["recording"]["video_crf"]))
    dataset_root = output_root / "lerobot"
    dataset = base.LeRobotDataset.create(
        repo_id="libero_push_box_board_touch_fixed5cm_mu015_A050_matrix_hai_machine",
        root=dataset_root,
        fps=int(experiment["fps"]),
        features=base.build_features(int(experiment["camera_resolution"])),
        use_videos=True,
        video_codec=str(experiment["recording"]["video_codec"]),
        is_compute_episode_stats_image=False,
    )

    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_board_touch_fixed5cm_mu015_A050_matrix_lerobot_hai-machine",
        "experiment_config": experiment,
        "episodes": [],
    }
    count = 0
    for friction_index, mu in enumerate(frictions):
        for action_id, amplitude in enumerate(amplitudes):
            case_id = f"fixed5cm_A{int(round(amplitude * 1000)):04d}_mu{int(round(mu * 10000)):04d}"
            action_cfg = {"action_id": int(action_id), "A": amplitude, "push_steps": 16}
            bddl = base.write_hidden_bddl(config, bddl_dir=output_root / "bddl", geometry_id=case_id)
            base_case = collector.make_case(
                config,
                mu=mu,
                action_cfg=action_cfg,
                case_id=case_id,
                bddl_file=bddl,
            )
            case = source.preserve_case_attributes(
                base_case,
                replace(base_case, pusher_max_pos_action=float(experiment["pusher_max_pos_action"])),
            )
            episode_index, metrics = source.rollout(
                case,
                dataset=dataset,
                amplitude=amplitude,
                first_fraction=float(experiment["first_frame_fraction"]),
                travel_m=float(experiment["travel_m"]),
                controller_cfg=experiment["controller"],
                recorded_steps=int(experiment["recorded_steps"]),
                seed=int(args.seed),
                fps=int(experiment["fps"]),
                jpeg_quality=int(experiment["recording"]["jpeg_quality"]),
            )
            row = {
                "episode_index": int(episode_index),
                "case_id": case_id,
                "friction_index": int(friction_index),
                "action_id": int(action_id),
                "mu": mu,
                "A": amplitude,
                "metrics": metrics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            count += 1
            print(
                f"matrix {count:02d}/{total:02d} mu={mu:.3f} A={amplitude:.2f} "
                f"box={metrics['final_box_displacement_from_launch_m'] * 100:.2f}cm "
                f"peak_vx={metrics['peak_box_vx_mps']:.3f}m/s "
                f"eef_max={metrics['max_eef_travel_m'] * 100:.2f}cm "
                f"contacts={metrics['sampled_contact_episode_count']}",
                flush=True,
            )

    summary = {
        "experiment": experiment["experiment"],
        "episode_count": len(rows),
        "experiment_config": experiment,
        "lerobot_root": str(dataset_root),
        "results": rows,
    }
    write_json(output_root / "summary.json", summary)
    base.write_dataset_metadata(dataset_root, metadata, rows)
    write_flat_csv(output_root / "rollout_metrics.csv", rows)
    write_tables(output_root / "rollout_tables.md", rows, frictions, amplitudes)
    plot_heatmaps(output_root / "action_friction_heatmaps.png", rows, frictions, amplitudes)
    print(f"summary={output_root / 'summary.json'}", flush=True)
    print(f"tables={output_root / 'rollout_tables.md'}", flush=True)
    print(f"plot={output_root / 'action_friction_heatmaps.png'}", flush=True)
    print(f"lerobot={dataset_root}", flush=True)


if __name__ == "__main__":
    main()

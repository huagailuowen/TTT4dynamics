#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_board_touch_fixed_travel_probe_lerobot_2026-07-17_hai-machine.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "libero_push_box_board_touch_fixed5cm_high_force_probe_2026-07-17_hai-machine.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pushbox"
    / "board_touch"
    / "fixed_travel"
    / "libero_push_box_board_touch_fixed5cm_aggressive_A1000_mu020_lerobot_2026-07-17_hai-machine"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_module(SOURCE_SCRIPT, "fixed5cm_high_force_probe_source_hai_machine")
collector = probe.collector
base = probe.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe extended real action limits at fixed 5 cm EEF travel.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def preserve_case_attributes(original: Any, updated: Any) -> Any:
    object.__setattr__(updated, "hai_action_id", getattr(original, "hai_action_id"))
    object.__setattr__(updated, "hai_action_profile", getattr(original, "hai_action_profile"))
    return updated


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def plot_results(path: Path, rows: list[dict[str, Any]], frictions: list[float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    colors = ["#176b87", "#d17a22", "#a13f35"]
    for mu, color in zip(frictions, colors):
        selected = sorted(
            (row["metrics"] for row in rows if abs(float(row["mu"]) - float(mu)) < 1e-12),
            key=lambda metrics: float(metrics["A"]),
        )
        a = [float(metrics["A"]) for metrics in selected]
        axes[0].plot(
            a,
            [float(metrics["final_box_displacement_from_launch_m"]) * 100.0 for metrics in selected],
            "o-",
            color=color,
            label=f"mu={mu:.3f}",
        )
        axes[1].plot(
            a,
            [float(metrics["peak_box_vx_mps"]) for metrics in selected],
            "o-",
            color=color,
        )
        axes[2].plot(
            a,
            [float(metrics["max_eef_travel_m"]) * 100.0 for metrics in selected],
            "o-",
            color=color,
        )
    axes[0].set_ylabel("Box displacement (cm)")
    axes[1].set_ylabel("Peak box vx (m/s)")
    axes[2].set_ylabel("Maximum EEF travel (cm)")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Real action limit A")
        axis.grid(alpha=0.25)
    fig.suptitle("Fixed 5 cm board travel: extended real action ceiling")
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

    experiment = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    formal_experiment = json.loads(collector.CONFIG_PATH.read_text(encoding="utf-8"))
    config = collector.configure_dataset(formal_experiment)
    config["camera_resolution"] = int(experiment["camera_resolution"])
    config["fps"] = int(experiment["fps"])
    total = len(experiment["frictions"]) * len(experiment["amplitudes"])
    if total != int(experiment["expected_episode_count"]):
        raise RuntimeError(f"Configured {total} episodes, expected {experiment['expected_episode_count']}")

    base.patch_lerobot_video_crf(int(experiment["recording"]["video_crf"]))
    dataset_root = output_root / "lerobot"
    dataset = base.LeRobotDataset.create(
        repo_id="libero_push_box_board_touch_fixed5cm_aggressive_A1000_probe_hai_machine",
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
        "dataset_type": "libero_push_box_board_touch_fixed5cm_aggressive_A1000_probe_lerobot_hai-machine",
        "experiment_config": experiment,
        "episodes": [],
    }
    count = 0
    for mu_index, mu in enumerate(experiment["frictions"]):
        for action_id, amplitude in enumerate(experiment["amplitudes"]):
            case_id = f"fixed5cm_high_force_mu{int(round(float(mu) * 10000)):04d}_A{int(round(float(amplitude) * 1000)):03d}"
            action_cfg = {"action_id": int(action_id), "A": float(amplitude), "push_steps": 16}
            bddl = base.write_hidden_bddl(config, bddl_dir=output_root / "bddl", geometry_id=case_id)
            base_case = collector.make_case(
                config,
                mu=float(mu),
                action_cfg=action_cfg,
                case_id=case_id,
                bddl_file=bddl,
            )
            case = preserve_case_attributes(
                base_case,
                replace(base_case, pusher_max_pos_action=float(experiment["pusher_max_pos_action"])),
            )
            episode_index, metrics = probe.rollout(
                case,
                dataset=dataset,
                mu=float(mu),
                amplitude=float(amplitude),
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
                "mu_index": int(mu_index),
                "action_id": int(action_id),
                "mu": float(mu),
                "A": float(amplitude),
                "target_travel_m": float(experiment["travel_m"]),
                "pusher_max_pos_action": float(experiment["pusher_max_pos_action"]),
                "metrics": metrics,
            }
            rows.append(row)
            metadata["episodes"].append(row)
            count += 1
            print(
                f"high-force {count:02d}/{total:02d} mu={float(mu):.3f} A={float(amplitude):.2f} "
                f"box={metrics['final_box_displacement_from_launch_m'] * 100.0:.2f}cm "
                f"peak_vx={metrics['peak_box_vx_mps']:.3f}m/s "
                f"eef_max={metrics['max_eef_travel_m'] * 100.0:.2f}cm "
                f"overshoot={metrics['eef_overshoot_m'] * 100.0:.2f}cm",
                flush=True,
            )
            base.write_dataset_metadata(dataset_root, metadata, rows)

    summary = {
        "experiment": experiment["experiment"],
        "episode_count": len(rows),
        "experiment_config": experiment,
        "lerobot_root": str(dataset_root),
        "results": rows,
    }
    write_json(output_root / "summary.json", summary)
    base.write_dataset_metadata(dataset_root, metadata, rows)
    plot_results(
        output_root / "fixed5cm_aggressive_force_comparison.png",
        rows,
        [float(mu) for mu in experiment["frictions"]],
    )
    print(f"summary={output_root / 'summary.json'}", flush=True)
    print(f"plot={output_root / 'fixed5cm_aggressive_force_comparison.png'}", flush=True)
    print(f"lerobot={dataset_root}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_6FRIC_SCRIPT = REPO_ROOT / "scripts" / "collect_libero_push_box_formal_6fric_50pair_35_35_direct_lerobot_hai-machine.py"
OUTPUT_ROOT = REPO_ROOT / "data" / "libero_push_box_15fric_interp_from_6fric_midA210_hidden_lerobot_2026-07-08_hai-machine"

FRICTIONS = [
    0.005,
    0.0075,
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.050,
    0.0625,
    0.075,
    0.0875,
    0.100,
    0.1166666667,
    0.1333333333,
    0.150,
]
ORIGINAL_6FRIC = [0.005, 0.010, 0.020, 0.050, 0.100, 0.150]
SELECTED_SOURCE_PAIR_ID = "mu0500_004_mid_event_hold_A210_h3_jitter"
SELECTED_SOURCE_DISPLACEMENT_M = 0.3392
ACTION_A = 0.209630
ACTION_HOLD = 3
JITTER_XY = (-0.020654035368863094, -0.02387665659926278)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("collect_6fric_base_hai_machine", BASE_6FRIC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


base = load_base_module()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.to_jsonable(value), indent=2), encoding="utf-8")


def make_hidden_case(*, mu_index: int, mu: float, bddl_root: Path) -> Any:
    mu_tag = base.mu_tag(mu)
    case_id = f"interp15_m{mu_index:02d}_{mu_tag}_A210_h3_hidden"
    target_xy = (float(base.INIT_XY[0]) + float(SELECTED_SOURCE_DISPLACEMENT_M), float(base.INIT_XY[1]))
    bddl_file = base.write_geometry_bddl(
        repo_root=REPO_ROOT,
        bddl_dir=bddl_root,
        geometry_id=case_id,
        init_xy=base.INIT_XY,
        target_xy=target_xy,
        init_half_size=0.02,
        target_radius=base.TARGET_RADIUS,
        target_rgba=(0.0, 0.8, 0.2, 0.0),
    )
    plan = {
        "mu": float(mu),
        "mode": "event_hold",
        "sampler_bucket": "mid",
        "A": float(ACTION_A),
        "hold": int(ACTION_HOLD),
    }
    case = base.build_push_case(
        plan,
        case_id=case_id,
        bddl_file=bddl_file,
        camera_resolution=int(base.CAMERA_RESOLUTION),
        target_xy=target_xy,
    )
    return case


def main() -> None:
    base.patch_lerobot_video_crf(base.VIDEO_CRF)
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bddl_root = OUTPUT_ROOT / "bddl" / "hidden"
    dataset_root = OUTPUT_ROOT / "hidden_straight_lerobot"
    dataset = base.create_dataset(
        dataset_root,
        repo_id="libero_push_box_15fric_interp_from_6fric_midA210_hidden_hai_machine",
    )
    rows: list[dict[str, Any]] = []
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_15fric_interp_from_6fric_midA210_hidden_lerobot_hai-machine",
        "domain": "observation",
        "target_visible": False,
        "split": "straight",
        "camera_resolution": int(base.CAMERA_RESOLUTION),
        "fps": int(base.FPS),
        "video_codec": str(base.VIDEO_CODEC),
        "video_crf": int(base.VIDEO_CRF),
        "jpeg_quality": int(base.JPEG_QUALITY),
        "state_source": "true LIBERO obs robot0_eef_pos, robot0_eef_quat converted to axis-angle, robot0_gripper_qpos",
        "source_6fric_pair": {
            "pair_id": SELECTED_SOURCE_PAIR_ID,
            "mu": 0.05,
            "bucket": "mid",
            "probe_displacement_m": SELECTED_SOURCE_DISPLACEMENT_M,
            "A": ACTION_A,
            "hold": ACTION_HOLD,
            "jitter_xy": list(JITTER_XY),
        },
        "frictions": FRICTIONS,
        "original_6fric": ORIGINAL_6FRIC,
        "episodes": [],
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dataset_type": "libero_push_box_15fric_interp_from_6fric_midA210_hidden_collection_hai-machine",
        "output_root": str(OUTPUT_ROOT),
        "hidden_straight_lerobot": str(dataset_root),
        "frictions": FRICTIONS,
        "original_6fric": ORIGINAL_6FRIC,
        "source_6fric_pair": metadata["source_6fric_pair"],
        "episodes": [],
    }

    def autosave() -> None:
        write_json(OUTPUT_ROOT / "manifest.json", manifest)
        base.write_dataset_metadata(dataset_root, metadata, rows)

    for mu_index, mu in enumerate(FRICTIONS):
        case = make_hidden_case(mu_index=mu_index, mu=float(mu), bddl_root=bddl_root)
        episode_index, metrics = base.dataset_rollout(case, dataset=dataset, domain="observation", jitter_xy=JITTER_XY)
        row = {
            "episode_index": int(episode_index),
            "case_id": case.case_id,
            "mu_index": int(mu_index),
            "mu": float(mu),
            "mu_tag": base.mu_tag(float(mu)),
            "is_original_6fric_anchor": bool(any(abs(float(mu) - float(anchor)) < 1e-10 for anchor in ORIGINAL_6FRIC)),
            "A": float(ACTION_A),
            "hold": int(ACTION_HOLD),
            "profile_kind": "event_hold",
            "jitter_xy": list(JITTER_XY),
            "target_visible": False,
            "bddl_file": case.bddl_file,
            "metrics": metrics,
        }
        rows.append(row)
        metadata["episodes"].append(row)
        manifest["episodes"].append(row)
        print(
            f"collect {mu_index + 1:02d}/{len(FRICTIONS):02d} "
            f"{case.case_id} disp={metrics['final_displacement_m'] * 100:.1f}cm "
            f"forward={metrics['final_forward_m'] * 100:.1f}cm lateral={metrics['lateral_m'] * 100:.1f}cm",
            flush=True,
        )
        autosave()

    summary = {
        "episode_count": len(rows),
        "expected_episode_count": len(FRICTIONS),
        "hidden_straight_lerobot": str(dataset_root),
        "frictions": FRICTIONS,
        "count_by_mu": dict(Counter(row["mu_tag"] for row in rows)),
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)
    autosave()
    print(json.dumps(base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

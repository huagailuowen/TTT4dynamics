#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "collect_libero_push_box_event_tap_segmented80_10action_lerobot_2026-07-05_hai-machine.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "pushbox"
    / "libero_push_box_mu023_025_028_030_fric80_9action_hidden_lerobot_2026-07-16_hai-machine"
)

FRICTIONS = [0.23, 0.25, 0.28, 0.30]
# The finalized fric80 collector contains ten actions. Keep its nine canonical
# strengths and omit only the extra A=0.34 bridge between A=0.32 and A=0.36.
OMITTED_SOURCE_ACTION_ID = 7


def load_source_module() -> Any:
    spec = importlib.util.spec_from_file_location("fric80_event_tap_source_hai_machine", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load source collector: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = load_source_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect four high-friction values with nine fric80 event-tap actions."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-codec",
        default=source.VIDEO_CODEC,
        choices=["h264", "hevc", "libsvtav1", "h264_nvenc"],
    )
    parser.add_argument("--video-crf", type=int, default=source.VIDEO_CRF)
    parser.add_argument("--jpeg-quality", type=int, default=source.JPEG_QUALITY)
    return parser.parse_args()


def configure_dataset() -> dict[str, Any]:
    config = source.base.load_config(source.CONFIG_PATH)
    config = source.configure_current_dataset(config)
    actions = [
        dict(action)
        for action in config["actions"]
        if int(action["action_id"]) != OMITTED_SOURCE_ACTION_ID
    ]
    if len(actions) != 9:
        raise ValueError(f"Expected nine selected fric80 actions, got {len(actions)}")
    config["dataset_name"] = (
        "libero_push_box_mu023_025_028_030_fric80_9action_hidden_lerobot_2026-07-16_hai-machine"
    )
    config["frictions"] = list(FRICTIONS)
    config["friction_count"] = len(FRICTIONS)
    config["friction_min"] = min(FRICTIONS)
    config["friction_max"] = max(FRICTIONS)
    config["friction_spacing"] = "explicit high-friction rollout points"
    config["friction_schedule"] = "explicit: [0.23, 0.25, 0.28, 0.30]"
    config["actions"] = actions
    config["action_count"] = len(actions)
    config["source_collector"] = str(SOURCE_SCRIPT)
    config["omitted_source_action"] = {
        "action_id": OMITTED_SOURCE_ACTION_ID,
        "A": 0.34,
        "reason": "extra dense bridge; requested rollout uses nine actions",
    }
    return config


def create_dataset(root: Path, *, config: dict[str, Any], video_codec: str) -> Any:
    return source.base.LeRobotDataset.create(
        repo_id="libero_push_box_mu023_030_fric80_9action_hidden_hai_machine",
        root=root,
        fps=int(config["fps"]),
        features=source.base.build_features(int(config["camera_resolution"])),
        use_videos=True,
        video_codec=video_codec,
        is_compute_episode_stats_image=False,
    )


def relabel_generated_metadata(output_root: Path) -> None:
    labels = {
        output_root / "manifest.json": (
            "libero_push_box_mu023_030_fric80_9action_hidden_lerobot_collection_hai-machine"
        ),
        output_root / "hidden_straight_lerobot" / "push_box_generation_metadata.json": (
            "libero_push_box_mu023_030_fric80_9action_hidden_lerobot_hai-machine"
        ),
    }
    for path, dataset_type in labels.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dataset_type"] = dataset_type
        source.write_json(path, payload)


def main() -> None:
    args = parse_args()
    config = configure_dataset()
    source.create_dataset = create_dataset
    summary = source.collect(
        config,
        output_root=args.output_root.resolve(),
        overwrite=bool(args.overwrite),
        seed=int(args.seed),
        video_codec=str(args.video_codec),
        video_crf=int(args.video_crf),
        jpeg_quality=int(args.jpeg_quality),
    )
    relabel_generated_metadata(args.output_root.resolve())
    print(json.dumps(source.base.to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

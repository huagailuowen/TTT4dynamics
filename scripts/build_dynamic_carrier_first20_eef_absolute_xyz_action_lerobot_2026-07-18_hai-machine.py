#!/usr/bin/env python3
"""Build the first 20 dynamic-carrier episodes with absolute EEF XYZ actions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data/dynamic_carrier_lerobot/"
    "dynamic_carrier_physical_grasp_piecewise_formal_200eps_crf18_2026-07-06_hai-machine"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/dynamic_carrier_lerobot/"
    "dynamic_carrier_physical_grasp_piecewise_formal_first20_crf18_"
    "eef_absolute_xyz_action_2026-07-18_hai-machine"
)
SOURCE_ROLLOUT_METADATA = "dynamic_carrier_physical_grasp_piecewise_formal_metadata.json"
CONVERSION_METADATA = "absolute_action_conversion_2026-07-18_hai-machine.json"
EPISODE_COUNT = 20
TRANSLATION_SCALE_M = 0.05
SOURCE_ACTION_NAMES = ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"]
OUTPUT_ACTION_NAMES = [
    "eef_target_x_m",
    "eef_target_y_m",
    "eef_target_z_m",
    "dax",
    "day",
    "daz",
    "gripper_open",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def action_stats(action: np.ndarray) -> dict[str, Any]:
    return {
        "min": action.min(axis=0).astype(float).tolist(),
        "max": action.max(axis=0).astype(float).tolist(),
        "mean": action.mean(axis=0).astype(float).tolist(),
        "std": action.std(axis=0).astype(float).tolist(),
        "count": [int(action.shape[0])],
    }


def replace_action(table: pa.Table, action: np.ndarray) -> pa.Table:
    flat = pa.array(action.astype(np.float32, copy=False).reshape(-1), type=pa.float32())
    values = pa.FixedSizeListArray.from_arrays(flat, action.shape[1])
    index = table.column_names.index("action")
    return table.set_column(index, pa.field("action", values.type), values)


def episode_path(root: Path, template: str, episode_index: int, chunks_size: int, **extra: str) -> Path:
    return root / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        **extra,
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_info = read_json(source / "meta/info.json")
    if int(source_info["total_episodes"]) < EPISODE_COUNT:
        raise RuntimeError(f"Source only has {source_info['total_episodes']} episodes")
    if source_info["features"]["action"]["names"] != SOURCE_ACTION_NAMES:
        raise RuntimeError(f"Unexpected source action schema: {source_info['features']['action']}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)

    output.mkdir(parents=True)
    chunks_size = int(source_info["chunks_size"])
    data_template = str(source_info["data_path"])
    video_template = str(source_info["video_path"])
    video_keys = [
        name
        for name, feature in source_info["features"].items()
        if feature.get("dtype") == "video"
    ]

    source_episode_rows = read_jsonl(source / "meta/episodes.jsonl")
    selected_episode_rows = [row for row in source_episode_rows if int(row["episode_index"]) < EPISODE_COUNT]
    if [int(row["episode_index"]) for row in selected_episode_rows] != list(range(EPISODE_COUNT)):
        raise RuntimeError("Source episodes 0-19 are not complete and ordered")
    source_stats_rows = read_jsonl(source / "meta/episodes_stats.jsonl")
    stats_by_episode = {int(row["episode_index"]): row for row in source_stats_rows}

    total_frames = 0
    global_min = np.full(7, np.inf, dtype=np.float64)
    global_max = np.full(7, -np.inf, dtype=np.float64)
    source_parquets: list[Path] = []
    output_parquets: list[Path] = []
    for episode_index in range(EPISODE_COUNT):
        source_path = episode_path(source, data_template, episode_index, chunks_size)
        output_path = episode_path(output, data_template, episode_index, chunks_size)
        table = pq.read_table(source_path)
        if table.num_rows != int(selected_episode_rows[episode_index]["length"]):
            raise RuntimeError(f"Episode length mismatch at {episode_index}")
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        if not np.all(episode_values == episode_index):
            raise RuntimeError(f"Episode index mismatch in {source_path.name}")

        observation = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        relative_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        absolute_action = relative_action.copy()
        absolute_action[:, :3] = observation[:, :3] + np.float32(TRANSLATION_SCALE_M) * relative_action[:, :3]
        converted = replace_action(table, absolute_action)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".parquet.tmp")
        pq.write_table(converted, temporary, compression="zstd")
        temporary.replace(output_path)

        stats_by_episode[episode_index]["stats"]["action"] = action_stats(absolute_action)
        total_frames += table.num_rows
        global_min = np.minimum(global_min, absolute_action.min(axis=0))
        global_max = np.maximum(global_max, absolute_action.max(axis=0))
        source_parquets.append(source_path)
        output_parquets.append(output_path)
        print(f"[convert] {episode_index + 1:02d}/{EPISODE_COUNT}", flush=True)

    linked_video_pairs: list[tuple[Path, Path]] = []
    for episode_index in range(EPISODE_COUNT):
        for video_key in video_keys:
            source_video = episode_path(
                source,
                video_template,
                episode_index,
                chunks_size,
                video_key=video_key,
            )
            output_video = episode_path(
                output,
                video_template,
                episode_index,
                chunks_size,
                video_key=video_key,
            )
            if not source_video.is_file() or source_video.stat().st_size == 0:
                raise RuntimeError(f"Missing source video: {source_video}")
            output_video.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_video, output_video)
            linked_video_pairs.append((source_video, output_video))

    output_info = json.loads(json.dumps(source_info))
    output_info["total_episodes"] = EPISODE_COUNT
    output_info["total_frames"] = total_frames
    output_info["total_videos"] = EPISODE_COUNT * len(video_keys)
    output_info["splits"] = {"train": f"0:{EPISODE_COUNT}"}
    output_info["features"]["action"]["names"] = OUTPUT_ACTION_NAMES
    write_json(output / "meta/info.json", output_info)
    write_jsonl(output / "meta/episodes.jsonl", selected_episode_rows)
    write_jsonl(
        output / "meta/episodes_stats.jsonl",
        [stats_by_episode[index] for index in range(EPISODE_COUNT)],
    )
    shutil.copy2(source / "meta/tasks.jsonl", output / "meta/tasks.jsonl")

    rollout_metadata = read_json(source / SOURCE_ROLLOUT_METADATA)
    selected_successes = [
        row for row in rollout_metadata["successes"] if int(row["episode_index"]) < EPISODE_COUNT
    ]
    if [int(row["episode_index"]) for row in selected_successes] != list(range(EPISODE_COUNT)):
        raise RuntimeError("Rollout metadata does not contain complete episodes 0-19")
    rollout_metadata["episodes_requested"] = EPISODE_COUNT
    rollout_metadata["episodes_collected"] = EPISODE_COUNT
    rollout_metadata["attempts"] = EPISODE_COUNT
    rollout_metadata["successes"] = selected_successes
    rollout_metadata["failures"] = []
    rollout_metadata["derived_dataset"] = {
        "source_dataset": str(source),
        "source_episode_range": [0, EPISODE_COUNT],
        "action_schema": "absolute_eef_xyz_action_v1",
        "conversion_metadata": f"meta/{CONVERSION_METADATA}",
        "note": "All rollout configuration and per-frame records are copied from source episodes 0-19.",
    }
    write_json(output / SOURCE_ROLLOUT_METADATA, rollout_metadata)

    conversion = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schema": "absolute_eef_xyz_action_v1",
        "source_dataset": str(source),
        "output_dataset": str(output),
        "source_episode_range": [0, EPISODE_COUNT],
        "paired_episode_count": EPISODE_COUNT,
        "paired_frame_count": total_frames,
        "source_action": SOURCE_ACTION_NAMES,
        "output_action": OUTPUT_ACTION_NAMES,
        "translation_formula": "eef_target_xyz_m = observation_eef_xyz_m + 0.05 * normalized_dxyz",
        "translation_scale_m_per_normalized_unit": TRANSLATION_SCALE_M,
        "orientation_semantics": "relative axis-angle command unchanged from source",
        "gripper_semantics": "gripper_open unchanged from source",
        "rollout_metadata_action_env_semantics": "raw LIBERO controller command unchanged from source",
        "terminal_frame_semantics": "zero relative command maps to current observed EEF XYZ",
        "non_action_columns": "value-identical to source episodes 0-19",
        "video_storage": "hardlinked byte-identical CRF18 files from source",
        "absolute_action_min": global_min.astype(float).tolist(),
        "absolute_action_max": global_max.astype(float).tolist(),
    }
    write_json(output / "meta" / CONVERSION_METADATA, conversion)

    max_formula_error = 0.0
    for source_path, output_path in zip(source_parquets, output_parquets):
        source_table = pq.read_table(source_path)
        output_table = pq.read_table(output_path)
        for column in source_table.column_names:
            if column != "action" and not source_table[column].equals(output_table[column]):
                raise RuntimeError(f"Non-action column changed: {source_path.name}:{column}")
        observation = np.asarray(source_table["observation.state"].to_pylist(), dtype=np.float32)
        relative_action = np.asarray(source_table["action"].to_pylist(), dtype=np.float32)
        absolute_action = np.asarray(output_table["action"].to_pylist(), dtype=np.float32)
        expected_xyz = observation[:, :3] + np.float32(TRANSLATION_SCALE_M) * relative_action[:, :3]
        max_formula_error = max(max_formula_error, float(np.max(np.abs(absolute_action[:, :3] - expected_xyz))))
        if not np.array_equal(absolute_action[:, 3:], relative_action[:, 3:]):
            raise RuntimeError(f"Rotation or gripper changed: {source_path.name}")
    if max_formula_error != 0.0:
        raise RuntimeError(f"Absolute action formula error: {max_formula_error}")
    if not all(
        source_video.stat().st_ino == output_video.stat().st_ino
        and source_video.stat().st_size == output_video.stat().st_size
        for source_video, output_video in linked_video_pairs
    ):
        raise RuntimeError("Output videos are not byte-identical hardlinks")

    result = {
        "status": "complete",
        "episodes": EPISODE_COUNT,
        "frames": total_frames,
        "videos": len(linked_video_pairs),
        "max_formula_error": max_formula_error,
        "non_action_columns_identical": True,
        "rotation_and_gripper_identical": True,
        "videos_byte_identical": True,
        "output": str(output),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

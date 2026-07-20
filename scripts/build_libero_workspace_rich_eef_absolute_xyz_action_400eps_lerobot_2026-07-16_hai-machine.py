#!/usr/bin/env python3
"""Build a frame-paired LeRobot dataset with absolute EEF XYZ target actions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    HERE.parent
    / "data/various_actions/"
    "libero_mu0100_workspace_rich_eef_400eps_lerobot_2026-07-16_hai-machine"
)
DEFAULT_OUTPUT = (
    HERE.parent
    / "data/various_actions/"
    "libero_mu0100_workspace_rich_eef_absolute_xyz_action_400eps_lerobot_2026-07-16_hai-machine"
)
TRANSLATION_SCALE_M = 0.05
EXPECTED_EPISODES = 400
EXPECTED_VIDEOS = 800


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def replace_action(table: pa.Table, absolute_action: np.ndarray) -> pa.Table:
    flat = pa.array(absolute_action.astype(np.float32, copy=False).reshape(-1), type=pa.float32())
    values = pa.FixedSizeListArray.from_arrays(flat, absolute_action.shape[1])
    index = table.column_names.index("action")
    field = pa.field("action", values.type)
    return table.set_column(index, field, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_info = json.loads((source / "meta/info.json").read_text())
    if source_info["total_episodes"] != EXPECTED_EPISODES:
        raise RuntimeError(f"Source has {source_info['total_episodes']} episodes")
    action_feature = source_info["features"]["action"]
    if action_feature["names"] != ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"]:
        raise RuntimeError(f"Unexpected source action schema: {action_feature}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)

    shutil.copytree(source, output, copy_function=os.link)
    stats_path = output / "meta/episodes_stats.jsonl"
    stats_rows = read_jsonl(stats_path)
    stats_by_episode = {int(row["episode_index"]): row for row in stats_rows}
    source_parquets = sorted((source / "data").glob("chunk-*/*.parquet"))
    if len(source_parquets) != EXPECTED_EPISODES:
        raise RuntimeError(f"Source parquet count is {len(source_parquets)}")

    total_frames = 0
    global_min = np.full(7, np.inf, dtype=np.float64)
    global_max = np.full(7, -np.inf, dtype=np.float64)
    for index, source_path in enumerate(source_parquets):
        relative_path = source_path.relative_to(source)
        output_path = output / relative_path
        table = pq.read_table(source_path)
        observation = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        relative_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        absolute_action = relative_action.copy()
        absolute_action[:, :3] = (
            observation[:, :3]
            + np.float32(TRANSLATION_SCALE_M) * relative_action[:, :3]
        )
        converted = replace_action(table, absolute_action)
        temporary = output_path.with_suffix(".parquet.tmp")
        pq.write_table(converted, temporary, compression="zstd")
        temporary.replace(output_path)

        episode_index = int(np.asarray(table["episode_index"].to_pylist())[0])
        if episode_index != index:
            raise RuntimeError(f"Unexpected episode order: file={index}, row={episode_index}")
        stats_by_episode[episode_index]["stats"]["action"] = action_stats(absolute_action)
        total_frames += len(absolute_action)
        global_min = np.minimum(global_min, absolute_action.min(axis=0))
        global_max = np.maximum(global_max, absolute_action.max(axis=0))
        if (index + 1) % 25 == 0 or index + 1 == EXPECTED_EPISODES:
            print(f"[convert] {index + 1:03d}/{EXPECTED_EPISODES}", flush=True)

    write_jsonl(stats_path, [stats_by_episode[index] for index in range(EXPECTED_EPISODES)])
    output_info = dict(source_info)
    output_info["features"] = json.loads(json.dumps(source_info["features"]))
    output_info["features"]["action"]["names"] = [
        "eef_target_x_m",
        "eef_target_y_m",
        "eef_target_z_m",
        "dax",
        "day",
        "daz",
        "gripper_open",
    ]
    info_path = output / "meta/info.json"
    info_temporary = info_path.with_suffix(".json.tmp")
    info_temporary.write_text(json.dumps(output_info, indent=4), encoding="utf-8")
    info_temporary.replace(info_path)

    plan_manifest_path = source / "meta/collection_plan_manifest_2026-07-16_hai-machine.json"
    plan_manifest = json.loads(plan_manifest_path.read_text())
    conversion = {
        "schema": "absolute_eef_xyz_action_v1",
        "source_dataset": str(source),
        "paired_episode_count": EXPECTED_EPISODES,
        "paired_frame_count": total_frames,
        "source_plan_sha256": plan_manifest["plan_sha256"],
        "source_action": ["dx", "dy", "dz", "dax", "day", "daz", "gripper_open"],
        "output_action": output_info["features"]["action"]["names"],
        "translation_formula": "eef_target_xyz_m = observation_eef_xyz_m + 0.05 * normalized_dxyz",
        "translation_scale_m_per_normalized_unit": TRANSLATION_SCALE_M,
        "orientation_semantics": "relative axis-angle command unchanged from source",
        "gripper_semantics": "gripper_open unchanged from source",
        "terminal_frame_semantics": "zero relative command maps to current observed EEF XYZ",
        "video_storage": "hardlinked byte-identical files from source",
        "absolute_action_min": global_min.astype(float).tolist(),
        "absolute_action_max": global_max.astype(float).tolist(),
    }
    conversion_path = output / "meta/absolute_action_conversion_2026-07-16_hai-machine.json"
    conversion_path.write_text(json.dumps(conversion, indent=2), encoding="utf-8")

    output_parquets = sorted((output / "data").glob("chunk-*/*.parquet"))
    source_videos = sorted((source / "videos").glob("chunk-*/*/*.mp4"))
    output_videos = sorted((output / "videos").glob("chunk-*/*/*.mp4"))
    if len(output_parquets) != EXPECTED_EPISODES:
        raise RuntimeError("Output parquet count mismatch")
    if len(source_videos) != EXPECTED_VIDEOS or len(output_videos) != EXPECTED_VIDEOS:
        raise RuntimeError("Video count mismatch")
    if not all(
        source_video.stat().st_ino == output_video.stat().st_ino
        and source_video.stat().st_size == output_video.stat().st_size
        and output_video.stat().st_size > 0
        for source_video, output_video in zip(source_videos, output_videos)
    ):
        raise RuntimeError("Videos are not byte-identical hardlinks")

    max_formula_error = 0.0
    for source_path, output_path in zip(source_parquets, output_parquets):
        source_table = pq.read_table(source_path)
        output_table = pq.read_table(output_path)
        if source_table.num_rows != output_table.num_rows:
            raise RuntimeError(f"Row mismatch: {source_path.name}")
        for column in source_table.column_names:
            if column != "action" and not source_table[column].equals(output_table[column]):
                raise RuntimeError(f"Non-action column changed: {source_path.name}:{column}")
        observation = np.asarray(source_table["observation.state"].to_pylist(), dtype=np.float32)
        relative_action = np.asarray(source_table["action"].to_pylist(), dtype=np.float32)
        absolute_action = np.asarray(output_table["action"].to_pylist(), dtype=np.float32)
        expected_xyz = observation[:, :3] + np.float32(TRANSLATION_SCALE_M) * relative_action[:, :3]
        max_formula_error = max(
            max_formula_error,
            float(np.max(np.abs(absolute_action[:, :3] - expected_xyz))),
        )
        if not np.array_equal(absolute_action[:, 3:], relative_action[:, 3:]):
            raise RuntimeError(f"Rotation/gripper changed: {source_path.name}")
    if max_formula_error != 0.0:
        raise RuntimeError(f"Absolute action formula error: {max_formula_error}")

    result = {
        "status": "complete",
        "episodes": EXPECTED_EPISODES,
        "frames": total_frames,
        "videos": EXPECTED_VIDEOS,
        "max_formula_error": max_formula_error,
        "non_action_columns_identical": True,
        "rotation_and_gripper_identical": True,
        "videos_byte_identical": True,
        "output": str(output),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

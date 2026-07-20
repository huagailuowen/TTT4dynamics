#!/usr/bin/env python3
"""Crop the first 20 absolute-action carrier episodes to robot execution only."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
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
    "dynamic_carrier_physical_grasp_piecewise_formal_first20_crf18_"
    "eef_absolute_xyz_action_2026-07-18_hai-machine"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/dynamic_carrier_lerobot/"
    "dynamic_carrier_physical_grasp_piecewise_formal_first20_execute_only_crf18_"
    "eef_absolute_xyz_action_2026-07-18_hai-machine"
)
ROLLOUT_METADATA = "dynamic_carrier_physical_grasp_piecewise_formal_metadata.json"
ABSOLUTE_ACTION_METADATA = "absolute_action_conversion_2026-07-18_hai-machine.json"
TEMPORAL_CROP_METADATA = "temporal_crop_2026-07-18_hai-machine.json"
EPISODE_COUNT = 20
EXPECTED_ACTION_NAMES = [
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
    parser.add_argument("--video-crf", type=int, default=18)
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


def episode_path(root: Path, template: str, episode_index: int, chunks_size: int, **extra: str) -> Path:
    return root / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        **extra,
    )


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.column_names.index(name)
    field = table.schema.field(name)
    return table.set_column(index, field, pa.array(values, type=field.type))


def feature_stats(table: pa.Table, info: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name, feature in info["features"].items():
        if feature.get("dtype") == "video" or name not in table.column_names:
            continue
        values = np.asarray(table[name].to_pylist())
        if values.ndim == 1:
            values = values[:, None]
        stats[name] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [int(values.shape[0])],
        }
    return stats


def crop_video(source: Path, output: Path, start_frame: int, frame_count: int, crf: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"trim=start_frame={start_frame},setpts=PTS-STARTPTS",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate,width,height,pix_fmt,codec_name",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream: {path}")
    return streams[0]


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_info = read_json(source / "meta/info.json")
    if int(source_info["total_episodes"]) != EPISODE_COUNT:
        raise RuntimeError(f"Expected {EPISODE_COUNT} source episodes")
    if source_info["features"]["action"]["names"] != EXPECTED_ACTION_NAMES:
        raise RuntimeError(f"Unexpected action schema: {source_info['features']['action']}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    fps = int(source_info["fps"])
    chunks_size = int(source_info["chunks_size"])
    data_template = str(source_info["data_path"])
    video_template = str(source_info["video_path"])
    video_keys = [
        name
        for name, feature in source_info["features"].items()
        if feature.get("dtype") == "video"
    ]
    source_episode_rows = read_jsonl(source / "meta/episodes.jsonl")
    if [int(row["episode_index"]) for row in source_episode_rows] != list(range(EPISODE_COUNT)):
        raise RuntimeError("Source episode metadata is not complete and ordered")

    output_episode_rows: list[dict[str, Any]] = []
    output_stats_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    table_pairs: list[tuple[Path, Path, int]] = []
    global_index = 0
    action_min = np.full(7, np.inf, dtype=np.float64)
    action_max = np.full(7, -np.inf, dtype=np.float64)

    for episode_index in range(EPISODE_COUNT):
        source_path = episode_path(source, data_template, episode_index, chunks_size)
        output_path = episode_path(output, data_template, episode_index, chunks_size)
        source_table = pq.read_table(source_path)
        dynamic_time = np.asarray(source_table["observation.dynamic_time"].to_pylist(), dtype=np.float32)
        execute_frames = np.flatnonzero(dynamic_time[:, 3] > 0.5)
        if len(execute_frames) == 0:
            raise RuntimeError(f"Episode {episode_index} has no execution frames")
        start_frame = int(execute_frames[0])
        if not np.all(dynamic_time[start_frame:, 3] > 0.5):
            raise RuntimeError(f"Episode {episode_index} returns to observation after execution starts")
        output_table = source_table.slice(start_frame)
        frame_count = output_table.num_rows
        frame_index = np.arange(frame_count, dtype=np.int64)
        timestamp = frame_index.astype(np.float64) / float(fps)
        index = np.arange(global_index, global_index + frame_count, dtype=np.int64)
        output_table = replace_column(output_table, "frame_index", frame_index)
        output_table = replace_column(output_table, "timestamp", timestamp)
        output_table = replace_column(output_table, "index", index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".parquet.tmp")
        pq.write_table(output_table, temporary, compression="zstd")
        temporary.replace(output_path)

        actions = np.asarray(output_table["action"].to_pylist(), dtype=np.float32)
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        output_episode_rows.append(
            {
                **source_episode_rows[episode_index],
                "length": frame_count,
            }
        )
        output_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": feature_stats(output_table, source_info),
            }
        )
        crop_rows.append(
            {
                "episode_index": episode_index,
                "source_start_frame": start_frame,
                "source_length": source_table.num_rows,
                "output_length": frame_count,
                "source_start_sim_time_s": float(dynamic_time[start_frame, 0]),
                "output_start_timestamp_s": 0.0,
            }
        )
        table_pairs.append((source_path, output_path, start_frame))
        global_index += frame_count
        print(
            f"[parquet] {episode_index + 1:02d}/{EPISODE_COUNT} "
            f"start={start_frame} frames={frame_count}",
            flush=True,
        )

    output_info = json.loads(json.dumps(source_info))
    output_info["total_frames"] = global_index
    output_info["total_videos"] = EPISODE_COUNT * len(video_keys)
    output_info["splits"] = {"train": f"0:{EPISODE_COUNT}"}
    write_json(output / "meta/info.json", output_info)
    write_jsonl(output / "meta/episodes.jsonl", output_episode_rows)
    write_jsonl(output / "meta/episodes_stats.jsonl", output_stats_rows)
    shutil.copy2(source / "meta/tasks.jsonl", output / "meta/tasks.jsonl")

    for crop in crop_rows:
        episode_index = int(crop["episode_index"])
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
            crop_video(
                source_video,
                output_video,
                int(crop["source_start_frame"]),
                int(crop["output_length"]),
                int(args.video_crf),
            )
        print(f"[video] {episode_index + 1:02d}/{EPISODE_COUNT}", flush=True)

    rollout_metadata = read_json(source / ROLLOUT_METADATA)
    successes_by_episode = {
        int(row["episode_index"]): row for row in rollout_metadata["successes"]
    }
    cropped_successes: list[dict[str, Any]] = []
    for crop in crop_rows:
        episode_index = int(crop["episode_index"])
        success = successes_by_episode[episode_index]
        source_rows = success["rows"]
        start_frame = int(crop["source_start_frame"])
        if len(source_rows) != int(crop["source_length"]):
            raise RuntimeError(f"Rollout row count mismatch at episode {episode_index}")
        kept_rows: list[dict[str, Any]] = []
        for new_frame, source_row in enumerate(source_rows[start_frame:]):
            row = dict(source_row)
            row["source_frame"] = int(source_row["frame"])
            row["frame"] = new_frame
            kept_rows.append(row)
        first_row = kept_rows[0]
        success["source_steps"] = int(success["steps"])
        success["steps"] = len(kept_rows)
        success["rows"] = kept_rows
        success["temporal_crop"] = dict(crop)
        success["dataset_start_conditions"] = {
            "source_frame": int(first_row["source_frame"]),
            "sim_time_s": float(first_row["t"]),
            "trajectory_phase": float(first_row["trajectory_phase"]),
            "carrier_xy": first_row["carrier_xy"],
            "carrier_velocity_xy": first_row["carrier_velocity_xy"],
            "payload_xyz": first_row["payload_xyz"],
            "object_poses": first_row["object_poses"],
            "eef_xyz": first_row["eef_xyz"],
            "eef_quat_wxyz": first_row["eef_quat_wxyz"],
            "first_action_env": first_row["action_env"],
        }
        cropped_successes.append(success)

    rollout_metadata["source_dataset_type"] = rollout_metadata["dataset_type"]
    rollout_metadata["dataset_type"] = (
        "dynamic_carrier_physical_grasp_piecewise_formal_execute_only_"
        "absolute_eef_xyz_action"
    )
    rollout_metadata["source_observe_frames"] = int(rollout_metadata["observe_frames"])
    rollout_metadata["observe_frames"] = 0
    rollout_metadata["successes"] = cropped_successes
    rollout_metadata["derived_dataset"] = {
        "source_dataset": str(source),
        "source_episode_range": [0, EPISODE_COUNT],
        "action_schema": "absolute_eef_xyz_action_v1",
        "temporal_scope": "first robot execution frame through original episode end",
        "crop_metadata": f"meta/{TEMPORAL_CROP_METADATA}",
        "source_initial_conditions_note": (
            "initial_conditions remains the original reset state; dataset_start_conditions "
            "records the state at the cropped execution boundary"
        ),
    }
    write_json(output / ROLLOUT_METADATA, rollout_metadata)

    absolute_action_metadata = read_json(source / "meta" / ABSOLUTE_ACTION_METADATA)
    absolute_action_metadata["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    absolute_action_metadata["derived_from_absolute_action_dataset"] = str(source)
    absolute_action_metadata["output_dataset"] = str(output)
    absolute_action_metadata["paired_frame_count"] = global_index
    absolute_action_metadata["absolute_action_min"] = action_min.astype(float).tolist()
    absolute_action_metadata["absolute_action_max"] = action_max.astype(float).tolist()
    absolute_action_metadata["temporal_scope"] = "execution only; action values unchanged from source"
    write_json(output / "meta" / ABSOLUTE_ACTION_METADATA, absolute_action_metadata)

    crop_metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schema": "dynamic_carrier_execution_only_crop_v1",
        "source_dataset": str(source),
        "output_dataset": str(output),
        "boundary_rule": "first frame where observation.dynamic_time.is_execute > 0.5",
        "retained_interval": "boundary frame through original episode end, inclusive",
        "action_semantics": "absolute_eef_xyz_action_v1; values unchanged",
        "dynamic_time_semantics": "original simulation time and trajectory phase retained",
        "timestamp_semantics": "reset to zero and spaced at 1/fps within each cropped episode",
        "frame_index_semantics": "reset to zero within each cropped episode",
        "index_semantics": "rebuilt as one contiguous global sequence over the cropped dataset",
        "metadata_row_semantics": "frame renumbered; source_frame preserves original frame index",
        "video_encoding": {
            "codec": "h264",
            "crf": int(args.video_crf),
            "pixel_format": "yuv420p",
            "fps": fps,
        },
        "episodes": crop_rows,
        "total_frames": global_index,
        "total_videos": EPISODE_COUNT * len(video_keys),
    }
    write_json(output / "meta" / TEMPORAL_CROP_METADATA, crop_metadata)

    expected_global_index = 0
    for source_path, output_path, start_frame in table_pairs:
        source_table = pq.read_table(source_path).slice(start_frame)
        output_table = pq.read_table(output_path)
        if source_table.num_rows != output_table.num_rows:
            raise RuntimeError(f"Cropped row count mismatch: {output_path.name}")
        for column in source_table.column_names:
            if column in {"timestamp", "frame_index", "index"}:
                continue
            if not source_table[column].equals(output_table[column]):
                raise RuntimeError(f"Retained column changed: {output_path.name}:{column}")
        count = output_table.num_rows
        if output_table["frame_index"].to_pylist() != list(range(count)):
            raise RuntimeError(f"Invalid frame index: {output_path.name}")
        if output_table["index"].to_pylist() != list(
            range(expected_global_index, expected_global_index + count)
        ):
            raise RuntimeError(f"Invalid global index: {output_path.name}")
        expected_global_index += count

    for crop in crop_rows:
        episode_index = int(crop["episode_index"])
        for video_key in video_keys:
            video_path = episode_path(
                output,
                video_template,
                episode_index,
                chunks_size,
                video_key=video_key,
            )
            stream = probe_video(video_path)
            if int(stream["nb_read_frames"]) != int(crop["output_length"]):
                raise RuntimeError(f"Video frame mismatch: {video_path}")
            if stream["r_frame_rate"] != f"{fps}/1":
                raise RuntimeError(f"Video FPS mismatch: {video_path}")
            if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p":
                raise RuntimeError(f"Video format mismatch: {video_path}")

    result = {
        "status": "complete",
        "episodes": EPISODE_COUNT,
        "frames": global_index,
        "videos": EPISODE_COUNT * len(video_keys),
        "source_start_frames": sorted({int(row["source_start_frame"]) for row in crop_rows}),
        "retained_columns_identical": True,
        "absolute_actions_identical": True,
        "video_frame_counts_match_parquet": True,
        "output": str(output),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Watch the formal 400-episode collection and persist structural health status."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROGRESS_RE = re.compile(r"^\[(\d{3})/(\d{3})\].*$", re.MULTILINE)
RETRY_RE = re.compile(r"^\[retry\].*$", re.MULTILINE)
ERROR_MARKERS = ("Traceback (most recent call last):", "RuntimeError:", "AssertionError:")


def session_running(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def video_stats(dataset: Path, key: str) -> tuple[int, int]:
    root = dataset / "videos" / "chunk-000" / key
    videos = list(root.glob("episode_*.mp4")) if root.exists() else []
    return len(videos), sum(path.stat().st_size == 0 for path in videos)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-session", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--base-episodes", type=int, default=0)
    parser.add_argument("--target-episodes", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stopped_checks = 0
    while True:
        log_text = args.log.read_text(errors="replace") if args.log.exists() else ""
        progress_lines = PROGRESS_RE.findall(log_text.replace("\r", "\n"))
        local_progress = int(progress_lines[-1][0]) if progress_lines else 0
        local_expected = int(progress_lines[-1][1]) if progress_lines else 0
        progress = args.base_episodes + local_progress
        retry_lines = RETRY_RE.findall(log_text.replace("\r", "\n"))
        collector_alive = session_running(args.collector_session)
        stopped_checks = 0 if collector_alive else stopped_checks + 1
        metadata_episodes = count_lines(args.dataset / "meta" / "episodes.jsonl")
        image_videos, image_zero = video_stats(args.dataset, "observation.images.image")
        wrist_videos, wrist_zero = video_stats(args.dataset, "observation.images.wrist_image")
        errors = [marker for marker in ERROR_MARKERS if marker in log_text]

        state = "running"
        if errors:
            state = "failed"
        elif progress == args.target_episodes and not collector_alive:
            if (
                metadata_episodes == args.target_episodes
                and image_videos == args.target_episodes
                and wrist_videos == args.target_episodes
                and image_zero == 0
                and wrist_zero == 0
            ):
                state = "complete"
            else:
                state = "incomplete"
        elif stopped_checks >= 3:
            state = "stopped_unexpectedly"

        status = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "collector_session": args.collector_session,
            "collector_alive": collector_alive,
            "progress": progress,
            "base_episodes": args.base_episodes,
            "local_progress": local_progress,
            "local_expected": local_expected,
            "resampled_retry_count": len(retry_lines),
            "latest_retry": retry_lines[-1] if retry_lines else None,
            "expected_episodes": args.target_episodes,
            "metadata_episodes": metadata_episodes,
            "image_videos": image_videos,
            "wrist_videos": wrist_videos,
            "zero_byte_image_videos": image_zero,
            "zero_byte_wrist_videos": wrist_zero,
            "error_markers": errors,
        }
        atomic_json(args.status, status)
        print(json.dumps(status, sort_keys=True), flush=True)

        if state in {"complete", "failed", "incomplete", "stopped_unexpectedly"}:
            return 0 if state == "complete" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

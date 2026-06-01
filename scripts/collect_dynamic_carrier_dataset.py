#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt4dynamics.cases import load_cases
from ttt4dynamics.dataset import CollectionConfig, collect_dataset, validate_static_gates
from ttt4dynamics.planner import PlannerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scripted demonstrations for dynamic carrier LIBERO tasks."
    )
    parser.add_argument("--cases", type=Path, required=True, help="JSON case config.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .h5/.hdf5 path, or a directory path for NPZ+JSON output.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="TTT4dynamics repo root.")
    parser.add_argument("--episodes-per-case", type=int, default=10)
    parser.add_argument("--camera-resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-static-gate", action="store_true")
    parser.add_argument("--static-success-threshold", type=float, default=0.95)
    parser.add_argument("--intercept-lead-s", type=float, default=0.35)
    parser.add_argument("--max-pos-action", type=float, default=1.0)
    parser.add_argument("--position-gain", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    planner_config = PlannerConfig(
        intercept_lead_s=args.intercept_lead_s,
        max_pos_action=args.max_pos_action,
        position_gain=args.position_gain,
    )
    collection_config = CollectionConfig(
        output_path=args.output,
        repo_root=args.repo_root,
        camera_resolution=args.camera_resolution,
        episodes_per_case=args.episodes_per_case,
        seed=args.seed,
        static_success_threshold=args.static_success_threshold,
        require_static_gate=args.require_static_gate,
    )

    for case in cases:
        min_dist = case.validate_target_separation()
        print(f"{case.case_id}: target/path min distance = {min_dist:.3f} m")

    if args.validate_only:
        rates = validate_static_gates(cases, collection_config, planner_config=planner_config)
        for case_id, rate in rates.items():
            print(f"{case_id}: static scripted success rate = {rate:.3f}")
        return

    collect_dataset(cases, collection_config, planner_config=planner_config)
    print(f"Wrote dataset: {args.output}")


if __name__ == "__main__":
    main()

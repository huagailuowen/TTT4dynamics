#!/usr/bin/env python3
"""Build a collision-speed lookup without collecting a new dataset.

The useful outcome variable for this experiment is the target block's first
free-sliding velocity after the projectile/target collision.  Under nominal
Coulomb friction, stopping distance is proportional to that velocity squared:

    d = v_post**2 / (2 * mu * g)

The script reads the existing 9-action x 20-mass rollout diagnostics, preserves
off-table episodes, ranks actions by collision regularity, and writes both the
measured lookup and a fitted 20-level grid with linearly spaced theoretical
stopping distances.  It never modifies or regenerates the LeRobot dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/libero_two_box_collision_9speed_20mass_2026-07-16_hai-machine.json"
)
DEFAULT_EPISODES = REPO_ROOT / "data/mass/original_mass_grid_9speed_20mass_2026-07-16_hai-machine/episodes.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/mass/original_mass_grid_9speed_20mass_2026-07-16_hai-machine/analysis/postimpact_speed_lookup_2026-07-17_hai-machine"
)
GRAVITY_MPS2 = 9.81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a post-collision speed and theoretical-distance lookup."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_diagnostics_path(raw_path: str, episodes_path: Path) -> Path:
    path = Path(raw_path)
    candidates = [
        path,
        REPO_ROOT / path,
        episodes_path.parent / "diagnostics" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve diagnostics file: {raw_path}")


def merged_contact_segments(flags: list[bool], allowed_false_gap: int = 1) -> list[tuple[int, int]]:
    contact_positions = [index for index, active in enumerate(flags) if active]
    if not contact_positions:
        return []
    segments: list[list[int]] = [[contact_positions[0], contact_positions[0]]]
    for position in contact_positions[1:]:
        if position - segments[-1][1] <= allowed_false_gap + 1:
            segments[-1][1] = position
        else:
            segments.append([position, position])
    return [(start, end) for start, end in segments]


def vector_xy(record: dict[str, Any], key: str) -> tuple[float, float]:
    values = record[key]
    return float(values[0]), float(values[1])


def extract_episode(
    episode: dict[str, Any],
    *,
    episodes_path: Path,
    friction_mu: float,
    projectile_mass_kg: float,
    impact_lateral_limit_m: float,
) -> dict[str, Any]:
    metrics = episode["metrics"]
    diagnostics_path = resolve_diagnostics_path(episode["diagnostics_file"], episodes_path)
    records = load_json(diagnostics_path)
    if not records:
        raise RuntimeError(f"Empty diagnostics: {diagnostics_path}")

    contact_flags = [bool(record["block_block_contact"]) for record in records]
    segments = merged_contact_segments(contact_flags, allowed_false_gap=1)
    transfer_frame = int(metrics["first_block_collision_frame"])
    transfer_position = next(
        (
            index
            for index, record in enumerate(records)
            if int(record["frame_index"]) >= transfer_frame
        ),
        len(records) - 1,
    )

    if segments:
        first_contact_start, first_contact_end = segments[0]
        post_position = min(first_contact_end + 1, len(records) - 1)
        source = (
            "first_free_frame_after_first_contact"
            if post_position > first_contact_end
            else "last_recorded_contact_frame"
        )
        local_start = first_contact_start
        local_end = min(post_position + 1, len(records) - 1)
    else:
        first_contact_start = None
        first_contact_end = None
        post_position = transfer_position
        source = "first_transfer_frame_no_sampled_contact"
        local_start = transfer_position
        local_end = min(transfer_position + 2, len(records) - 1)

    post_record = records[post_position]
    post_vx, post_vy = vector_xy(post_record, "target_velocity")
    post_speed_xy = math.hypot(post_vx, post_vy)
    local_peak_vx = max(
        float(records[index]["target_velocity"][0])
        for index in range(local_start, local_end + 1)
    )
    global_peak_vx = max(float(record["target_velocity"][0]) for record in records)

    theory_distance_x_m = max(post_vx, 0.0) ** 2 / (2.0 * friction_mu * GRAVITY_MPS2)
    theory_distance_planar_m = post_speed_xy**2 / (2.0 * friction_mu * GRAVITY_MPS2)
    preimpact_vx = float(metrics["preimpact_projectile_vx_mps"])
    target_mass_kg = float(episode["target_mass_kg"])
    collision_factor = (
        post_vx * (projectile_mass_kg + target_mass_kg)
        / (projectile_mass_kg * preimpact_vx)
        if preimpact_vx > 0.0
        else math.nan
    )
    lateral_speed_ratio = abs(post_vy) / max(abs(post_vx), 1e-12)
    impact_lateral_offset_m = float(metrics["impact_lateral_offset_m"])
    robot_target_contact_count = len(metrics["robot_target_contact_frames"])
    first_contact_frame = (
        int(records[first_contact_start]["frame_index"])
        if first_contact_start is not None
        else None
    )
    separation_frame = int(post_record["frame_index"])
    final_target_xyz = metrics["final_target_xyz"]
    on_table_at_60f = float(final_target_xyz[2]) >= 0.85
    lookup_usable = (
        post_vx > 0.0
        and math.isfinite(post_vx)
        and len(segments) <= 1
        and robot_target_contact_count == 0
        and impact_lateral_offset_m <= impact_lateral_limit_m
        and lateral_speed_ratio <= 0.10
    )

    return {
        "episode_index": int(episode["episode_index"]),
        "action_id": int(episode["action_id"]),
        "A": float(episode["A"]),
        "push_steps": int(episode["push_steps"]),
        "target_mass_kg": target_mass_kg,
        "target_mass_g": target_mass_kg * 1000.0,
        "projectile_mass_kg": projectile_mass_kg,
        "preimpact_projectile_vx_mps": preimpact_vx,
        "contact_observed_at_sample_time": bool(segments),
        "merged_contact_segment_count": len(segments),
        "first_contact_frame": first_contact_frame,
        "postcollision_sample_frame": separation_frame,
        "postcollision_speed_source": source,
        "target_postcollision_vx_mps": post_vx,
        "target_postcollision_vy_mps": post_vy,
        "target_postcollision_speed_xy_mps": post_speed_xy,
        "first_collision_local_peak_vx_mps": local_peak_vx,
        "target_global_peak_vx_mps": global_peak_vx,
        "nominal_theory_distance_x_m": theory_distance_x_m,
        "nominal_theory_distance_x_cm": theory_distance_x_m * 100.0,
        "nominal_theory_distance_planar_cm": theory_distance_planar_m * 100.0,
        "collision_factor_1_plus_e": collision_factor,
        "estimated_restitution_e": collision_factor - 1.0,
        "lateral_speed_ratio": lateral_speed_ratio,
        "impact_lateral_offset_m": impact_lateral_offset_m,
        "robot_target_contact_count": robot_target_contact_count,
        "on_table_at_60f": on_table_at_60f,
        "measured_distance_60f_cm": float(metrics["target_displacement_m"]) * 100.0,
        "lookup_usable": lookup_usable,
        "diagnostics_file": str(diagnostics_path),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for ordered_index in order[index:end]:
            ranks[ordered_index] = rank
        index = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else math.nan


def action_regularity(
    action_rows: list[dict[str, Any]], projectile_mass_kg: float
) -> dict[str, Any]:
    rows = sorted(
        (row for row in action_rows if row["lookup_usable"]),
        key=lambda row: row["target_mass_kg"],
    )
    strict_violations = []
    significant_violations = []
    for lower_mass, higher_mass in zip(rows, rows[1:]):
        increase = (
            higher_mass["target_postcollision_vx_mps"]
            - lower_mass["target_postcollision_vx_mps"]
        )
        if increase > 0.0:
            strict_violations.append((lower_mass, higher_mass, increase))
        threshold = max(0.005, 0.03 * abs(lower_mass["target_postcollision_vx_mps"]))
        if increase > threshold:
            significant_violations.append((lower_mass, higher_mass, increase))

    masses = [float(row["target_mass_kg"]) for row in rows]
    speeds = [float(row["target_postcollision_vx_mps"]) for row in rows]
    spearman = (
        correlation(average_ranks(masses), average_ranks(speeds))
        if len(rows) >= 2
        else math.nan
    )
    predictors = [
        float(row["preimpact_projectile_vx_mps"])
        * projectile_mass_kg
        / (projectile_mass_kg + float(row["target_mass_kg"]))
        for row in rows
    ]
    denominator = sum(value * value for value in predictors)
    collision_factor_fit = (
        sum(x * y for x, y in zip(predictors, speeds)) / denominator
        if denominator > 0.0
        else math.nan
    )
    fitted = [collision_factor_fit * value for value in predictors]
    speed_mean = fmean(speeds)
    residual_sum = sum((actual - predicted) ** 2 for actual, predicted in zip(speeds, fitted))
    total_sum = sum((actual - speed_mean) ** 2 for actual in speeds)
    collision_law_r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else math.nan
    rmse = math.sqrt(residual_sum / len(rows))
    factors = [float(row["collision_factor_1_plus_e"]) for row in rows]
    factor_mean = fmean(factors)
    factor_cv = pstdev(factors) / abs(factor_mean) if factor_mean != 0.0 else math.nan

    return {
        "action_id": int(action_rows[0]["action_id"]),
        "A": float(action_rows[0]["A"]),
        "total_count": len(action_rows),
        "usable_count": len(rows),
        "single_contact_count": sum(
            int(row["merged_contact_segment_count"] == 1) for row in action_rows
        ),
        "off_table_count": sum(int(not row["on_table_at_60f"]) for row in action_rows),
        "strict_monotonic_violation_count": len(strict_violations),
        "significant_monotonic_violation_count": len(significant_violations),
        "spearman_mass_vs_post_vx": spearman,
        "collision_factor_fit_1_plus_e": collision_factor_fit,
        "estimated_restitution_fit_e": collision_factor_fit - 1.0,
        "collision_law_r2": collision_law_r2,
        "collision_law_rmse_mps": rmse,
        "collision_factor_cv": factor_cv,
        "mean_preimpact_vx_mps": fmean(
            float(row["preimpact_projectile_vx_mps"]) for row in rows
        ),
        "min_postcollision_vx_mps": min(speeds),
        "max_postcollision_vx_mps": max(speeds),
        "min_theory_distance_cm": min(
            float(row["nominal_theory_distance_x_cm"]) for row in rows
        ),
        "max_theory_distance_cm": max(
            float(row["nominal_theory_distance_x_cm"]) for row in rows
        ),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    action_ids: list[int],
) -> None:
    by_pair = {
        (int(row["action_id"]), float(row["target_mass_kg"])): row for row in rows
    }
    masses = sorted({float(row["target_mass_kg"]) for row in rows})
    matrix_rows: list[dict[str, Any]] = []
    for mass in masses:
        matrix_row: dict[str, Any] = {
            "target_mass_kg": mass,
            "target_mass_g": mass * 1000.0,
        }
        for action_id in action_ids:
            matrix_row[f"action_{action_id}"] = by_pair[(action_id, mass)][value_key]
        matrix_rows.append(matrix_row)
    write_csv(
        path,
        matrix_rows,
        ["target_mass_kg", "target_mass_g"]
        + [f"action_{action_id}" for action_id in action_ids],
    )


def build_uniform_theory_grid(
    selected_stat: dict[str, Any],
    *,
    projectile_mass_kg: float,
    friction_mu: float,
    level_count: int,
) -> list[dict[str, Any]]:
    collision_factor = float(selected_stat["collision_factor_fit_1_plus_e"])
    preimpact_vx = float(selected_stat["mean_preimpact_vx_mps"])
    min_mass = 0.01
    max_mass = 2.0

    def fitted_speed(target_mass_kg: float) -> float:
        return (
            collision_factor
            * preimpact_vx
            * projectile_mass_kg
            / (projectile_mass_kg + target_mass_kg)
        )

    min_distance_m = fitted_speed(max_mass) ** 2 / (2.0 * friction_mu * GRAVITY_MPS2)
    max_distance_m = fitted_speed(min_mass) ** 2 / (2.0 * friction_mu * GRAVITY_MPS2)
    rows: list[dict[str, Any]] = []
    for level in range(level_count):
        fraction = level / (level_count - 1)
        target_distance_m = min_distance_m + fraction * (max_distance_m - min_distance_m)
        target_speed = math.sqrt(2.0 * friction_mu * GRAVITY_MPS2 * target_distance_m)
        target_mass = (
            collision_factor * preimpact_vx * projectile_mass_kg / target_speed
            - projectile_mass_kg
        )
        rows.append(
            {
                "distance_level_short_to_long": level,
                "linear_fraction": fraction,
                "target_nominal_theory_distance_cm": target_distance_m * 100.0,
                "target_postcollision_vx_mps": target_speed,
                "fitted_target_mass_kg": target_mass,
                "fitted_target_mass_g": target_mass * 1000.0,
                "reference_action_id": int(selected_stat["action_id"]),
                "reference_A": float(selected_stat["A"]),
                "collision_factor_fit_1_plus_e": collision_factor,
                "mean_preimpact_vx_mps": preimpact_vx,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    episodes = load_jsonl(args.episodes)
    expected_count = len(config["actions"]) * len(config["target_masses_kg"])
    if len(episodes) != expected_count:
        raise RuntimeError(f"Expected {expected_count} episodes, found {len(episodes)}")

    friction_mu = float(config["friction_mu"])
    projectile_mass_kg = float(config["projectile_mass_kg"])
    rows = [
        extract_episode(
            episode,
            episodes_path=args.episodes,
            friction_mu=friction_mu,
            projectile_mass_kg=projectile_mass_kg,
            impact_lateral_limit_m=float(config["impact_lateral_offset_max_m"]),
        )
        for episode in episodes
    ]
    rows.sort(key=lambda row: (row["action_id"], row["target_mass_kg"]))

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["action_id"])].append(row)
    stats = [action_regularity(grouped[action_id], projectile_mass_kg) for action_id in sorted(grouped)]
    selected = min(
        stats,
        key=lambda stat: (
            -int(stat["usable_count"]),
            int(stat["significant_monotonic_violation_count"]),
            -float(stat["collision_law_r2"]),
            abs(int(stat["action_id"]) - 4),
        ),
    )
    selected_action_id = int(selected["action_id"])
    selected_rows = grouped[selected_action_id]
    uniform_grid = build_uniform_theory_grid(
        selected,
        projectile_mass_kg=projectile_mass_kg,
        friction_mu=friction_mu,
        level_count=len(config["target_masses_kg"]),
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    row_fields = list(rows[0].keys())
    stat_fields = list(stats[0].keys())
    grid_fields = list(uniform_grid[0].keys())
    write_csv(output / "all_postimpact_lookup.csv", rows, row_fields)
    write_csv(output / "action_regularity.csv", stats, stat_fields)
    write_csv(output / "selected_reference_action_lookup.csv", selected_rows, row_fields)
    write_csv(output / "uniform_theoretical_distance_20level_candidate.csv", uniform_grid, grid_fields)
    action_ids = sorted(grouped)
    write_matrix(
        output / "postimpact_vx_matrix_mps.csv",
        rows,
        value_key="target_postcollision_vx_mps",
        action_ids=action_ids,
    )
    write_matrix(
        output / "nominal_theoretical_distance_matrix_cm.csv",
        rows,
        value_key="nominal_theory_distance_x_cm",
        action_ids=action_ids,
    )

    report_lines = [
        "# Post-impact speed lookup",
        "",
        f"- Source episodes: `{args.episodes.resolve()}`",
        f"- Episode count: {len(rows)}",
        f"- Nominal friction coefficient: {friction_mu}",
        f"- Projectile mass: {projectile_mass_kg} kg",
        "- Off-table rollouts are retained.",
        "- Primary speed is the first sampled free-sliding target velocity after the first contact cluster.",
        "- Nominal stopping distance uses `d = vx_post^2 / (2 * mu * g)`.",
        "",
        "## Action regularity",
        "",
        "| action | A | usable | significant violations | Spearman(mass,vx) | collision-law R2 | theory range (cm) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stat in stats:
        report_lines.append(
            f"| {stat['action_id']} | {stat['A']:.2f} | {stat['usable_count']}/{stat['total_count']} "
            f"| {stat['significant_monotonic_violation_count']} "
            f"| {stat['spearman_mass_vs_post_vx']:.4f} "
            f"| {stat['collision_law_r2']:.4f} "
            f"| {stat['min_theory_distance_cm']:.2f}-{stat['max_theory_distance_cm']:.2f} |"
        )
    report_lines.extend(
        [
            "",
            "## Selected reference action",
            "",
            f"Action `{selected_action_id}` with `A={selected['A']:.2f}` was selected by usable coverage, "
            "significant monotonic violations, and collision-law R2.",
            "",
            "The candidate 20-level table spaces nominal theoretical stopping distance linearly and inverts "
            "the fitted one-dimensional collision law to obtain target masses. It is a calibration proposal, "
            "not a generated dataset.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "output": str(output.resolve()),
        "episode_count": len(rows),
        "friction_mu": friction_mu,
        "off_table_retained": sum(int(not row["on_table_at_60f"]) for row in rows),
        "selected_reference_action": selected,
        "actions": stats,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"output={output.resolve()}")
    print(
        f"episodes={len(rows)} off_table_retained={summary['off_table_retained']} "
        f"mu={friction_mu}"
    )
    for stat in stats:
        print(
            f"action={stat['action_id']} A={stat['A']:.2f} usable={stat['usable_count']}/20 "
            f"violations={stat['significant_monotonic_violation_count']} "
            f"rho={stat['spearman_mass_vs_post_vx']:.4f} "
            f"R2={stat['collision_law_r2']:.4f} "
            f"theory={stat['min_theory_distance_cm']:.2f}-{stat['max_theory_distance_cm']:.2f}cm"
        )
    print(
        f"selected_action={selected_action_id} A={selected['A']:.2f} "
        f"collision_factor={selected['collision_factor_fit_1_plus_e']:.5f}"
    )
    print("selected_reference_samples:")
    sample_indices = sorted({0, 3, 7, 11, 15, len(selected_rows) - 1})
    for index in sample_indices:
        row = selected_rows[index]
        print(
            f"  mass={row['target_mass_g']:.1f}g "
            f"v_post={row['target_postcollision_vx_mps']:.5f}m/s "
            f"d_theory={row['nominal_theory_distance_x_cm']:.2f}cm "
            f"contacts={row['merged_contact_segment_count']} "
            f"off_table={not row['on_table_at_60f']}"
        )
    print("uniform_candidate_endpoints:")
    for row in (uniform_grid[0], uniform_grid[len(uniform_grid) // 2], uniform_grid[-1]):
        print(
            f"  level={row['distance_level_short_to_long']} "
            f"mass={row['fitted_target_mass_g']:.2f}g "
            f"v={row['target_postcollision_vx_mps']:.5f}m/s "
            f"d={row['target_nominal_theory_distance_cm']:.2f}cm"
        )


if __name__ == "__main__":
    main()

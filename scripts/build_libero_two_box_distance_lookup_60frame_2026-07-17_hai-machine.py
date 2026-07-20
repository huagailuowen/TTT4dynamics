#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "mass" / "episodes.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "mass" / "distance_lookup_60frame_2026-07-17_hai-machine"
TABLETOP_Z_MIN_M = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raw 60-frame mass/action/distance lookup table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    initial = metrics["initial_target_xyz"]
    final = metrics["final_target_xyz"]
    on_table = float(final[2]) >= TABLETOP_Z_MIN_M
    return {
        "episode_index": int(row["episode_index"]),
        "action_id": int(row["action_id"]),
        "A": float(row["A"]),
        "push_steps": int(row["push_steps"]),
        "target_mass_kg": float(row["target_mass_kg"]),
        "target_mass_g": float(row["target_mass_g"]),
        "projectile_mass_kg": float(row["projectile_mass_kg"]),
        "mass_ratio_target_to_projectile": float(row["target_mass_kg"]) / float(row["projectile_mass_kg"]),
        "preimpact_projectile_vx_mps": float(metrics["preimpact_projectile_vx_mps"]),
        "target_peak_vx_mps": float(metrics["target_peak_vx_mps"]),
        "distance_60f_planar_cm": float(metrics["target_displacement_m"]) * 100.0,
        "distance_60f_forward_cm": (float(final[0]) - float(initial[0])) * 100.0,
        "lateral_60f_cm": (float(final[1]) - float(initial[1])) * 100.0,
        "initial_target_z_m": float(initial[2]),
        "final_target_z_m": float(final[2]),
        "on_table_at_60f": on_table,
        "lookup_valid": on_table,
        "invalid_reason": "" if on_table else "OFF_TABLE",
    }


def monotonic_violations(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    valid = sorted((row for row in rows if row["lookup_valid"]), key=lambda row: row["target_mass_kg"])
    violations: list[dict[str, float]] = []
    for left, right in zip(valid, valid[1:]):
        if right["distance_60f_planar_cm"] > left["distance_60f_planar_cm"]:
            violations.append(
                {
                    "from_mass_g": left["target_mass_g"],
                    "to_mass_g": right["target_mass_g"],
                    "from_distance_cm": left["distance_60f_planar_cm"],
                    "to_distance_cm": right["distance_60f_planar_cm"],
                    "increase_cm": right["distance_60f_planar_cm"] - left["distance_60f_planar_cm"],
                }
            )
    return violations


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def matrix_cell(row: dict[str, Any]) -> str:
    value = f"{row['distance_60f_planar_cm']:.2f}"
    return value if row["lookup_valid"] else f"OFF_TABLE({value})"


def build_report(
    *,
    rows: list[dict[str, Any]],
    action_ids: list[int],
    masses_g: list[float],
    matrix: dict[tuple[float, int], dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    lines = [
        "# LIBERO two-box 60-frame distance lookup",
        "",
        "Generated: 2026-07-17 on hai-machine.",
        "",
        "- Source: 180 real LIBERO/MuJoCo rollouts from the completed 9-action x 20-mass dataset.",
        "- Distance definition: target block planar displacement at frame 60, in centimeters.",
        f"- Validity: final target z >= {TABLETOP_Z_MIN_M:.2f} m. `OFF_TABLE(...)` retains the raw measurement but must not be used for inverse lookup.",
        "- No smoothing, interpolation, isotonic fitting, or synthetic values are applied in this table.",
        "",
        "## Action summary",
        "",
        "| Action | Mean impact vx (m/s) | Valid / 20 | Valid mass range (g) | Valid distance range (cm) | Monotonic violations |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['action_id']} | {summary['mean_preimpact_vx_mps']:.3f} | "
            f"{summary['valid_count']} / 20 | {summary['valid_mass_min_g']:.0f}-{summary['valid_mass_max_g']:.0f} | "
            f"{summary['valid_distance_min_cm']:.2f}-{summary['valid_distance_max_cm']:.2f} | "
            f"{summary['monotonic_violation_count']} |"
        )
    lines.extend(
        [
            "",
            "## Raw distance matrix (cm at frame 60)",
            "",
            "| Mass (g) | " + " | ".join(f"A{action_id}" for action_id in action_ids) + " |",
            "|---:|" + "---:|" * len(action_ids),
        ]
    )
    for mass_g in masses_g:
        lines.append(
            f"| {mass_g:.0f} | "
            + " | ".join(matrix_cell(matrix[(mass_g, action_id)]) for action_id in action_ids)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The raw map is not globally invertible. Several low-mass/high-action points leave the table, and valid curves still contain local non-monotonicity. The next dataset should therefore be designed from a monotone fitted valid branch or from a denser calibration sweep, then inverted at evenly spaced target distances. This report intentionally stops before that decision.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_rows = load_rows(input_path)
    rows = sorted((enrich(row) for row in source_rows), key=lambda row: (row["target_mass_g"], row["action_id"]))
    action_ids = sorted({int(row["action_id"]) for row in rows})
    masses_g = sorted({float(row["target_mass_g"]) for row in rows})
    if len(rows) != 180 or len(action_ids) != 9 or len(masses_g) != 20:
        raise RuntimeError(
            f"Expected 180 rows, 9 actions, and 20 masses; got {len(rows)}, {len(action_ids)}, {len(masses_g)}"
        )

    matrix = {(row["target_mass_g"], row["action_id"]): row for row in rows}
    by_action: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[int(row["action_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    action_payloads: list[dict[str, Any]] = []
    for action_id in action_ids:
        action_rows = sorted(by_action[action_id], key=lambda row: row["target_mass_g"])
        valid = [row for row in action_rows if row["lookup_valid"]]
        violations = monotonic_violations(action_rows)
        summary = {
            "action_id": action_id,
            "A": action_rows[0]["A"],
            "push_steps": action_rows[0]["push_steps"],
            "mean_preimpact_vx_mps": sum(row["preimpact_projectile_vx_mps"] for row in action_rows) / len(action_rows),
            "valid_count": len(valid),
            "off_table_count": len(action_rows) - len(valid),
            "valid_mass_min_g": min(row["target_mass_g"] for row in valid),
            "valid_mass_max_g": max(row["target_mass_g"] for row in valid),
            "valid_distance_min_cm": min(row["distance_60f_planar_cm"] for row in valid),
            "valid_distance_max_cm": max(row["distance_60f_planar_cm"] for row in valid),
            "monotonic_violation_count": len(violations),
            "monotonic_violations": violations,
        }
        summaries.append(summary)
        action_payloads.append({"summary": summary, "measurements": action_rows})

    fields = list(rows[0].keys())
    write_csv(output / "raw_lookup_long.csv", rows, fields)
    write_csv(output / "valid_planar_lookup_long.csv", [row for row in rows if row["lookup_valid"]], fields)

    matrix_fields = ["target_mass_g"] + [f"action_{action_id:02d}_distance_60f_cm" for action_id in action_ids]
    matrix_rows: list[dict[str, Any]] = []
    for mass_g in masses_g:
        matrix_row: dict[str, Any] = {"target_mass_g": mass_g}
        for action_id in action_ids:
            matrix_row[f"action_{action_id:02d}_distance_60f_cm"] = matrix_cell(matrix[(mass_g, action_id)])
        matrix_rows.append(matrix_row)
    write_csv(output / "raw_distance_matrix_cm.csv", matrix_rows, matrix_fields)

    payload = {
        "source": str(input_path),
        "horizon_frames": 60,
        "fps": 20,
        "distance_definition": "target planar displacement from initial pose at frame 60",
        "tabletop_z_min_m": TABLETOP_Z_MIN_M,
        "raw_measurement_count": len(rows),
        "valid_measurement_count": sum(1 for row in rows if row["lookup_valid"]),
        "off_table_measurement_count": sum(1 for row in rows if not row["lookup_valid"]),
        "actions": action_payloads,
    }
    (output / "distance_lookup.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "distance_lookup_report.md").write_text(
        build_report(rows=rows, action_ids=action_ids, masses_g=masses_g, matrix=matrix, summaries=summaries),
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"raw={len(rows)} valid={payload['valid_measurement_count']} off_table={payload['off_table_measurement_count']}")
    for summary in summaries:
        print(
            f"action={summary['action_id']} valid={summary['valid_count']}/20 "
            f"distance={summary['valid_distance_min_cm']:.2f}-{summary['valid_distance_max_cm']:.2f}cm "
            f"violations={summary['monotonic_violation_count']}"
        )


if __name__ == "__main__":
    main()

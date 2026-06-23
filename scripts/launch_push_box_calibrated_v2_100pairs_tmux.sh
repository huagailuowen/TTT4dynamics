#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation_Project/TTTdynamics/repos}
PROJECT_ROOT=${PROJECT_ROOT:-$REPO/TTT4dynamics}
FASTWAM_ROOT=${FASTWAM_ROOT:-$REPO/FastWAM}
PY=${PY:-$FASTWAM_ROOT/.venv/bin/python}

PREFIX=${PREFIX:-libero_push_box_calibrated_v2_100pairs}
TMP_ROOT=${TMP_ROOT:-$PROJECT_ROOT/tmp/$PREFIX}
BUILD_ROOT=${BUILD_ROOT:-$TMP_ROOT/build}
BDDL_ROOT=${BDDL_ROOT:-$TMP_ROOT/bddl}
LOG_DIR=${LOG_DIR:-$TMP_ROOT/logs}
FINAL_PARENT=${FINAL_PARENT:-$FASTWAM_ROOT/data}
FINAL_ROOT=${FINAL_ROOT:-$FINAL_PARENT/$PREFIX}
CALIBRATION_TABLE=${CALIBRATION_TABLE:-$PROJECT_ROOT/configs/libero_push_box_push_calibration_table.json}

FRICTIONS=${FRICTIONS:-"0.005 0.02 0.05 0.1 0.2"}
DISPLACEMENT_BIN_EDGES=${DISPLACEMENT_BIN_EDGES:-"0.20 0.35"}
DISPLACEMENT_BIN_QUOTAS=${DISPLACEMENT_BIN_QUOTAS:-"14 20 16"}
BALANCE_DIMENSIONS=${BALANCE_DIMENSIONS:-"split displacement_bin"}
MAX_PAIRS_PER_FRICTION=${MAX_PAIRS_PER_FRICTION:-10}
INIT_XYS=${INIT_XYS:-"-0.255,-0.055 -0.245,-0.035 -0.235,-0.015 -0.225,0.005 -0.215,0.025"}
MAX_TRIALS=${MAX_TRIALS:-1200}
MAX_STEPS=${MAX_STEPS:-360}
CAMERA_RESOLUTION=${CAMERA_RESOLUTION:-224}
PROBE_RESOLUTION=${PROBE_RESOLUTION:-24}
VIDEO_CRF=${VIDEO_CRF:-18}
JPEG_QUALITY=${JPEG_QUALITY:-98}
REPORT_INTERVAL=${REPORT_INTERVAL:-120}
SEED_BASE=${SEED_BASE:-2026062300}

cd "$PROJECT_ROOT"
mkdir -p "$BUILD_ROOT" "$BDDL_ROOT" "$LOG_DIR" "$FINAL_PARENT"
rm -rf "$BUILD_ROOT" "$BDDL_ROOT" "$LOG_DIR"
mkdir -p "$BUILD_ROOT" "$BDDL_ROOT" "$LOG_DIR"

final_path() {
  local domain="$1"
  local split="$2"
  local domain_name
  if [[ "$domain" == "observation" ]]; then
    domain_name="hidden"
  else
    domain_name="visible"
  fi
  echo "$FINAL_ROOT/${PREFIX}_${domain_name}_${split}_lerobot"
}

tmp_prefix_for_split() {
  local split="$1"
  echo "$BUILD_ROOT/${PREFIX}_${split}"
}

cleanup_final() {
  rm -rf "$FINAL_ROOT"
  mkdir -p "$FINAL_ROOT"
}

run_split() {
  local split="$1"
  local seed="$2"
  local angles
  if [[ "$split" == "straight" ]]; then
    angles="--straight-angles 0"
  else
    angles="--angled-angles -30 -20 -10 10 20 30"
  fi

  local prefix
  prefix=$(tmp_prefix_for_split "$split")
  local log="$LOG_DIR/${split}.log"

  # shellcheck disable=SC2086
  env \
    PYTHONPATH="$REPO/LIBERO:$PROJECT_ROOT:$FASTWAM_ROOT/src" \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$PY" scripts/collect_libero_push_box_rollout_target_lerobot_dataset.py \
      --output-prefix "$prefix" \
      --bddl-dir "$BDDL_ROOT/$split" \
      --splits "$split" \
      --frictions $FRICTIONS \
      $angles \
      $(for xy in $INIT_XYS; do printf -- '--init-xys=%s ' "$xy"; done) \
      --calibration-table "$CALIBRATION_TABLE" \
      --balance-dimensions $BALANCE_DIMENSIONS \
      --displacement-bin-edges $DISPLACEMENT_BIN_EDGES \
      --displacement-bin-quotas $DISPLACEMENT_BIN_QUOTAS \
      --max-pairs-per-friction "$MAX_PAIRS_PER_FRICTION" \
      --min-displacement 0.10 \
      --max-displacement 0.56 \
      --x-bounds -0.24 0.40 \
      --y-bounds -0.28 0.28 \
      --max-final-speed 0.040 \
      --max-eef-step 0.065 \
      --pairs-per-bucket 1 \
      --max-trials-per-bucket "$MAX_TRIALS" \
      --max-steps "$MAX_STEPS" \
      --camera-resolution "$CAMERA_RESOLUTION" \
      --probe-resolution "$PROBE_RESOLUTION" \
      --video-crf "$VIDEO_CRF" \
      --jpeg-quality "$JPEG_QUALITY" \
      --seed "$seed" \
      --repo-id-prefix "${PREFIX}_${split}" \
      --overwrite \
      > "$log" 2>&1 &
  echo "$! $split $log"
}

report() {
  {
    echo "=== $(date -Iseconds) ==="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    ps -eo pcpu,cmd | awk '/collect_libero_push_box_rollout_target_lerobot_dataset.py/ && !/awk/ {s+=$1; n++} END {print "collect_workers=" (n+0), "cpu_sum=" (s+0)}'
    PREFIX="$PREFIX" BUILD_ROOT="$BUILD_ROOT" "$PY" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

prefix = os.environ["PREFIX"]
build = Path(os.environ["BUILD_ROOT"])
rows = []
missing = Counter()
for split in ("straight", "angled"):
    manifest = build / f"{prefix}_{split}_manifest.json"
    if not manifest.exists():
        continue
    payload = json.loads(manifest.read_text())
    rows.extend(payload.get("pairs", []))
    for key, value in payload.get("missing_buckets", {}).items():
        missing[key] += int(value)
vals = [r["metrics"]["displacement_m"] * 100.0 for r in rows]
print(
    "pairs", len(rows),
    "mean_cm", round(sum(vals) / len(vals), 1) if vals else None,
    "minmax_cm", (round(min(vals), 1), round(max(vals), 1)) if vals else None,
)
print("split", dict(Counter(r["candidate"]["split"] for r in rows)))
print("friction", dict(Counter(str(r["candidate"]["friction_mu"]) for r in rows)))
print("disp_bin", dict(Counter(r["metrics"]["displacement_bin"] for r in rows)))
print("missing_buckets", sum(missing.values()))
PY
  } | tee -a "$LOG_DIR/supervisor.log"
}

move_split_outputs() {
  local split="$1"
  local tmp_prefix
  tmp_prefix=$(tmp_prefix_for_split "$split")
  mv "${tmp_prefix}_hidden_${split}_lerobot" "$(final_path observation "$split")"
  mv "${tmp_prefix}_visible_${split}_lerobot" "$(final_path task "$split")"
}

write_final_manifest() {
  PREFIX="$PREFIX" BUILD_ROOT="$BUILD_ROOT" FINAL_ROOT="$FINAL_ROOT" "$PY" - <<'PY'
import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path

prefix = os.environ["PREFIX"]
build = Path(os.environ["BUILD_ROOT"])
final = Path(os.environ["FINAL_ROOT"])
rows = []
missing = {}
sources = {}
for split in ("straight", "angled"):
    manifest = build / f"{prefix}_{split}_manifest.json"
    if not manifest.exists():
        continue
    payload = json.loads(manifest.read_text())
    sources[split] = str(manifest)
    rows.extend(payload.get("pairs", []))
    for key, value in payload.get("missing_buckets", {}).items():
        missing[f"{split}:{key}"] = int(value)
vals = [r["metrics"]["displacement_m"] * 100.0 for r in rows]
payload = {
    "created_at": dt.datetime.now().isoformat(),
    "dataset_type": "libero_push_box_calibrated_v2_100pairs_collection",
    "source_manifests": sources,
    "subset_roots": {
        "observation_straight": str(final / f"{prefix}_hidden_straight_lerobot"),
        "task_straight": str(final / f"{prefix}_visible_straight_lerobot"),
        "observation_angled": str(final / f"{prefix}_hidden_angled_lerobot"),
        "task_angled": str(final / f"{prefix}_visible_angled_lerobot"),
    },
    "pairs": rows,
    "missing_buckets": missing,
    "summary": {
        "pairs": len(rows),
        "mean_displacement_cm": round(sum(vals) / len(vals), 3) if vals else None,
        "min_displacement_cm": round(min(vals), 3) if vals else None,
        "max_displacement_cm": round(max(vals), 3) if vals else None,
        "split": dict(Counter(r["candidate"]["split"] for r in rows)),
        "friction": dict(Counter(str(r["candidate"]["friction_mu"]) for r in rows)),
        "displacement_bin": dict(Counter(r["metrics"]["displacement_bin"] for r in rows)),
        "push_mode": dict(Counter(r["candidate"].get("push_mode", "position") for r in rows)),
    },
}
(final / f"{prefix}_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

for root in payload["subset_roots"].values():
    meta_path = Path(root) / "push_box_generation_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["output_root"] = str(root)
        meta["final_collection_manifest"] = str(final / f"{prefix}_manifest.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
PY
}

echo "launching calibrated v2 push-box dataset PREFIX=$PREFIX TMP_ROOT=$TMP_ROOT FINAL_ROOT=$FINAL_ROOT" | tee "$LOG_DIR/supervisor.log"
cleanup_final
run_split straight "$((SEED_BASE + 0))" >> "$LOG_DIR/pids.txt"
run_split angled "$((SEED_BASE + 1000))" >> "$LOG_DIR/pids.txt"
cat "$LOG_DIR/pids.txt" | tee -a "$LOG_DIR/supervisor.log"

report
while [ "$(jobs -pr | wc -l)" -gt 0 ]; do
  sleep "$REPORT_INTERVAL"
  report
done

wait
report
cleanup_final
move_split_outputs straight
move_split_outputs angled
write_final_manifest
rm -rf "$TMP_ROOT"

echo "done: $FINAL_ROOT/${PREFIX}_manifest.json"
for split in straight angled; do
  echo "hidden_$split: $(final_path observation "$split")"
  echo "visible_$split: $(final_path task "$split")"
done

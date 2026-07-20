#!/usr/bin/env bash
set -euo pipefail

TTT4DYNAMICS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${TTT4DYNAMICS_ROOT}/../.." && pwd)"
VENV_ROOT="${TTT4DYNAMICS_ROOT}/.venv"
LIBERO_PLUS_ROOT="${WORKSPACE_ROOT}/repos/LIBERO-plus"
LIBERO_PLUS_CONFIG="${TTT4DYNAMICS_ROOT}/configs/libero_plus_runtime_2026-07-18_hai-machine"
MAGICK_PREFIX="${VENV_ROOT}/lib/libmagickwand-hai-machine"
MAGICK_LIB="${MAGICK_PREFIX}/usr/lib/x86_64-linux-gnu"

if [[ $# -eq 0 ]]; then
    printf 'usage: %s <command> [args...]\n' "$0" >&2
    exit 2
fi

source "${VENV_ROOT}/bin/activate"

export PYTHONPATH="${LIBERO_PLUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG}"
export MAGICK_HOME="${MAGICK_PREFIX}/usr"
export MAGICK_CONFIGURE_PATH="${MAGICK_PREFIX}/etc/ImageMagick-6"
export LD_LIBRARY_PATH="${MAGICK_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${MAGICK_LIB}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

MAGICK_MODULE_ROOT="$(find "${MAGICK_LIB}" -maxdepth 2 -type d -path '*/ImageMagick-*/modules-Q16' -print -quit)"
if [[ -n "${MAGICK_MODULE_ROOT}" ]]; then
    export MAGICK_CODER_MODULE_PATH="${MAGICK_MODULE_ROOT}/coders"
    export MAGICK_FILTER_MODULE_PATH="${MAGICK_MODULE_ROOT}/filters"
fi

exec "$@"

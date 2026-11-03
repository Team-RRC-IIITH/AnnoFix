#!/usr/bin/env bash
# Open AnnoFix on a folder of pipeline samples (CVAT COCO, one folder per sample, any depth).
#
#   scripts/annofix.sh                      the usual dataset -> DataSet/Annotations_fixed
#   scripts/annofix.sh SAMPLES [OUT]        another folder (OUT defaults to SAMPLES_fixed next to it)
#   scripts/annofix.sh -- --device cpu      extra options go to AnnoFix (after --)
#
# New scenes copied into SAMPLES are picked up on every start; fixes already made in OUT are kept.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$HOME/Documents/Honours/SamsungProject/DataSet"
DEFAULT_SRC="$DATASET/Annotations-20261030T173246Z-1-001/Annotations"
DEFAULT_OUT="$DATASET/Annotations_fixed"

positional=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    positional+=("$1")
    shift
done
[[ "${1:-}" == "--" ]] && shift   # what is left goes to AnnoFix as it is

if [[ ${#positional[@]} -gt 2 ]]; then
    echo "usage: $0 [SAMPLES [OUT]] [-- annofix options]" >&2
    exit 2
fi
SRC="${positional[0]:-$DEFAULT_SRC}"
if [[ ${#positional[@]} -ge 2 ]]; then
    OUT="${positional[1]}"
elif [[ ${#positional[@]} -eq 1 ]]; then
    OUT="${SRC%/}_fixed"
else
    OUT="$DEFAULT_OUT"
fi

if [[ ! -d "$SRC" ]]; then
    echo "annofix: no such folder: $SRC" >&2
    exit 1
fi
# Absolute now: AnnoFix runs from the repo (cd below), where a relative path would mean another folder.
SRC="$(realpath -- "$SRC")"
OUT="$(realpath -m -- "$OUT")"
if ! command -v uv >/dev/null; then
    echo "annofix: uv is not installed (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

# CVAT's SAM holds most of the GPU while it runs; AnnoFix then loads SAM on the CPU (slower).
if command -v docker >/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'sam-vit-h'; then
    echo "note: CVAT's SAM is running and holds the GPU, so AnnoFix's SAM will use the CPU."
    echo "      To free the GPU: docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h"
fi

branch="$(git -C "$REPO" branch --show-current 2>/dev/null || true)"
echo "AnnoFix (branch: ${branch:-?})"
echo "  samples: $SRC"
echo "  fixes:   $OUT"

cd "$REPO"
exec uv run python -m annofix.cvat "$SRC" --out "$OUT" "$@"

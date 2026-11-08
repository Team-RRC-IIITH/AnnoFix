#!/usr/bin/env bash
# AnnoFix in Docker, used in the browser at http://localhost:6080. Needs only Docker.
#
#   scripts/annofix-docker.sh SAMPLES [OUT]                  fixes go to OUT (default: SAMPLES_fixed)
#   scripts/annofix-docker.sh --gpu SAMPLES [OUT]            the GPU image (NVIDIA + Container Toolkit)
#   scripts/annofix-docker.sh SAMPLES -- --annotator NAME    options after -- go to AnnoFix
#
# The first run builds the image (several minutes). SAMPLES is mounted read-only; Ctrl+C saves
# and closes AnnoFix.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service=annofix
profile=()
positional=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --gpu) service=annofix-gpu; profile=(--profile gpu) ;;
        *) positional+=("$1") ;;
    esac
    shift
done
[[ "${1:-}" == "--" ]] && shift   # what is left goes to AnnoFix

if [[ ${#positional[@]} -lt 1 || ${#positional[@]} -gt 2 ]]; then
    echo "usage: $0 [--gpu] SAMPLES [OUT] [-- annofix options]" >&2
    echo "  SAMPLES: folder of pipeline samples (<sample>/annotations/instances_default.json)" >&2
    echo "  OUT:     where the fixed samples go (default: SAMPLES_fixed)" >&2
    exit 2
fi
if [[ ! -d "${positional[0]}" ]]; then
    echo "annofix: no such folder: ${positional[0]}" >&2
    exit 1
fi
SAMPLES="$(realpath -- "${positional[0]}")"
OUT="$(realpath -m -- "${positional[1]:-${SAMPLES%/}_fixed}")"
mkdir -p "$OUT"   # created now, by you: if Docker made it, it would belong to root

export SAMPLES OUT HOST_UID="$(id -u)" HOST_GID="$(id -g)" ANNOFIX_ARGS="$*"
echo "AnnoFix in Docker ($service)"
echo "  samples: $SAMPLES (read-only)"
echo "  fixes:   $OUT"
echo "  open http://localhost:${PORT:-6080} once it says it is starting; Ctrl+C saves and stops"
exec docker compose -f "$REPO/compose.yaml" "${profile[@]}" up --build "$service"

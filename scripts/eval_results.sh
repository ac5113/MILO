#!/usr/bin/env bash
# Run milo/eval/eval_results.py (CD_h / CD_o / CD_comb metrics) from the repo
# root — see that file's docstring for metrics, inputs and flags.
#
#   bash scripts/eval_results.sh --data_root demo [--save_mesh]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILO="$(cd "$HERE/.." && pwd)"
cd "$MILO"

python milo/eval/eval_results.py "$@"

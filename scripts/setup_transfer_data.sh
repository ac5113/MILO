#!/usr/bin/env bash
# Fetch the one gated asset milo/eval/convert_intercap_gt_human.py needs:
# the SMPL-X -> SMPL+H deformation-transfer setup (smplx2smplh_deftrafo_setup.pkl),
# placed at _DATA/transfer_data/. (The gendered SMPL-H used for the fit is MILO's
# own _DATA/body_models/smplh/<gender>/model.npz from download_models.sh — no
# separate body model is needed here.)
#
# smplx2smplh_deftrafo_setup.pkl is the SMPL-X "Model correspondences" download —
# registration-gated under the MPI license and NOT in the smplx git repo (the repo
# only holds the transfer_model code/configs). See
# https://github.com/vchoutas/smplx/tree/main/transfer_model#data — register at
# https://smpl-x.is.tue.mpg.de, download+extract the correspondences, then:
#
#   # (a) point at an already-extracted transfer_data dir
#   bash scripts/setup_transfer_data.sh --src /path/to/transfer_data
#
#   # (b) no --src: clone the smplx repo and read the setup from its transfer_data/
#   #     (extract your downloaded correspondences zip into it first)
#   bash scripts/setup_transfer_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/_DATA/transfer_data"
SETUP="smplx2smplh_deftrafo_setup.pkl"
SRC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

TMP=""
cleanup() { [ -n "$TMP" ] && rm -rf "$TMP"; }
trap cleanup EXIT

# No source given: clone the repo (official step 1) and read its transfer_data/,
# which you must have populated with the gated correspondences download.
if [ -z "$SRC" ]; then
  TMP="$(mktemp -d)"
  echo "[setup_transfer_data] cloning vchoutas/smplx ..."
  git clone --depth 1 https://github.com/vchoutas/smplx.git "$TMP/smplx"
  SRC="$TMP/smplx/transfer_data"
fi

if [ ! -f "$SRC/$SETUP" ]; then
  echo "[setup_transfer_data] missing $SRC/$SETUP" >&2
  echo "This is the registration-gated SMPL-X 'Model correspondences' download" >&2
  echo "(not in the git repo). Register at https://smpl-x.is.tue.mpg.de, extract it," >&2
  echo "and re-run with --src <extracted transfer_data dir>." >&2
  exit 1
fi

mkdir -p "$DEST"
cp "$SRC/$SETUP" "$DEST/"
echo "[setup_transfer_data] placed $DEST/$SETUP (temp clone + rest discarded)."

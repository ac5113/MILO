#!/usr/bin/env bash
# ===========================================================================
# MILO — model-weight downloader
# ===========================================================================
# Fetches the weights that are NOT auto-downloaded at runtime, into _DATA/ and
# the Grounded-SAM-2 checkpoint dirs. Run once after install_milo.sh.
#
# Auto-downloaded on first pipeline run (NOT handled here): HMR2.0/4D-Humans,
# HaMeR weights, the ViTDet person detector. Hunyuan3D-2 IS pre-fetched here
# (step 4) into _DATA/h3d_checkpoints.
#
# See docs/DATA.md for the full _DATA/ layout and manual fallbacks.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILO="$(cd "$HERE/.." && pwd)"
cd "$MILO"

GSAM=third-party/Grounded-SAM-2

# --- 1. Grounded-SAM-2: SAM2 + GroundingDINO checkpoints -------------------
if [ ! -e "$GSAM/checkpoints/sam2.1_hiera_large.pt" ]; then
  echo "[download_models] SAM2 checkpoints..."
  ( cd "$GSAM/checkpoints" && bash download_ckpts.sh )
else
  echo "[download_models] SAM2 checkpoints already present — skipping."
fi
if [ ! -e "$GSAM/gdino_checkpoints/groundingdino_swint_ogc.pth" ]; then
  echo "[download_models] GroundingDINO checkpoints..."
  ( cd "$GSAM/gdino_checkpoints" && bash download_ckpts.sh )
else
  echo "[download_models] GroundingDINO checkpoints already present — skipping."
fi

# --- 2. HaMeR demo data (hand model + HaMeR checkpoint) --------------------
# Also auto-downloads at runtime; fetching now avoids a ~5.7 GB stall mid-run.
if [ ! -f _DATA/hamer_ckpts/checkpoints/hamer.ckpt ]; then
  echo "[download_models] HaMeR demo data (~5.7 GB)..."
  wget -c https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz -P _DATA/
  tar --warning=no-unknown-keyword --exclude=".*" -xf _DATA/hamer_demo_data.tar.gz -C . || echo "  (HaMeR extract failed — see docs/DATA.md)"
else
  echo "[download_models] HaMeR data already present — skipping."
fi

# --- 3. Registration-gated body models (optional, prompted) ----------------
# SMPL-H / VPoser v1.0 / SMPL-X / MANO live behind the MPI download server and
# need a free account at the project pages (see docs/DATA.md). This automates the
# authenticated download (à la ICON); skip it to place them by hand.
read -rp "[download_models] Download gated body models (SMPL-H / VPoser / SMPL-X / MANO) now? [y/N] " ans || ans=N
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  read -rp  "  MPI account e-mail (is.tue.mpg.de): " MPI_USER
  read -rsp "  MPI account password: " MPI_PASS; echo
  BM=_DATA/body_models
  MANO_DIR=_DATA/data/mano
  mkdir -p "$BM/smplh" "$BM/smplx" "$MANO_DIR"

  # _fetch <domain> <sfile> <out-file> : authenticated download; returns wget status.
  _fetch() {
    echo "  -> $2"
    wget --post-data "username=${MPI_USER}&password=${MPI_PASS}" --no-check-certificate -c \
      "https://download.is.tue.mpg.de/download.php?domain=$1&sfile=$2&resume=1" -O "$3"
  }
  # NOTE: if any of these 404, check the exact file name on your account's
  # Downloads page and update the sfile below (see docs/DATA.md).
  if _fetch mano  smplh.tar.xz          "$BM/smplh.tar.xz";          then tar -xf "$BM/smplh.tar.xz" -C "$BM/smplh" || echo "     (SMPL-H extract failed — see docs/DATA.md)"; else echo "     (SMPL-H download failed — see docs/DATA.md)"; fi
  if _fetch smplx vposer_v1_0.zip       "$BM/vposer_v1_0.zip";       then unzip -o "$BM/vposer_v1_0.zip" -d "$BM" || echo "     (VPoser extract failed — see docs/DATA.md)"; else echo "     (VPoser download failed — see docs/DATA.md)"; fi
  # SMPL-X: keep only SMPLX_NEUTRAL.npz (the fit reads it for hand PCA). -j flattens the in-zip models/smplx/ path.
  if _fetch smplx models_smplx_v1_1.zip "$BM/models_smplx_v1_1.zip"; then unzip -j -o "$BM/models_smplx_v1_1.zip" '*SMPLX_NEUTRAL.npz' -d "$BM/smplx" || echo "     (SMPL-X extract failed — see docs/DATA.md)"; else echo "     (SMPL-X download failed — see docs/DATA.md)"; fi
  # MANO: keep only MANO_RIGHT.pkl (HaMeR right-hand model; left hands are mirrored). -j flattens the in-zip path.
  if _fetch mano  mano_v1_2.zip         "$MANO_DIR/mano_v1_2.zip";    then unzip -j -o "$MANO_DIR/mano_v1_2.zip" '*MANO_RIGHT.pkl' -d "$MANO_DIR" || echo "     (MANO extract failed — see docs/DATA.md)"; else echo "     (MANO download failed — see docs/DATA.md)"; fi
  echo "[download_models] Body models fetched (verify layout against docs/DATA.md)."
else
  echo "[download_models] Skipped gated body models — see docs/DATA.md to add them."
fi

# --- 4. Hunyuan3D-2 weights (shape + texture) ------------------------------
# Pre-fetch into _DATA/h3d_checkpoints so the run_lrm step loads them locally
# instead of downloading mid-run. hy3dgen resolves these via HY3DGEN_MODELS,
# which the pipeline points at _DATA/h3d_checkpoints (milo/pipeline/config.py),
# so the on-disk layout must be <HY3DGEN_MODELS>/<repo_id>/<subfolder>.
# Needs the active `milo` env (huggingface_hub). Only the three subfolders the
# pipeline uses are fetched (shape dit + texture delight/paint).
H3D=_DATA/h3d_checkpoints/tencent/Hunyuan3D-2
if [ ! -f "$H3D/hunyuan3d-dit-v2-0/model.fp16.safetensors" ]; then
  echo "[download_models] Hunyuan3D-2 weights (shape + texture)..."
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="tencent/Hunyuan3D-2",
    local_dir="_DATA/h3d_checkpoints/tencent/Hunyuan3D-2",
    allow_patterns=[
        # shape: only the fp16 safetensors the pipeline loads (skip the 4 other
        # redundant ~4.6 GB variants the repo ships for this subfolder).
        "hunyuan3d-dit-v2-0/config.yaml",
        "hunyuan3d-dit-v2-0/model.fp16.safetensors",
        "hunyuan3d-delight-v2-0/*",      # texture: light/shadow removal
        "hunyuan3d-paint-v2-0-turbo/*",  # texture: multi-view diffusion
    ],
)
PY
else
  echo "[download_models] Hunyuan3D-2 weights already present — skipping."
fi

# --- 4b. SAM 3D Objects weights (optional, prompted — only for --lrm sam3d) --
# ~13 GB, and unused by the default --lrm hy3d backend, so this one asks first
# (answer y only if you need `--lrm sam3d`; a non-interactive run skips it).
# GATED on Hugging Face: request access at
# https://huggingface.co/facebook/sam-3d-objects and `huggingface-cli login`
# first, or this download fails. The HF repo nests everything under a
# checkpoints/ folder; the run_lrm step reads
# _DATA/sam3d_checkpoints/hf/pipeline.yaml (milo/pipeline/config.py).
SAM3D_CKPT=_DATA/sam3d_checkpoints
if [ -f "$SAM3D_CKPT/hf/pipeline.yaml" ]; then
  echo "[download_models] SAM 3D Objects weights already present — skipping."
else
  read -rp "[download_models] Download SAM 3D Objects weights (~13 GB, only for --lrm sam3d)? [y/N] " s3d || s3d=N
  if [[ "${s3d:-N}" =~ ^[Yy]$ ]]; then
    echo "[download_models] SAM 3D Objects weights (gated)..."
    if python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/sam-3d-objects",
    local_dir="_DATA/sam3d_checkpoints/_download",
)
PY
    then
      mv "$SAM3D_CKPT/_download/checkpoints" "$SAM3D_CKPT/hf"
      rm -rf "$SAM3D_CKPT/_download"
      echo "[download_models] SAM 3D Objects weights → $SAM3D_CKPT/hf"
    else
      echo "  (SAM 3D Objects download failed — gated repo; request access on HF and"
      echo "   huggingface-cli login, or skip if you only use the default --lrm hy3d.)"
    fi
  else
    echo "[download_models] Skipped SAM 3D Objects weights — see docs/DATA.md to add them."
  fi
fi

# --- 5. GeoAware-SC dinov2 pin (optional — only for the `correspond` step) ---
# GeoAware-SC loads dinov2 via an UNPINNED torch.hub.load, which now resolves to
# a py3.10-only main (PEP 604 `float | None`) that crashes the documented py3.9
# geo-aware env. Pre-populate the hub cache (the dir the unpinned `main` load
# looks for) with a py3.9-compatible commit so the load uses it instead of
# re-fetching the broken main. Hub dir mirrors milo/pipeline/steps/_geoaware
# (TORCH_HOME = ${MILO_GEO_CACHE:-_cache/geo}). Idempotent: only (re)provisions
# when the cache is missing or still carries the py3.10 syntax.
DINOV2_COMMIT=81b2b6419385a321287de91e00282ef7cbd26f94   # 2023-08-31, last py3.9-safe (pre #528)
DINOV2_DIR="${MILO_GEO_CACHE:-$MILO/_cache/geo}/hub/facebookresearch_dinov2_main"
if [ ! -f "$DINOV2_DIR/dinov2/layers/attention.py" ] || grep -q "float | None" "$DINOV2_DIR/dinov2/layers/attention.py"; then
  echo "[download_models] Pinning dinov2 @ ${DINOV2_COMMIT:0:7} for the geo-aware (py3.9) correspond step..."
  _tmp="$(mktemp -d)"
  if wget -qO- "https://github.com/facebookresearch/dinov2/archive/$DINOV2_COMMIT.tar.gz" | tar -xz -C "$_tmp"; then
    rm -rf "$DINOV2_DIR"; mkdir -p "$(dirname "$DINOV2_DIR")"
    mv "$_tmp"/dinov2-* "$DINOV2_DIR"
    echo "[download_models] dinov2 pinned → $DINOV2_DIR"
  else
    echo "  (dinov2 pin download failed — correspond will break until fixed; see docs/INSTALL.md)"
  fi
  rm -rf "$_tmp"
else
  echo "[download_models] dinov2 cache already py3.9-compatible — skipping."
fi

echo
echo "[download_models] Done."
echo "  Auto-downloaded on first pipeline run: HMR2.0, HaMeR, ViTDet."
echo "  Optional GeoAware-SC checkpoint (template alignment): see docs/INSTALL.md."

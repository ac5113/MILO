#!/usr/bin/env bash
# ===========================================================================
# MILO — unified-environment installer (one conda env, the whole pipeline)
# ===========================================================================
# Builds every pipeline stage into the CURRENTLY ACTIVE conda env: Hunyuan3D-2
# LRM, PyTorch3D rendering, Grounded-SAM-2, ViTDet/ViTPose/HaMeR/HMR2.0 human
# fitting, and the optional GeoAware-SC template alignment.
#
# Target stack: Python 3.10, PyTorch 2.6.0 + CUDA 12.6 (cu126). cu126 is the
# newest target where every component (notably Detectron2 / ViTDet) still builds.
#
# Usage:
#   conda create -n milo python=3.10 && conda activate milo
#   export CUDA_HOME=/path/to/cuda-12.6        # your CUDA 12.6 toolkit
#   bash scripts/install_milo.sh               # source-build detectron2 + pytorch3d (~20 min)
#   USE_PREBUILT_WHEELS=1 bash scripts/install_milo.sh   # fast path (prebuilt wheels)
#
# Afterwards `conda activate milo` is self-sufficient: this script writes an
# activate.d hook with the runtime environment (offscreen-rendering + CUDA/host
# compiler), so no per-session sourcing is needed.
#
# Model weights install SEPARATELY — see scripts/download_models.sh / docs/DATA.md.
# Full guide, prerequisites and troubleshooting: docs/INSTALL.md.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILO="$(cd "$HERE/.." && pwd)"

# --- prerequisites ---------------------------------------------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: no active conda env. Create and activate one first:" >&2
  echo "    conda create -n milo python=3.10 && conda activate milo" >&2
  exit 1
fi
if [ -z "${CUDA_HOME:-}" ]; then
  echo "ERROR: CUDA_HOME is not set. Point it at a CUDA 12.6 toolkit, e.g.:" >&2
  echo "    export CUDA_HOME=/usr/local/cuda-12.6" >&2
  echo "(see docs/INSTALL.md)" >&2
  exit 1
fi

PY="${CONDA_PREFIX}/bin/python"
PIP="$PY -m pip install --no-input"
CONDA="${CONDA_EXE:-conda}"

# --- CUDA-extension build flags --------------------------------------------
# sm_86 (A6000/A100-class) + sm_89 (RTX 6000 Ada / 4090) cover common GPUs.
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"

# --- host compiler: nvcc (cu126) needs gcc <= 12 ---------------------------
# Use the system gcc if it's already old enough; else a system gcc-12 if present;
# else pull gcc/g++-12 from conda-forge. A small auto-generated shim then exposes
# the chosen compiler as plain gcc/g++/cc/c++ on PATH so nvcc picks it up. No
# manual setup — the shim lives under the env and is baked into the activate hook.
gcc_ver="$( (gcc --version 2>/dev/null | head -1 | awk '{print $NF}') || true )"
gcc_major="${gcc_ver%%.*}"; gcc_major="${gcc_major%%[!0-9]*}"; gcc_major="${gcc_major:-0}"
if [ "${gcc_major:-0}" -ge 1 ] && [ "${gcc_major:-0}" -le 12 ]; then
  CC12="$(command -v gcc)"; CXX12="$(command -v g++)"
elif command -v g++-12 >/dev/null 2>&1; then
  CC12="$(command -v gcc-12)"; CXX12="$(command -v g++-12)"
else
  echo "[install_milo] system gcc=${gcc_major:-none} (>12 or missing); installing gcc/g++-12 from conda-forge"
  "$CONDA" install -y -p "$CONDA_PREFIX" -c conda-forge gxx=12
  CC12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  CXX12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
fi
MILO_GCC_SHIM="${CONDA_PREFIX}/etc/milo/gccshim"
mkdir -p "$MILO_GCC_SHIM"
ln -sf "$CC12"  "$MILO_GCC_SHIM/gcc"; ln -sf "$CC12"  "$MILO_GCC_SHIM/cc"
ln -sf "$CXX12" "$MILO_GCC_SHIM/g++"; ln -sf "$CXX12" "$MILO_GCC_SHIM/c++"
export PATH="${MILO_GCC_SHIM}:${PATH}"
export CC="${MILO_GCC_SHIM}/gcc" CXX="${MILO_GCC_SHIM}/g++" CUDAHOSTCXX="${MILO_GCC_SHIM}/g++"
echo "[install_milo] host compiler: $("$CXX12" --version | head -1)"

# --- 1. PyTorch 2.6.0 + cu126 ---------------------------------------------
$PIP torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

# The pip CUDA wheels' libs must come BEFORE any system CUDA/cuDNN directory a
# login shell leaves on LD_LIBRARY_PATH: cuDNN 9's libcudnn.so.9 is a small
# dispatcher that dlopen()s libcudnn_{graph,ops,cnn,adv}, and dlopen searches
# LD_LIBRARY_PATH BEFORE that dispatcher's RUNPATH. A stale system cuDNN 9.0
# mixed into the wheel's 9.5 then dies with `undefined symbol: cudnnGetLibConfig`.
MILO_NVIDIA_LIBS="$("$PY" -c '
import glob, os, sysconfig
print(os.pathsep.join(sorted(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib")))))')"
export LD_LIBRARY_PATH="${MILO_NVIDIA_LIBS:+${MILO_NVIDIA_LIBS}:}${LD_LIBRARY_PATH}"

# --- 2. pinned base (numpy<2 for mmcv/chumpy; setuptools<81 keeps pkg_resources) ---
$PIP "numpy==1.26.4" "setuptools<81"
$PIP scipy scikit-image scikit-learn pandas joblib \
     trimesh opencv-python open3d==0.19.0 pyrender==0.1.45 PyOpenGL==3.1.0 imageio imageio-ffmpeg matplotlib \
     hydra-core omegaconf pyyaml einops tqdm \
     "smplx==0.1.28" "transformers==4.50.2" "diffusers==0.32.2" accelerate safetensors huggingface-hub \
     "timm==0.6.13" "iopath==0.1.9" pygltflib supervision ninja pybind11 cython \
     configer torchgeometry dill \
     "pymeshlab==2023.12.post1" "xatlas==0.0.9" "rembg==2.0.65" "onnxruntime==1.21.0" "numba==0.60.0"  # fit (VPoser) + hy3d deps
# rembg drags in numba/opencv-python-headless that want numpy>=2; re-assert the
# numpy-1.26 ABI and drop the headless cv2 so opencv-python stays the single cv2.
$PY -m pip uninstall -y opencv-python-headless 2>/dev/null || true
$PIP --force-reinstall --no-deps "numpy==1.26.4" "opencv-python==4.11.0.86"
# mmcv 1.3.9 (pure-Python here; ViTPose caps it at <=1.5.0) + chumpy need the env's
# setuptools (pkg_resources), so build them without isolation.
MMCV_WITH_OPS=0 $PIP --no-build-isolation "mmcv==1.3.9"
$PIP xtcocotools
$PIP --no-build-isolation "git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"

# --- 3. detectron2 + pytorch3d + torch-scatter -----------------------------
# Default: source-build (official, reproducible; both compile cleanly on cu126).
# Set USE_PREBUILT_WHEELS=1 to skip the ~20 min compile via the (unofficial)
# MiroPsota prebuilt-wheel index instead. torch-scatter always uses the
# official PyG wheel index.
if [ "${USE_PREBUILT_WHEELS:-0}" = "1" ]; then
  $PIP --extra-index-url https://miropsota.github.io/torch_packages_builder \
       "detectron2==0.6+fd27788pt2.6.0cu126" "pytorch3d==0.7.9+d9839a9pt2.6.0cu126"
else
  $PIP --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
  $PIP --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
fi
$PIP "torch-scatter==2.1.2" -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# --- 4. Hunyuan3D custom ops (in-repo; no wheels exist) --------------------
H3D="$MILO/third-party/Hunyuan3D-2/hy3dgen/texgen"
$PIP --no-build-isolation "$H3D/custom_rasterizer"
$PIP --no-build-isolation "$H3D/differentiable_renderer"

# --- 5. Grounded-SAM-2 (sam2 + groundingdino._C) ---------------------------
GSAM="$MILO/third-party/Grounded-SAM-2"
$PIP --no-deps --no-build-isolation -e "$GSAM"                  # SAM-2 (keeps iopath 0.1.9)
$PIP --no-deps --no-build-isolation -e "$GSAM/grounding_dino"   # builds groundingdino._C

# --- 5b. SAM 3 (default segmenter; shares this env) -------------------------
# --no-deps is deliberate: sam3's iopath>=0.1.10 / timm>=1.0.17 pins are
# over-strict (its code has timm.layers -> timm.models.layers fallbacks and only
# uses iopath's g_pathmgr); a plain install would break detectron2/hamer.
# Verified working on py3.10 / torch 2.6 / timm 0.6.13 — see docs/INSTALL.md.
# NOTE: the facebook/sam3 checkpoint is gated on Hugging Face — request access
# and `huggingface-cli login` before the first run (docs/INSTALL.md).
$PIP "ftfy==6.1.1"
$PIP --no-deps "git+https://github.com/facebookresearch/sam3.git@967fdd651f71ca14949122fed4c918a778ca9334"

# --- 5c. SAM 3D Objects (optional --lrm sam3d backend; shares this env) ------
# Imported from third-party/sam-3d-objects (no pip install of the package).
# Verified on py3.10 / torch 2.6 cu126: attention runs on plain torch sdpa
# (flash-attn/xformers not needed), sparse meshing uses spconv + kaolin's
# flexicubes, texture baking renders gaussians via gsplat, mesh hole-filling
# rasterizes via nvdiffrast's CUDA backend (no OpenGL; JIT-compiles on first
# use with the CUDA_HOME/gcc-shim above, like gsplat). bpy is NOT needed.
# NOTE: the facebook/sam-3d-objects checkpoint is gated on Hugging Face —
# request access and `huggingface-cli login` before the first run (docs/INSTALL.md).
$PIP "spconv-cu126==2.3.8" "astor==0.8.1" loguru easydict optree pyvista pymeshfix igraph plyfile "lightning==2.5.0.post0"
$PIP "kaolin==0.18.0" --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html
# MoGe goes in --no-deps deliberately: its unpinned `gradio` dependency now
# resolves to a gradio that requires huggingface-hub>=1.16, which violates the
# pinned transformers 4.50.2 (<1.0) and breaks run_lrm. Only moge.model.v1 and
# moge.utils.geometry_* are used here (gradio is imported lazily by MoGe's demo
# app alone), so its real deps are the pinned utils3d plus packages already
# installed above. `pip check` therefore reports "moge requires gradio" — inert,
# like the sam3 --no-deps lines above.
$PIP "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"  # MoGe's pinned utils3d
$PIP --no-deps "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"   # pointmap depth
$PIP --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
$PIP "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"   # utils3d hole-fill rasterizer (CUDA backend)
# MoGe/kaolin dependencies may bump numpy; re-assert the numpy-1.26 ABI
# (same fix-up as the rembg block above).
$PIP --force-reinstall --no-deps "numpy==1.26.4"
# Likewise re-assert huggingface-hub<1.0 (what transformers 4.50.2 pins), so a
# re-run also repairs an env where an earlier install pulled hub 1.x in.
$PIP "huggingface-hub>=0.26,<1.0"

# --- 6. HaMeR + ViTPose + HMR2 (4D-Humans), all --no-deps ------------------
# --no-deps so they don't reinstall detectron2/chumpy from git or bump timm.
$PIP "pytorch-lightning==2.5.0.post0" gdown json_tricks munkres "webdataset==0.2.100" rich "pyrootutils==1.0.4"
$PIP --no-deps --no-build-isolation -e "$MILO/third-party/hamer"
$PIP --no-deps --no-build-isolation -e "$MILO/third-party/ViTPose"
$PIP --no-deps --no-build-isolation \
     "git+https://github.com/shubham-goel/4D-Humans.git@6ec79656a23c33237c724742ca2a0ec00b398b53"

# --- 7. the MILO package + chamfer JIT warm-up -----------------------------
$PIP --no-deps --no-build-isolation -e "$MILO"
( cd "$MILO/milo/fit" && "$PY" -c "import torch; from utils.chamfer_distance import ChamferDistance; \
  ChamferDistance()(torch.rand(1,8,3).cuda(), torch.rand(1,8,3).cuda()); print('chamfer JIT OK')" )

# --- 8. Fast-Robust-ICP (FRICP) — binary for the eval metric ---------------
# milo/eval/eval_results.py aligns predicted/GT meshes with Fast-Robust-ICP.
# Build the vendored submodule into third-party/Fast-Robust-ICP/build/FRICP.
FRICP_DIR="$MILO/third-party/Fast-Robust-ICP"
if [ ! -f "$FRICP_DIR/CMakeLists.txt" ]; then
  echo "[install_milo] FRICP submodule missing — run: git submodule update --init $FRICP_DIR (skipping eval binary)"
elif [ -f "$FRICP_DIR/build/FRICP" ]; then
  echo "[install_milo] FRICP already built — skipping."
else
  command -v cmake >/dev/null 2>&1 || "$CONDA" install -y -p "$CONDA_PREFIX" -c conda-forge cmake
  # FRICP's sources include <eigen/Eigen/Dense>, so Eigen must live at include/eigen.
  if [ ! -e "$FRICP_DIR/include/eigen/Eigen/Dense" ]; then
    if [ -e /usr/include/eigen3/Eigen/Dense ]; then
      ln -sfn /usr/include/eigen3 "$FRICP_DIR/include/eigen"          # system Eigen (apt: libeigen3-dev)
    else
      echo "[install_milo] fetching Eigen 3.4.0 for FRICP..."
      wget -qO "$FRICP_DIR/eigen-3.4.0.tar.gz" https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      tar -xf "$FRICP_DIR/eigen-3.4.0.tar.gz" -C "$FRICP_DIR" \
        && rm -rf "$FRICP_DIR/include/eigen" && mv "$FRICP_DIR/eigen-3.4.0" "$FRICP_DIR/include/eigen"
      rm -f "$FRICP_DIR/eigen-3.4.0.tar.gz"
    fi
  fi
  if cmake -S "$FRICP_DIR" -B "$FRICP_DIR/build" -DCMAKE_BUILD_TYPE=Release \
       && cmake --build "$FRICP_DIR/build" -j; then
    echo "[install_milo] FRICP built -> $FRICP_DIR/build/FRICP"
  else
    echo "[install_milo] FRICP build failed — eval metric unavailable (see docs/INSTALL.md)."
  fi
fi

# --- 9. runtime activation hook --------------------------------------------
# So plain `conda activate milo` reproduces the runtime env (offscreen rendering
# + the CUDA toolkit/host compiler the chamfer-distance JIT needs). Idempotent.
ACT_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
cat > "${ACT_DIR}/zzz_milo.sh" <<EOF
# Auto-generated by scripts/install_milo.sh — MILO runtime environment.
# Offscreen rendering (pyrender / pytorch3d):
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="\${EGL_DEVICE_ID:-0}"
export OPENCV_IO_ENABLE_OPENEXR=1
# CUDA toolkit + a gcc<=12 host compiler (runtime chamfer-distance JIT build):
export CUDA_HOME="${CUDA_HOME}"
export PATH="${MILO_GCC_SHIM}:${CUDA_HOME}/bin:\${PATH}"
# Wheel CUDA libs first, so a stale system CUDA/cuDNN already on the shell's
# LD_LIBRARY_PATH cannot shadow them (see the cuDNN note in install_milo.sh).
export LD_LIBRARY_PATH="${MILO_NVIDIA_LIBS:+${MILO_NVIDIA_LIBS}:}${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export CUDAHOSTCXX="${MILO_GCC_SHIM}/g++"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
EOF

echo
echo "[install_milo] Done. Env ready at ${CONDA_PREFIX}"
echo "[install_milo] New shells just need:  conda activate $(basename "$CONDA_PREFIX")"
echo "[install_milo] Next: model weights — bash scripts/download_models.sh  (see docs/DATA.md)"

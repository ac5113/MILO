# Installation

MILO runs end-to-end in a single conda environment, `milo`
(Python 3.10, PyTorch 2.6 + CUDA 12.6). The only exception is the optional
`correspond` step (template alignment), which uses a second env — see
[Optional: correspond](#optional-correspond-template-alignment).

## Prerequisites

- Linux with a CUDA-capable GPU.
- `conda` / `mamba`.
- A CUDA 12.6 toolkit, with `CUDA_HOME` pointing at it.

## Install

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/ac5113/MILO.git && cd MILO
#    (already cloned?  git submodule update --init --recursive)

# 2. Create and activate the env
conda create -n milo python=3.10 && conda activate milo

# 3. Build everything into the active env
export CUDA_HOME=/usr/local/cuda-12.6     # your CUDA 12.6 toolkit
bash scripts/install_milo.sh
```

`install_milo.sh` installs PyTorch 2.6 (cu126), source-builds Detectron2 +
PyTorch3D (`USE_PREBUILT_WHEELS=1` for a faster wheel-based install), and installs
the Hunyuan3D-2 custom ops, SAM 3, SAM 3D Objects deps, Grounded-SAM-2, and the
HaMeR / ViTPose / HMR2.0 stack. It also CMake-builds the vendored
[Fast-Robust-ICP](https://github.com/yaoyx689/Fast-Robust-ICP) binary used by the
eval (needs CMake + Eigen; system Eigen is used if present, else 3.4.0 is fetched).
It writes an `activate.d` hook, so afterwards `conda activate milo` is self-sufficient.

Next step: [download the model weights](DATA.md).

## SAM 3 checkpoint (default segmenter)

SAM 3 itself is installed into the `milo` env by `install_milo.sh`, but its
`facebook/sam3` checkpoint is gated on Hugging Face (one-time): request access at
<https://huggingface.co/facebook/sam3>, then `huggingface-cli login` (or export
`HF_TOKEN`). It auto-downloads into the HF cache on the first run.

## SAM 3D Objects checkpoint (optional `--lrm sam3d` backend)

The dependencies are installed into the `milo` env by `install_milo.sh`. The
`facebook/sam-3d-objects` checkpoint is gated on Hugging Face: request access at
<https://huggingface.co/facebook/sam-3d-objects>, `huggingface-cli login`, then
`bash scripts/download_models.sh` offers it as a prompt (answer `y`; ~13 GB) and
fetches it into `_DATA/sam3d_checkpoints/hf/` (see [DATA.md](DATA.md)). The
default `--lrm hy3d` backend never needs it. DINOv2 auto-downloads via
`torch.hub` on first run.
Upstream requires a GPU with ≥ 32 GB VRAM for inference.

## Optional: `correspond` (template alignment)

The `correspond` step (used only with `--template`) runs
[GeoAware-SC](https://github.com/Junyi42/GeoAware-SC)'s Stable-Diffusion stack in a
separate env. Set it up only if you need template alignment:

1. Create the `geo-aware` env per the GeoAware-SC repo's "Environment Setup", and get
   its `results_spair/best_856.PTH` checkpoint per its README.
2. Bind the two paths to the `milo` env once (plain `export`s also work per shell):
   ```bash
   conda env config vars set -n milo \
     MILO_CORRESPOND_ENV=/path/to/envs/geo-aware \
     MILO_CORRESPOND_CUFFT=/path/to/cuda-11.8/lib64/libcufft.so.10
   conda deactivate && conda activate milo
   ```
   `MILO_CORRESPOND_ENV` is the geo-aware env prefix (the pipeline shells out to
   `<prefix>/bin/python` for this step); `MILO_CORRESPOND_CUFFT` is a
   driver-compatible cuFFT 10.9 (`.so.10`), `LD_PRELOAD`ed for the step.

GeoAware-SC loads DINOv2 via an unpinned `torch.hub.load`, and recent dinov2 `main`
is py3.10-only, which the py3.9 geo-aware env can't parse.
`scripts/download_models.sh` pins a py3.9-compatible dinov2 commit into the geo hub
cache (`${MILO_GEO_CACHE:-_cache/geo}/hub`) automatically; re-run it if you point
`MILO_GEO_CACHE` somewhere fresh.

## Troubleshooting

- **`ModuleNotFoundError: hy3dgen`** — the Hunyuan3D-2 submodule isn't initialized;
  `git submodule update --init --recursive` and re-run `install_milo.sh`.
- **`ModuleNotFoundError` for a heavy package** — make sure `milo` is active.
- **`run_lrm` can't find masks** — the `auto_masks` step generates
  `image_human.png` / `image_object.png` when absent, so it was skipped, failed, or
  found no match. Provide the masks, refine `--object_prompt`, or re-run `auto_masks`
  (see [the pipeline README](../milo/pipeline/README.md)).
- **`ModuleNotFoundError: sam3` / `ftfy`** — the SAM 3 install didn't complete;
  re-run `install_milo.sh`, or use `--segmenter gsam2`.
- **`401` / `GatedRepoError` for `facebook/sam3`** — request access on the
  [SAM 3 HF repo](https://huggingface.co/facebook/sam3) and `huggingface-cli login`.
- **`--lrm sam3d`: missing `_DATA/sam3d_checkpoints/hf/pipeline.yaml`** — the gated
  weights aren't downloaded; see
  [SAM 3D Objects checkpoint](#sam-3d-objects-checkpoint-optional---lrm-sam3d-backend).
- **`ModuleNotFoundError: sam3d_objects`** — the sam-3d-objects submodule isn't
  initialized (`git submodule update --init --recursive`); for missing deps
  (`spconv`, `kaolin`, ...) re-run `install_milo.sh`.
- **SAM 3 finds no match** — use a more descriptive `--object_prompt`
  (`"a green trolleycase"` rather than `"trolley"`), or fall back to `--segmenter gsam2`.
- **`fit` can't find `_DATA`** — the engine anchors `_DATA` to the repo root; the
  pipeline sets the right working directory automatically.
- **`undefined symbol: cudnnGetLibConfig`** (often as a step dying with no
  traceback) — a system cuDNN on `LD_LIBRARY_PATH` is shadowing the wheel's.
  `install_milo.sh` puts the env's `site-packages/nvidia/*/lib` first on
  `LD_LIBRARY_PATH` (in the build and in the activate hook) to prevent this; if
  an older env predates that, re-run `install_milo.sh`, or `unset
  LD_LIBRARY_PATH` before `conda activate milo` (the hook re-adds what it needs).
- **`correspond` `CUFFT_INTERNAL_ERROR`** — set `MILO_CORRESPOND_CUFFT` to a
  driver-compatible cuFFT 10.9 (`.so.10`).
- **`eval_results.py`: "FRICP binary not found"** — build it:
  `cmake -S third-party/Fast-Robust-ICP -B third-party/Fast-Robust-ICP/build -DCMAKE_BUILD_TYPE=Release && cmake --build third-party/Fast-Robust-ICP/build -j`
  (re-running `install_milo.sh` also does this), or point `--fricp_bin` / `FRICP_BIN`
  at an existing build. If the build errors on `eigen/Eigen/Dense`, ensure Eigen is at
  `third-party/Fast-Robust-ICP/include/eigen`.

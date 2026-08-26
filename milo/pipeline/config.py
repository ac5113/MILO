"""
pipeline/config.py

Centralized machine-specific configuration for the pipeline: paths, per-step
interpreters, and dataset-specific naming. Edit this single file when setting up
on a new machine.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Hunyuan3D-2 is vendored as a third-party submodule (third-party/Hunyuan3D-2).
# The run_lrm step injects this onto sys.path so `hy3dgen` imports resolve.
HUNYUAN3D_DIR = os.path.join(REPO_ROOT, "third-party", "Hunyuan3D-2")

# Hunyuan3D-2 weights are pre-fetched by scripts/download_models.sh into
# _DATA/h3d_checkpoints (see docs/DATA.md). hy3dgen resolves checkpoints via the
# HY3DGEN_MODELS env var (default ~/.cache/hy3dgen); the run_lrm step points it
# here so it loads the pre-downloaded weights instead of re-downloading mid-run.
H3D_CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "_DATA", "h3d_checkpoints")

# SAM 3D Objects (optional --lrm sam3d backend) is vendored as a third-party
# submodule (third-party/sam-3d-objects). The run_lrm step injects this onto
# sys.path so `sam3d_objects` imports resolve.
SAM3D_DIR = os.path.join(REPO_ROOT, "third-party", "sam-3d-objects")

# SAM 3D Objects weights (HF-gated facebook/sam-3d-objects) are pre-fetched by
# scripts/download_models.sh into _DATA/sam3d_checkpoints/hf, which holds the
# pipeline.yaml the run_lrm step instantiates the model from.
SAM3D_CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "_DATA", "sam3d_checkpoints")

# ---------------------------------------------------------------------------
# Interpreter each pipeline step runs in
# ---------------------------------------------------------------------------
# MILO runs in a single unified conda environment (see scripts/install_milo.sh),
# so every step — including tmpl_render and template_align — uses the current
# interpreter. Only the semantic-correspondence step (Stable Diffusion + DINOv2)
# needs a separate env: GeoAware-SC's torch 1.13/cu118 stack does not fit the
# unified `milo` env. It is configured entirely from the environment — no machine
# paths are hard-coded for release:
#   MILO_CORRESPOND_ENV    — geo-aware conda env prefix; the correspond
#                            interpreter is <prefix>/bin/python.
#   MILO_CORRESPOND_CUFFT  — driver-compatible cuFFT 10.9 (.so.10). Must be set
#                            explicitly (it lives outside the env); run_pipeline
#                            LD_PRELOADs it for the correspond step so torch-1.13's
#                            cuFFT doesn't crash with CUFFT_INTERNAL_ERROR on
#                            newer host drivers.
MILO_PYTHON = sys.executable

_STEPS = (
    "run_lrm", "render", "img_segment", "mesh_segment", "kp2d",
    "triangulate", "init_smpl", "fit", "isolate",
    "tmpl_render", "correspond", "template_align", "render_final",
)
STEP_PYTHON = {step: MILO_PYTHON for step in _STEPS}
if os.environ.get("MILO_CORRESPOND_ENV"):
    STEP_PYTHON["correspond"] = os.path.join(
        os.environ["MILO_CORRESPOND_ENV"], "bin", "python"
    )

# ---------------------------------------------------------------------------
# Generic single-image fit config (confs/data/image.yaml) — the default,
# dataset-agnostic inference path. Used when no --dataset is given.
# ---------------------------------------------------------------------------
DEFAULT_FIT_CFG = "image"

# ---------------------------------------------------------------------------
# Back-compat: dataset shorthand → fit hydra config name (confs/data/<cfg>.yaml).
# Only used by the optional --dataset flag, which re-enables the dataset-specific
# eval paths (folder-name object naming, per-object template-by-id resolution).
# Plain single-image inference does not need any of this.
# ---------------------------------------------------------------------------
DATASET_CFG = {
    "hodome":   "hodome_test",
    "intercap": "intercap_test",
    "imhd":     "imhd_test",
}

DATASETS = list(DATASET_CFG.keys())

# ---------------------------------------------------------------------------
# Dataset-specific object naming
# ---------------------------------------------------------------------------
# intercap: 1-indexed object ID → text label
_INTERCAP_MAPPING = [
    'trolley', 'skateboard', 'sports ball', 'umbrella', 'tennis racquet',
    'suitcase', 'chair', 'bottle', 'cup', 'stool',
]


def obj_name(folder_name: str, dataset: str) -> str:
    """Extract the object text label from a sequence folder name."""
    if dataset == "hodome":
        return folder_name.split("__")[0].split("_")[-1]
    elif dataset == "imhd":
        return folder_name.split("__")[1].split("_")[-1]
    elif dataset == "intercap":
        obj_idx = int(folder_name.split("__")[1]) - 1
        return _INTERCAP_MAPPING[obj_idx]
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {DATASETS}")


def resolve_object(folder_name: str, dataset: str, object_label: str) -> str:
    """Resolve the object text label for one sequence.

    Default (dataset-agnostic) inference passes the label directly via --object.
    The optional --dataset back-compat mode instead derives it from the folder
    name (e.g. InterCap's 1-indexed object id).
    """
    if dataset:
        return obj_name(folder_name, dataset)
    return object_label or "object"


def fit_dir(seq_dir: str) -> str:
    """Per-sequence fit (run_opt.py) output dir: <seq_dir>/fit, matching the fit
    hydra run dir ${data.root}/${data.seq}/fit."""
    return os.path.join(seq_dir, "fit")


# ---------------------------------------------------------------------------
# Template-mesh resolution (optional template-alignment steps)
# ---------------------------------------------------------------------------
def template_path(folder_name: str, dataset: str, template: str) -> str:
    """Resolve the object template mesh for one sequence.

    `template` may be a single mesh file (used as-is for every sequence) or a
    directory of per-object templates resolved by object id / label (e.g. a
    directory of `<id>.obj` meshes).
    """
    if os.path.isfile(template):
        return template
    if os.path.isdir(template):
        if dataset == "intercap":
            obj_id = folder_name.split("__")[1]          # e.g. "01" → objects/01.obj
            return os.path.join(template, f"{obj_id}.obj")
        return os.path.join(template, f"{obj_name(folder_name, dataset)}.obj")
    return template

"""
pipeline/steps/run_lrm.py

LRM reconstruction — generate the combined human+object textured mesh
(full_img_textured.glb) that the render step consumes, from a sequence's input
image plus its human and object masks. Two backends, selected via --lrm:

  hy3d (default) — Hunyuan3D-2. The single-sequence core of the Hunyuan3D-2
      generation script, without the batch / producer-consumer machinery.
      hy3dgen is imported from the third-party/Hunyuan3D-2 submodule
      (config.HUNYUAN3D_DIR). The two masks are unioned, smoothed and packed
      into the alpha channel.
  sam3d — SAM 3D Objects. sam3d_objects is imported from the
      third-party/sam-3d-objects submodule (config.SAM3D_DIR); the model is
      instantiated from the pre-fetched checkpoints' pipeline.yaml
      (config.SAM3D_CHECKPOINTS_DIR). The two masks are unioned WITHOUT any
      smoothing. The output mesh comes out in the same LRM frame Hunyuan3D
      emits (no flip needed), so downstream steps are backend-agnostic.

Inputs (seq_dir convention, auto-discovered):
    <seq_dir>/<name>.jpg            input image (mask PNGs are skipped)
    <seq_dir>/image_human.png       human mask  (or <imgbase>_human.png)
    <seq_dir>/image_object.png      object mask (or <imgbase>_object.png)
Output:
    <seq_dir>/full_img_textured.glb combined textured mesh, in camera frame

Standalone usage:
    python -m milo.pipeline.steps.run_lrm --seq_dir /path/to/seq [--lrm sam3d]

Module usage:
    from milo.pipeline.steps.run_lrm import run
    run(seq_dir="/path/to/seq", lrm="sam3d")
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

from milo.pipeline.config import (
    HUNYUAN3D_DIR, H3D_CHECKPOINTS_DIR, SAM3D_DIR, SAM3D_CHECKPOINTS_DIR,
)
from milo.pipeline.steps._common import find_image as _find_image
from milo.pipeline.steps._log import vprint, set_verbose

# Make `hy3dgen` (Hunyuan3D-2 submodule) importable.
if HUNYUAN3D_DIR not in sys.path:
    sys.path.insert(0, HUNYUAN3D_DIR)

# Resolve Hunyuan3D-2 weights from the pre-fetched _DATA/h3d_checkpoints dir
# (scripts/download_models.sh) instead of re-downloading. hy3dgen reads this at
# from_pretrained() time; setdefault respects an explicit override.
os.environ.setdefault("HY3DGEN_MODELS", H3D_CHECKPOINTS_DIR)


def _output_path(seq_dir: str) -> str:
    return os.path.join(seq_dir, "full_img_textured.glb")


def _find_mask(img_root: str, img_base: str, kind: str) -> str:
    """Locate a human/object mask. Prefers the fixed name `image_<kind>.png`,
    then `<img_base>_<kind>.png` (or its `__`-suffix), covering both the generic
    single-image contract and dataset-style names. Falls back to the canonical
    fixed name so a missing-mask error points at the expected path."""
    suffix = img_base.split("__")[-1]
    for name in (f"image_{kind}.png", f"{img_base}_{kind}.png", f"{suffix}_{kind}.png"):
        p = os.path.join(img_root, name)
        if os.path.exists(p):
            return p
    return os.path.join(img_root, f"image_{kind}.png")


def _load_image_and_masks(img_path: str):
    """Load the input image (RGB) plus the human and object masks (grayscale)."""
    img_root = os.path.dirname(img_path)
    img_base = os.path.splitext(os.path.basename(img_path))[0]
    mask_hum_path = _find_mask(img_root, img_base, "human")
    mask_obj_path = _find_mask(img_root, img_base, "object")

    image = cv2.imread(img_path)
    if image is None:
        raise IOError(f"Could not read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_hum = cv2.imread(mask_hum_path, cv2.IMREAD_GRAYSCALE)
    mask_obj = cv2.imread(mask_obj_path, cv2.IMREAD_GRAYSCALE)
    if mask_hum is None or mask_obj is None:
        raise IOError(
            f"Missing mask(s): {mask_hum_path} / {mask_obj_path}. "
            "The run_lrm step consumes pre-computed human/object masks."
        )
    return image, mask_hum, mask_obj


def _preprocess_hy3d(img_path: str):
    """Load image + human/object masks and return a combined RGBA PIL image.

    Mirrors gen_intercap.preprocess_image: combine the two masks, smooth the
    union, and pack it into the alpha channel.
    """
    image, mask_hum, mask_obj = _load_image_and_masks(img_path)

    mask_combined = np.clip(
        mask_hum.astype(np.int16) + mask_obj.astype(np.int16),
        a_min=0, a_max=255,
    ).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.GaussianBlur(mask_combined, (15, 15), 0)
    mask_combined = cv2.addWeighted(
        mask_combined, 0.8,
        cv2.GaussianBlur(mask_combined, (31, 31), 0), 0.2, 0,
    )

    image_with_mask = np.concatenate((image, mask_combined[:, :, None]), axis=2)
    return Image.fromarray(image_with_mask)


def _preprocess_sam3d(img_path: str) -> np.ndarray:
    """Load image + human/object masks and return a combined RGBA numpy image.

    Plain binary union of the two masks in the alpha channel — no smoothing
    (SAM 3D Objects expects a hard mask)."""
    image, mask_hum, mask_obj = _load_image_and_masks(img_path)

    union = (
        (mask_hum.astype(np.int16) + mask_obj.astype(np.int16)) > 0
    ).astype(np.uint8) * 255

    return np.concatenate((image, union[:, :, None]), axis=2)


def _run_hy3d(pil_image, out_path: str) -> None:
    """Hunyuan3D-2 backend: shape gen + texture paint on the RGBA input."""
    # Heavy imports / model loads only once we know we will run.
    import torch
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FloaterRemover,
        DegenerateFaceRemover,
    )
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh

    vprint("[run_lrm:hy3d] Loading Hunyuan3D pipelines...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0",
        use_safetensors=True,
        device="cuda",
    )
    texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    floater_remover = FloaterRemover()
    degenerate_remover = DegenerateFaceRemover()

    vprint("[run_lrm:hy3d] Running shape generation...")
    generator = torch.Generator(device="cpu").manual_seed(1234)
    outputs = shape_pipeline(
        image=pil_image,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        octree_resolution=256,
        num_chunks=200000,
        output_type="mesh",
    )
    mesh = export_to_trimesh(outputs)[0]
    mesh.process(validate=True)
    mesh = floater_remover(mesh)
    mesh = degenerate_remover(mesh)

    vprint("[run_lrm:hy3d] Running texture generation...")
    textured_mesh = texture_pipeline(mesh, image=pil_image)

    textured_mesh.export(out_path)


def _run_sam3d(rgba_image: np.ndarray, out_path: str) -> None:
    """SAM 3D Objects backend: instantiate the pipeline from the pre-fetched
    checkpoints and export the texture-baked mesh."""
    config_path = os.path.join(SAM3D_CHECKPOINTS_DIR, "hf", "pipeline.yaml")
    if not os.path.exists(config_path):
        raise IOError(
            f"SAM 3D Objects checkpoints not found: {config_path}. "
            "Download the HF-gated facebook/sam-3d-objects weights first "
            "(see scripts/download_models.sh / docs/INSTALL.md)."
        )

    # sam3d_objects.init (distributed/env setup) is unnecessary for inference —
    # the reference notebook wrapper skips it the same way.
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    # Pin the attention backends to sdpa (pure torch; flash-attn/xformers are
    # not installed). The env must be latched BEFORE sam3d_objects' pipeline
    # module is imported: on A100/H100/H200 its import force-sets
    # ATTN_BACKEND=flash_attn, but the attention/sparse modules read the env
    # exactly once at their own import — importing them first wins.
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    if SAM3D_DIR not in sys.path:
        sys.path.insert(0, SAM3D_DIR)

    import sam3d_objects.model.backbone.tdfy_dit.modules.attention  # noqa: F401  (latches sdpa)
    import sam3d_objects.model.backbone.tdfy_dit.modules.sparse  # noqa: F401  (latches sdpa + spconv)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Texture baking renders the gaussians via render_multiview, which does not
    # set a renderer backend and so falls back to "inria" — that needs the
    # mip-splatting fork of diff_gaussian_rasterization, which is not installed.
    # Default render_frames to the "gsplat" backend instead (upstream's own
    # demo renders with it); an explicit options["backend"] still wins.
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

    _orig_render_frames = render_utils.render_frames

    def _render_frames_gsplat(sample, extrinsics, intrinsics, options={}, **kwargs):
        return _orig_render_frames(
            sample, extrinsics, intrinsics, {"backend": "gsplat", **options}, **kwargs
        )

    render_utils.render_frames = _render_frames_gsplat

    vprint("[run_lrm:sam3d] Loading SAM 3D Objects pipeline...")
    config = OmegaConf.load(config_path)
    # Texture baking on the pytorch3d engine (as upstream's inference wrapper);
    # nvdiffrast is still needed by the hole-filling rasterizer (utils3d, CUDA
    # backend — no OpenGL).
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = os.path.dirname(config_path)
    pipeline = instantiate(config)

    vprint("[run_lrm:sam3d] Running reconstruction...")
    out = pipeline.run(
        rgba_image,
        None,  # mask already embedded in the alpha channel
        seed=42,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
        decode_formats=["mesh", "gaussian"],
    )
    glb = out.get("glb")
    if glb is None:
        raise RuntimeError(
            "SAM 3D Objects returned no mesh (out['glb'] is None) — "
            "mesh decoding/postprocessing failed upstream."
        )

    # SAM3D's output is already in the same frame as hy3d's (LRM frame) —
    # verified visually on the demo sequence; no flip. Downstream consumers
    # stay backend-agnostic.
    glb.export(out_path)


def run(seq_dir: str, lrm: str = "hy3d", verbose: bool = False) -> bool:
    """
    Generate the combined human+object textured mesh for one sequence.

    Args:
        seq_dir: Path to sequence directory.
        lrm: Reconstruction backend, "hy3d" (default) or "sam3d".

    Returns:
        True if the step ran, False if skipped.
    """
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    out_path = _output_path(seq_dir)

    img_path = _find_image(seq_dir)
    if img_path is None:
        print(f"[run_lrm] Skipping — no input image (*.jpg) found in {seq_dir}")
        return False

    vprint(f"[run_lrm:{lrm}] Preprocessing {img_path}")
    if lrm == "sam3d":
        _run_sam3d(_preprocess_sam3d(img_path), out_path)
    else:
        _run_hy3d(_preprocess_hy3d(img_path), out_path)

    print(f"[run_lrm:{lrm}] Done → {out_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LRM reconstruction of the combined human+object mesh for "
                    "one sequence (Hunyuan3D-2 or SAM 3D Objects)"
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory")
    parser.add_argument(
        "--lrm", default="hy3d", choices=["hy3d", "sam3d"],
        help="Reconstruction backend: Hunyuan3D-2 (default) or SAM 3D Objects.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, lrm=args.lrm, verbose=args.verbose)

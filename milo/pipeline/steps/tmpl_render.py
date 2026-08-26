"""
pipeline/steps/tmpl_render.py

Optional template-alignment pre-step. Renders an object TEMPLATE mesh (the
`--template` argument) from the same multi-view set used for the LRM renders,
writing grey shaded views + visible-vertex arrays into
`render_segment/renders_gt/`, plus a binary object mask per view into
`render_segment/segments_gt/` taken straight from the rasterizer depth (the whole
rendered object is the template, so depth>0 IS the mask — no thresholding or
extra segmentation model needed). These drive the geometry-aware semantic
correspondence stage (`correspond`) against the LRM object renders.

Convention matches the reference template renderer (render_intercap_gt.py,
mesh_type='original'): the template is centered at its centroid, rotated 180°
about X, rendered grey (flat ambient) at a fit-to-frame distance using the
sequence's metadata intrinsics. The same convention is reconstructed by the
correspondence stage to project template vertices, so it MUST stay in sync.

Standalone usage:
    python -m milo.pipeline.steps.tmpl_render --seq_dir /path/to/seq \
        --template /path/to/template.obj

Module usage:
    from milo.pipeline.steps.tmpl_render import run
    run(seq_dir="/path/to/seq", template="/path/to/template.obj")
"""

import argparse
import glob
import os

import numpy as np
import torch
import trimesh
from pytorch3d.renderer import look_at_view_transform

# Reuse the LRM renderer + the shared batched render/visibility helper so the template
# and LRM renders share one camera convention and one (byte-identical) code path.
from milo.pipeline.steps.render import (
    _render_and_visibility_batched,
    _load_img,
    _DEFAULT_ROTATION_ANGLES,
)
from milo.pipeline.steps._log import set_verbose

_GT_SUBDIR = "renders_gt"
_SEG_GT_SUBDIR = "segments_gt"


def _output_dir(seq_dir: str) -> str:
    return os.path.join(seq_dir, "render_segment", _GT_SUBDIR)


def _seg_output_dir(seq_dir: str) -> str:
    return os.path.join(seq_dir, "render_segment", _SEG_GT_SUBDIR)


def _render_template_views(seq_dir, template_path, obj_name, rotation_angles, device,
                           backend="pyrender"):
    """Render the template mesh grey from every (elevation, azimuth) view and save
    PNGs + visible-vertex arrays into render_segment/renders_gt/, plus a binary
    object mask per view (depth>0) into render_segment/segments_gt/."""
    save_path = _output_dir(seq_dir)
    os.makedirs(save_path, exist_ok=True)
    seg_save_path = _seg_output_dir(seq_dir)
    os.makedirs(seg_save_path, exist_ok=True)
    obj_us = obj_name.replace(" ", "_")

    img_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No *.jpg found in {seq_dir}")
    full_img = _load_img(img_paths[0])
    pad = [
        abs((full_img.shape[0] // 2) * 2 - full_img.shape[0]),
        abs((full_img.shape[1] // 2) * 2 - full_img.shape[1]),
    ]
    H = full_img.shape[0] + pad[0]
    W = full_img.shape[1] + pad[1]

    _meta_path = os.path.join(seq_dir, "metadata.npz")
    if os.path.exists(_meta_path):
        metadata = dict(np.load(_meta_path))
        focal = np.asarray(metadata["focal_length"], dtype=np.float64)
        princpt = np.asarray(metadata["principal_point"], dtype=np.float64)
    else:
        _f = 0.5 * (H + W)
        focal = np.array([_f, _f], dtype=np.float64)
        princpt = np.array([W / 2.0, H / 2.0], dtype=np.float64)
        print(f"[tmpl_render] No metadata.npz — default focal={_f:.1f}, principal=center")

    # --- load + canonicalize the template: center + rot_x_180 (reference convention) ---
    mesh = trimesh.load(template_path, process=False, maintain_order=True, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts = verts - verts.mean(axis=0)
    rot_x_180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
    verts = verts @ rot_x_180.T
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # flat grey per-face atlas (F,1,1,3) = 0.5  -> reuses _RendererMesh's TexturesAtlas path
    grey_atlas = np.full((len(faces), 1, 1, 3), 0.5, dtype=np.float32)

    image_size = torch.tensor([[H, W]], device=device)

    # fit-to-frame distance (reference render_intercap_gt convention; no *1.1 margin)
    max_extent = max(np.linalg.norm(verts.min(0)), np.linalg.norm(verts.max(0)))
    distance = max(
        (max_extent * focal[0]) / (W / 2.0),
        (max_extent * focal[1]) / (H / 2.0),
    )

    verts_t = torch.tensor(verts, device=device)
    faces_t = torch.tensor(faces)
    atlas_t = torch.tensor(grey_atlas)
    flip = torch.tensor(
        [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]], device=device
    )

    # Enumerate views once, stacking per-view camera transforms and
    # directional-light dirs (-T/‖T‖).
    views, R_list, T_list, light_list = [], [], [], []
    for elevation_raw in rotation_angles:
        elevation = elevation_raw - 180
        if elevation <= -90 or elevation >= 90:
            continue
        for azimuth in rotation_angles:
            R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
            views.append((elevation, azimuth))
            R_list.append(R)
            T_list.append(T)
            light_list.append(-T / torch.norm(T))  # fixed camera-space light direction

    def _name_fn(e, a):
        return (
            os.path.join(save_path, f"rend_img_obj_e{e}_a{a}.png"),
            os.path.join(save_path, f"visible_vertices_e{e}_a{a}.npy"),
        )

    def _mask_fn(e, a):
        # Binary template mask straight from the rasterizer depth: the whole rendered
        # object IS the template, so depth>0 marks every template pixel (hard raster,
        # blur_radius=0 → crisp silhouette, no thresholding).
        return os.path.join(seg_save_path, f"rend_img_{obj_us}_e{e}_a{a}.png")

    _render_and_visibility_batched(
        views=views,
        R_all=torch.cat(R_list, dim=0),
        T_all=torch.cat(T_list, dim=0),
        verts_t=verts_t,
        faces_t=faces_t,
        atlas_t=atlas_t,
        focal=focal,
        princpt=princpt,
        image_size=image_size,
        flip_t=flip,
        device=device,
        name_fn=_name_fn,
        directional_light=True,
        light_dirs=torch.cat(light_list, dim=0),
        mask_fn=_mask_fn,
        coord_dtype=torch.long,
        backend=backend,
    )

    print(f"[tmpl_render] Done → {save_path} (+ masks → {seg_save_path})")


def run(seq_dir: str, template: str, object_label: str = "object",
        verbose: bool = False, use_pytorch3d: bool = False) -> bool:
    """Render the template mesh into render_segment/renders_gt/ (+ binary masks
    into render_segment/segments_gt/) for one sequence."""
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    if not os.path.isfile(template):
        print(f"[tmpl_render] Skipping — template mesh not found: {template}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "pytorch3d" if use_pytorch3d else "pyrender"
    print(f"[tmpl_render] Rendering template {os.path.basename(template)} for "
          f"{os.path.basename(seq_dir)} (renderer={backend})")
    _render_template_views(seq_dir, template, object_label, _DEFAULT_ROTATION_ANGLES,
                           device, backend=backend)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an object template into renders_gt/ for one sequence")
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory")
    parser.add_argument("--template", required=True, help="Path to the object template mesh (.obj/.glb)")
    parser.add_argument("--object", default="object",
                        help="Object text label (e.g. 'chair') used to name the mask files. Default: 'object'.")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    parser.add_argument("--pytorch3d", action="store_true",
                        help="Use the pytorch3d renderer instead of the default pyrender backend.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, template=args.template, object_label=args.object,
        verbose=args.verbose, use_pytorch3d=args.pytorch3d)

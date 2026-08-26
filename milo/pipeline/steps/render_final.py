"""
pipeline/steps/render_final.py

Final qualitative renders of the collated MILO deliverables. Saves white-background,
flat-colored PNGs of the two reconstructions, each from a single front view:
  render_human_object.png    fitted_human.obj + segmented_object.obj
  render_human_template.png  fitted_human.obj + aligned_template.obj   (only if present)

Runs after `collate`, on the top-level final meshes (which are co-registered in the
LRM frame). Renders via renderer_utils.make_renderer (pyrender by default, pytorch3d
under --pytorch3d) so the camera convention matches the other renders; the two meshes
are merged into one and colored with each mesh's own vertex colors, then the
silhouette is composited onto a white background.

Standalone usage:
    python -m milo.pipeline.steps.render_final --seq_dir /path/to/seq

Module usage:
    from milo.pipeline.steps.render_final import run
    run(seq_dir="/path/to/seq")
"""

import argparse
import os

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import look_at_view_transform

from milo.pipeline.steps.renderer_utils import make_renderer
from milo.pipeline.steps._log import vprint, set_verbose

# Fallback flat colors (RGB in [0, 1]) used only when a mesh carries no vertex
# colors of its own; otherwise the mesh's stored vertex colors are used.
HUMAN_COLOR = (0.66, 0.74, 0.85)      # soft blue
OBJECT_COLOR = (0.95, 0.55, 0.25)     # orange

_CANVAS = 1024                         # square render canvas (px)
_FRONT_ELEV = 0.0                      # single front view
_FRONT_AZIM = 0.0
_FILL_FRAC = 0.40                      # silhouette radius as a fraction of the canvas

# (output png, object mesh, match_object_color). The human is shared and always
# uses its own vertex colors. match_object_color=False colors the object mesh by
# its own vertex colors; True renders it in the segmented object's color (so the
# aligned template comes out the same color as the object).
_PAIRS = [
    ("render_human_object.png", "segmented_object.obj", False),
    ("render_human_template.png", "aligned_template.obj", True),
]


def _load_mesh(seq_dir, name):
    """Load a final mesh, looking at the top of the folder first, then the
    intermediate_results/ and correspondences/ fallbacks (for an uncollated run).
    Returns a Trimesh with faces, or None if missing/empty."""
    for rel in (name, os.path.join("intermediate_results", name),
                os.path.join("correspondences", name)):
        p = os.path.join(seq_dir, rel)
        if os.path.isfile(p):
            m = trimesh.load(p, process=False, force="mesh")
            if m is not None and hasattr(m, "faces") and len(m.faces) > 0:
                return m
            return None
    return None


def _vertex_rgb(mesh):
    """The mesh's per-vertex RGB in [0, 1], or None if it carries no vertex colors."""
    try:
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
        return vc if len(vc) == len(mesh.vertices) else None
    except Exception:
        return None


def _face_colors(mesh, flat_rgb):
    """Per-face RGB in [0, 1]. If flat_rgb is given, every face takes that color;
    otherwise the mesh's own vertex colors are averaged per face (falling back to
    flat HUMAN/OBJECT color via the caller when the mesh has none)."""
    if flat_rgb is not None:
        return np.tile(np.asarray(flat_rgb, np.float32), (len(mesh.faces), 1))
    vc = _vertex_rgb(mesh)
    if vc is None:
        return None
    return vc[np.asarray(mesh.faces)].mean(axis=1)


def _render_pair(human, obj, obj_flat_color, device, backend="pyrender"):
    """Render human + obj together: white bg, single front view. The human uses its
    own vertex colors; the object uses obj_flat_color if given, else its own vertex
    colors. Returns a uint8 BGR image ready for cv2.imwrite."""
    vh, fh = np.asarray(human.vertices, np.float64), np.asarray(human.faces, np.int64)
    vo, fo = np.asarray(obj.vertices, np.float64), np.asarray(obj.faces, np.int64)

    # Merge into one mesh (offset the object faces) and build the per-face color atlas.
    verts = np.concatenate([vh, vo], axis=0)
    faces = np.concatenate([fh, fo + len(vh)], axis=0)
    human_fc = _face_colors(human, None)
    if human_fc is None:
        human_fc = np.tile(np.asarray(HUMAN_COLOR, np.float32), (len(fh), 1))
    obj_fc = _face_colors(obj, obj_flat_color)
    if obj_fc is None:
        obj_fc = np.tile(np.asarray(OBJECT_COLOR, np.float32), (len(fo), 1))
    atlas = np.concatenate([human_fc, obj_fc], axis=0)[:, None, None, :].astype(np.float32)

    # Center at the combined bounding-box center so the front view frames both
    # meshes symmetrically (the vertex mean is pulled toward the dense object PC).
    verts = verts - 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    max_extent = float(np.linalg.norm(verts, axis=1).max())

    focal = float(_CANVAS)
    princpt = _CANVAS / 2.0
    distance = (max_extent * focal) / (_FILL_FRAC * _CANVAS)

    R, T = look_at_view_transform(distance, _FRONT_ELEV, _FRONT_AZIM, device=device)
    renderer = make_renderer(
        backend,
        image_size=[_CANVAS, _CANVAS],
        focal_length=torch.tensor([focal, focal]).float(),
        principal_point=torch.tensor([princpt, princpt]).float(),
        rotation=R,
        translation=T,
        # Light from the camera side (front), so the camera-facing surfaces are lit
        # rather than in shadow. (tmpl_render uses -T for its back-lit grey GT look.)
        light_direction=T / torch.norm(T),
        device=device,
        directional_light=True,   # shade the flat colors so the meshes read as 3D
    )
    rendered, depth = renderer.render(
        verts=torch.tensor(verts).float(),
        faces=torch.tensor(faces),
        texture=torch.tensor(atlas),
    )

    rgb = rendered[..., :3].clamp(0.0, 1.0).cpu().numpy()
    silhouette = (depth.cpu().numpy() > 0)[..., None]
    composited = rgb * silhouette + (1.0 - silhouette)   # white background
    img = (composited * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def run(seq_dir: str, verbose: bool = False, use_pytorch3d: bool = False) -> bool:
    """Render the final deliverables for one sequence. Returns True if it ran."""
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)

    human = _load_mesh(seq_dir, "fitted_human.obj")
    if human is None:
        print(f"[render_final] Skipping — fitted_human.obj not found in {seq_dir}.")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "pytorch3d" if use_pytorch3d else "pyrender"
    obj_match_color = None  # the segmented object's color, so the template can match it
    wrote = []
    for png, obj_name, match_object_color in _PAIRS:
        out_p = os.path.join(seq_dir, png)
        obj = _load_mesh(seq_dir, obj_name)
        if obj is None:
            vprint(f"[render_final] {obj_name} not found — skipping {png}.")
            continue
        if match_object_color:
            if obj_match_color is None:
                seg = _load_mesh(seq_dir, "segmented_object.obj")
                seg_rgb = _vertex_rgb(seg) if seg is not None else None
                obj_match_color = tuple(seg_rgb.mean(axis=0)) if seg_rgb is not None else OBJECT_COLOR
            obj_flat = obj_match_color
        else:
            obj_flat = None
        cv2.imwrite(out_p, _render_pair(human, obj, obj_flat, device, backend=backend))
        wrote.append(png)

    name = os.path.basename(seq_dir)
    if wrote:
        print(f"[render_final] {name}: wrote {', '.join(wrote)}")
    else:
        print(f"[render_final] {name}: nothing to render "
              f"(no segmented_object.obj / aligned_template.obj).")
    return bool(wrote)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render the final MILO deliverables (human+object, human+template) "
                    "as white-bg colored PNGs for one sequence."
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--pytorch3d", action="store_true",
                        help="Use the pytorch3d renderer instead of the default pyrender backend.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, verbose=args.verbose, use_pytorch3d=args.pytorch3d)

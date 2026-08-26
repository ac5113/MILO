"""
pipeline/steps/render.py

Multi-view render of LRM mesh — saves rendered PNGs and visible_vertices .npy
files into render_segment/renders/.

Runs BEFORE img_segment; does NOT do vertex classification (that is mesh_segment.py).

Standalone usage:
    python -m milo.pipeline.steps.render --seq_dir /path/to/seq

Module usage:
    from milo.pipeline.steps.render import run
    run(seq_dir="/path/to/seq")
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.io import load_obj

from milo.pipeline.steps.renderer_utils import _perspective_projection, make_renderer
from milo.pipeline.steps._log import vprint, set_verbose

_DEFAULT_ROTATION_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# Views rasterised per GPU batch in _render_and_visibility_batched. Lower if a big
# mesh / high resolution OOMs; output is identical for any value.
_RENDER_BATCH = 8

# One-sided depth slack for the pyrender visibility test (see _render_and_visibility_batched):
# occlusion only ever makes a vertex's own depth *larger* than the rendered surface's, so the
# slack tolerates it being trivially larger (float/rasteriser noise), never smaller. Tuned
# empirically against the pytorch3d reference. The pytorch3d backend keeps its symmetric 1e-3 test.
_VIS_EPS = 0.003


# ---------------------------------------------------------------------------
# Rendering utilities
# ---------------------------------------------------------------------------

def _load_img(path, order="RGB"):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)
    if order == "RGB":
        img = img[:, :, ::-1]
    return img.astype(np.float32)


def _write_png(path, rend_img_obj):
    """Write one rendered view as PNG (BGR→RGB, clip to [0, 255], uint8)."""
    cv2.imwrite(
        path,
        np.clip(cv2.cvtColor(rend_img_obj, cv2.COLOR_BGR2RGB), 0, 255).astype(np.uint8),
    )


def _load_mesh_texture(mesh_path):
    vprint(f"Loading textures from {mesh_path}")
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".glb":
        return _load_mesh_texture_glb(mesh_path)
    _, _, aux = load_obj(mesh_path, load_textures=True, create_texture_atlas=True)
    if aux.texture_atlas is not None and aux.texture_atlas.shape[0] > 0:
        vprint(f"Texture atlas found with shape {aux.texture_atlas.shape}")
        return aux.texture_atlas, True
    vprint("No texture found in mesh file, using default vertex coloring")
    return aux.texture_atlas, False


def _load_mesh_texture_glb(mesh_path, atlas_res=4):
    """Extract per-face texture atlas from a GLB mesh (UV-mapped image texture).

    Returns (texture_atlas, mesh_has_texture) matching the format of _load_mesh_texture.
    texture_atlas shape: (F, atlas_res, atlas_res, 3) float32 in [0, 1].
    """
    mesh = trimesh.load(mesh_path, force="mesh", process=False, maintain_order=True)

    visual = mesh.visual
    # Try to get UV-mapped texture
    if hasattr(visual, "to_texture") or hasattr(visual, "uv"):
        try:
            if not hasattr(visual, "uv") or visual.uv is None:
                raise AttributeError("no uv")
            uv = np.array(visual.uv, dtype=np.float32)  # (V, 2)
            faces = np.array(mesh.faces, dtype=np.int64)  # (F, 3)

            # Get texture image
            if hasattr(visual, "material") and hasattr(visual.material, "baseColorTexture") and visual.material.baseColorTexture is not None:
                tex_img = np.array(visual.material.baseColorTexture, dtype=np.float32) / 255.0
            elif hasattr(visual, "material") and hasattr(visual.material, "image") and visual.material.image is not None:
                tex_img = np.array(visual.material.image, dtype=np.float32) / 255.0
            else:
                raise AttributeError("no texture image")

            if tex_img.ndim == 2:
                tex_img = tex_img[:, :, None].repeat(3, axis=2)
            tex_img = tex_img[:, :, :3]
            H, W = tex_img.shape[:2]

            # Per-face atlas: sample the texture at barycentric positions within each face
            F = faces.shape[0]
            atlas = np.zeros((F, atlas_res, atlas_res, 3), dtype=np.float32)
            face_uvs = uv[faces]  # (F, 3, 2)
            for ri in range(atlas_res):
                for ci in range(atlas_res):
                    # Barycentric-like weights for this atlas-tile sample
                    u_norm = (ci + 0.5) / atlas_res
                    v_norm = (ri + 0.5) / atlas_res
                    w0 = 1.0 - u_norm - v_norm * 0.5
                    w1 = u_norm
                    w2 = v_norm * 0.5
                    w_sum = w0 + w1 + w2
                    w0, w1, w2 = w0 / w_sum, w1 / w_sum, w2 / w_sum
                    sample_uv = (
                        w0 * face_uvs[:, 0, :]
                        + w1 * face_uvs[:, 1, :]
                        + w2 * face_uvs[:, 2, :]
                    )  # (F, 2)
                    # UV convention: u→x (width), v→y (height, flipped)
                    px = np.clip((sample_uv[:, 0] * W).astype(np.int32), 0, W - 1)
                    py = np.clip(((1.0 - sample_uv[:, 1]) * H).astype(np.int32), 0, H - 1)
                    atlas[:, ri, ci, :] = tex_img[py, px, :]

            vprint(f"GLB texture atlas built with shape {atlas.shape}")
            return atlas, True
        except Exception as e:
            print(f"GLB UV texture extraction failed ({e}), falling back to vertex colors")

    # Fallback: vertex colors
    try:
        vc = np.array(visual.to_color().vertex_colors, dtype=np.float32)[:, :3] / 255.0
        faces = np.array(mesh.faces, dtype=np.int64)
        F = faces.shape[0]
        atlas = np.zeros((F, atlas_res, atlas_res, 3), dtype=np.float32)
        face_colors = vc[faces].mean(axis=1)  # (F, 3)
        atlas[:, :, :, :] = face_colors[:, None, None, :]
        vprint(f"GLB vertex color atlas built with shape {atlas.shape}")
        return atlas, True
    except Exception as e:
        print(f"GLB vertex color extraction failed ({e}), no texture")
        return None, False


def _render_and_visibility_batched(
    views,
    R_all,
    T_all,
    verts_t,
    faces_t,
    atlas_t,
    focal,
    princpt,
    image_size,
    flip_t,
    device,
    name_fn,
    directional_light=False,
    light_dirs=None,
    mask_fn=None,
    coord_dtype=torch.long,
    backend="pyrender",
):
    """Batched render + visible-vertex extraction shared by render.py (ambient) and
    tmpl_render.py (directional): ``_RENDER_BATCH`` views per GPU call, with the
    perspective projection + visibility test vectorised across each chunk.

    NOT bit-exact vs a one-view-at-a-time render: batching the pytorch3d rasteriser
    perturbs floats by ~3e-5 (ambient) / ~1e-3 (directional), flipping a
    silhouette-edge pixel on a few percent of views after the uint8 cast. The
    visible_vertices .npy are saved in ascending index order (not set() order);
    downstream consumers read membership only.

    Args:
        views:      ordered list of (elevation, azimuth) — aligned with the rows of R_all.
        R_all,T_all:(N,3,3),(N,3) stacked per-view camera transforms.
        verts_t:    (V,3) double mesh verts — ``.float()`` for raster, double for projection.
        faces_t:    (F,3) faces. atlas_t: (F,ar,ar,3) per-face texture atlas.
        focal,princpt: intrinsics, numpy (2,).
        image_size: (1,2) tensor [[H,W]]. flip_t: (1,3,3) axis-flip.
        name_fn:    (e,a) -> (png_path, npy_path).
        directional_light/light_dirs: when True, feed DirectionalLights a per-view (N,3) dir.
        mask_fn:    optional (e,a) -> path; when set, also saves the (depth>0) template mask.
        coord_dtype: int cast for projected pixel coords (int32 for render, long for tmpl).
    """
    N = len(views)
    H = int(image_size[0, 0].item())
    W = int(image_size[0, 1].item())

    verts_render = verts_t.float()            # (V,3) float32 for rasterisation
    verts_proj = verts_t.unsqueeze(0)         # (1,V,3) double for projection

    focal_render = torch.tensor(focal).float()
    princpt_render = torch.tensor(princpt).float()
    focal_proj = torch.tensor(focal, device=device).unsqueeze(0)
    princpt_proj = torch.tensor(princpt, device=device).unsqueeze(0)

    # proj_R = inv(R @ flip), inverted per view then stacked — batched linalg.inv
    # drifts in the last bit.
    proj_R_all = torch.cat(
        [torch.linalg.inv(R_all[i:i + 1] @ flip_t) for i in range(N)], dim=0
    )

    for start in tqdm(
        range(0, N, _RENDER_BATCH), desc="render batch", dynamic_ncols=True
    ):
        end = min(start + _RENDER_BATCH, N)
        B = end - start

        renderer = make_renderer(
            backend,
            # (B,2) so the renderer derives batch_size == B and builds B cameras.
            image_size=image_size[0:1].expand(B, -1).contiguous(),
            focal_length=focal_render,
            principal_point=princpt_render,
            rotation=R_all[start:end],
            translation=T_all[start:end],
            light_direction=(light_dirs[start:end] if directional_light else None),
            device=device,
            directional_light=directional_light,
        )

        faces_in = faces_t.to(device)
        if faces_in.dim() == 2:
            faces_in = faces_in.unsqueeze(0)
        faces_in = faces_in.expand(B, -1, -1).contiguous()

        rendered, depth = renderer.render(
            verts=verts_render.unsqueeze(0).expand(B, -1, -1).contiguous(),
            faces=faces_in,
            texture=atlas_t,
        )
        rgb = rendered[..., :3].cpu().numpy() * 255.0            # (B,H,W,3)
        masks = (
            (depth > 0).cpu().numpy().astype(np.uint8) * 255      # (B,H,W)
            if mask_fn is not None else None
        )

        # Vectorised perspective projection + visibility across the whole chunk.
        proj_verts, proj_depth = _perspective_projection(
            verts_proj.expand(B, -1, -1),
            translation=T_all[start:end].to(dtype=torch.double),
            focal_length=focal_proj.expand(B, -1),
            camera_center=princpt_proj.expand(B, -1),
            rotation=proj_R_all[start:end].to(dtype=torch.double),
        )
        x = proj_verts[..., 0].to(coord_dtype)                   # (B,V)
        y = proj_verts[..., 1].to(coord_dtype)
        z = proj_depth[..., 0]                                   # (B,V) double
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (z > 0)
        # depth[b, y, x] via a flat gather; clamp keeps invalid coords in-bounds
        # (they are masked out by `valid` anyway).
        flat_idx = y.clamp(0, H - 1).long() * W + x.clamp(0, W - 1).long()
        pix_d = torch.gather(depth.reshape(B, H * W), 1, flat_idx)
        if backend == "pyrender":
            # One-sided test: visible unless clearly BEHIND the rendered surface
            # (z >> pix_d). OpenGL depth is noisier than pytorch3d's analytic zbuf,
            # so a symmetric 1e-3 band wrongly drops front-surface verts; the
            # one-sided _VIS_EPS slack keeps them.
            match = valid & ((pix_d - z) > -_VIS_EPS)
        else:
            match = valid & (torch.abs(z - pix_d) < 0.001)

        # torch.nonzero over the (B,V) match returns (b, v) pairs in ascending order,
        # so splitting the vertex column by per-view counts yields each view's
        # ascending unique indices.
        counts = match.sum(dim=1).cpu().numpy()
        vert_ids = torch.nonzero(match, as_tuple=True)[1].cpu().numpy()
        per_view = np.split(vert_ids, np.cumsum(counts)[:-1])

        for j in range(B):
            e, a = views[start + j]
            png_path, npy_path = name_fn(e, a)
            _write_png(png_path, rgb[j])
            if mask_fn is not None:
                cv2.imwrite(mask_fn(e, a), masks[j])
            np.save(npy_path, per_view[j])


def _render_views_cam_rot(seq_dir, rotation_angles, device, resolution_multiplier=1.0,
                          backend="pyrender"):
    """
    Render 3D mesh views for all elevation/azimuth combinations and save
    rendered PNGs + visible_vertices .npy files into render_segment/renders/.
    """
    resolution_suffix = "_highres" if resolution_multiplier > 1.0 else ""
    save_path = os.path.join(seq_dir, "render_segment", "renders")
    os.makedirs(save_path, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No *.jpg found in {seq_dir}")
    img_path = img_paths[0]

    metadata_path = os.path.join(seq_dir, "metadata.npz")
    metadata = dict(np.load(metadata_path)) if os.path.exists(metadata_path) else None

    full_img = _load_img(img_path)

    pad = [
        abs((full_img.shape[0] // 2) * 2 - full_img.shape[0]),
        abs((full_img.shape[1] // 2) * 2 - full_img.shape[1]),
    ]

    for _ext in (".obj", ".glb"):
        _candidate = os.path.join(seq_dir, f"full_img_textured{_ext}")
        if os.path.exists(_candidate):
            h3d_obj_mesh_path = _candidate
            break
    else:
        raise FileNotFoundError(
            f"No full_img_textured.obj or full_img_textured.glb found in {seq_dir}"
        )
    vprint(f"[render] Loading mesh from {h3d_obj_mesh_path}")
    _glb = h3d_obj_mesh_path.endswith(".glb")
    h3d_obj_mesh = trimesh.load(
        h3d_obj_mesh_path,
        force="mesh" if _glb else None,
        process=False,
        maintain_order=True,
    )
    texture_atlas, mesh_has_texture = _load_mesh_texture(h3d_obj_mesh_path)

    verts_cam_3d = h3d_obj_mesh.vertices.copy()

    original_height = full_img.shape[0] + pad[0]
    original_width = full_img.shape[1] + pad[1]
    high_res_height = int(original_height * resolution_multiplier)
    high_res_width = int(original_width * resolution_multiplier)

    if metadata is not None:
        scaled_focal_length = metadata["focal_length"] * resolution_multiplier
        scaled_principal_point = metadata["principal_point"] * resolution_multiplier
    else:
        # In-the-wild fallback: standard focal heuristic, principal at image center.
        _f = 0.5 * (original_height + original_width)
        scaled_focal_length = np.array([_f, _f], dtype=np.float32) * resolution_multiplier
        scaled_principal_point = np.array(
            [original_width / 2.0, original_height / 2.0], dtype=np.float32
        ) * resolution_multiplier
        print(f"[render] No metadata.npz — default focal={_f:.1f}, principal=center")

    image_size = torch.tensor([[high_res_height, high_res_width]], device=device)

    centered_bounds_min = verts_cam_3d.min(axis=0)
    centered_bounds_max = verts_cam_3d.max(axis=0)
    max_extent = max(
        np.linalg.norm(centered_bounds_min), np.linalg.norm(centered_bounds_max)
    )
    safety_margin = 1.0
    distance_width = (
        max_extent * scaled_focal_length[0] * safety_margin
    ) / (high_res_width / 2.0)
    distance_height = (
        max_extent * scaled_focal_length[1] * safety_margin
    ) / (high_res_height / 2.0)
    distance = max(distance_width, distance_height) * 1.1

    vprint(
        f"Rendering at {high_res_width}x{high_res_height} "
        f"(original: {original_width}x{original_height})"
    )

    # Mesh + texture live on the GPU once; the axis-flip is view-invariant.
    verts_t = torch.tensor(verts_cam_3d, device=device)
    faces_t = torch.tensor(h3d_obj_mesh.faces, device=device)
    atlas_t = (
        torch.tensor(texture_atlas, device=device) if mesh_has_texture else None
    )
    flip_t = torch.tensor(
        [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]], device=device
    )

    # Enumerate the views once and stack their per-view camera transforms.
    views, R_list, T_list = [], [], []
    for elevation in rotation_angles:
        elevation_c = elevation - 180
        if elevation_c <= -90 or elevation_c >= 90:
            continue
        for azimuth in rotation_angles:
            R, T = look_at_view_transform(distance, elevation_c, azimuth, device=device)
            views.append((elevation_c, azimuth))
            R_list.append(R)
            T_list.append(T)

    def _name_fn(e, a):
        return (
            os.path.join(save_path, f"rend_img_obj_e{e}_a{a}{resolution_suffix}.png"),
            os.path.join(save_path, f"visible_vertices_e{e}_a{a}{resolution_suffix}.npy"),
        )

    _render_and_visibility_batched(
        views=views,
        R_all=torch.cat(R_list, dim=0),
        T_all=torch.cat(T_list, dim=0),
        verts_t=verts_t,
        faces_t=faces_t,
        atlas_t=atlas_t,
        focal=scaled_focal_length,
        princpt=scaled_principal_point,
        image_size=image_size,
        flip_t=flip_t,
        device=device,
        name_fn=_name_fn,
        directional_light=False,
        coord_dtype=torch.int32,
        backend=backend,
    )

    vprint(f"[render] Completed rendering for {os.path.basename(seq_dir)}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seq_dir: str, verbose: bool = False, use_pytorch3d: bool = False) -> bool:
    """
    Run multi-view rendering for one sequence.

    Args:
        seq_dir: Absolute path to sequence directory. Must contain *.jpg,
                 metadata.npz, and full_img_textured.obj.
        use_pytorch3d: render with the pytorch3d backend instead of the default
                 pyrender backend.

    Returns:
        True if the step ran, False if skipped.
    """
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)

    if not glob.glob(os.path.join(seq_dir, "*.jpg")):
        print(f"[render] Skipping — no *.jpg in {seq_dir}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = "pytorch3d" if use_pytorch3d else "pyrender"
    print(f"[render] Running on {seq_dir} (renderer={backend})")
    _render_views_cam_rot(
        seq_dir=seq_dir,
        rotation_angles=_DEFAULT_ROTATION_ANGLES,
        device=device,
        backend=backend,
    )

    renders_dir = os.path.join(seq_dir, "render_segment", "renders")
    print(f"[render] Done → {renders_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render mesh views for one sequence"
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    parser.add_argument("--pytorch3d", action="store_true",
                        help="Use the pytorch3d renderer instead of the default pyrender backend.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, verbose=args.verbose, use_pytorch3d=args.pytorch3d)

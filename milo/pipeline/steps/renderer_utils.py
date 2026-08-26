"""
pipeline/steps/renderer_utils.py

Shared rendering backends for the pipeline render steps (render.py, tmpl_render.py,
render_final.py). Two interchangeable renderers with an identical constructor +
``render(verts, faces, texture) -> (rendered_output, depth)`` contract:

  * ``_RendererMesh``  — pytorch3d rasteriser (the original path).
  * ``PyrenderMesh``   — pyrender / OpenGL rasteriser (the default).

``make_renderer(backend, ...)`` picks one by name so callers never name a class.

Output contract: ``rendered_output`` is a float tensor in ``[0, 1]`` whose first
three channels are RGB (pytorch3d yields RGBA ``(...,4)``, pyrender RGB ``(...,3)``;
every caller slices ``[..., :3]``); ``depth`` is a ``(B,H,W)`` (or ``(H,W)``
unbatched) float tensor holding camera-space z, 0 at background. The pyrender
camera reproduces the exact projection used by the visibility test (OpenCV
extrinsic ``inv(R @ flip)``, ``T``; intrinsics ``fx,fy,cx,cy``), and its linear
depth equals pytorch3d's ``zbuf``. The per-vertex visibility test is the one place
the backends differ — pytorch3d: symmetric ``|z - depth| < 0.001`` on its analytic
zbuf; pyrender: one-sided ``z - depth < 0.003`` on its noisier OpenGL depth,
intentionally more inclusive (see ``_render_and_visibility_batched``).
"""

import os
from typing import Optional

import numpy as np
import torch

# X→-X, Y→-Y, Z→Z: maps the pytorch3d camera axes (+X left, +Y up) to the OpenCV
# convention (+X right, +Y down) used by the pinhole projection / visibility test.
_FLIP = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Shared pinhole projection (used by render.py's visibility test)
# ---------------------------------------------------------------------------

def _perspective_projection(
    points: torch.Tensor,
    translation: torch.Tensor,
    focal_length: torch.Tensor,
    camera_center: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pinhole-project (B,N,3) camera/world points; returns ((B,N,2) pixel coords,
    (B,N,1) camera-space depth). Despite the annotation, the return is a tuple."""
    batch_size = points.shape[0]
    if rotation is None:
        rotation = (
            torch.eye(3, device=points.device, dtype=points.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center
    points = torch.einsum("bij,bkj->bki", rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    depth = points[:, :, [2]].clone()
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)
    return projected_points[:, :, :-1], depth


# ---------------------------------------------------------------------------
# pytorch3d backend (moved verbatim from render.py)
# ---------------------------------------------------------------------------

from pytorch3d.renderer import (
    AmbientLights,
    DirectionalLights,
    MeshRasterizer,
    MeshRendererWithFragments,
    PerspectiveCameras,
    RasterizationSettings,
    SoftPhongShader,
    TexturesAtlas,
)
from pytorch3d.structures import Meshes


class _RendererMesh:
    def __init__(
        self,
        image_size,
        focal_length=None,
        principal_point=None,
        rotation=None,
        translation=None,
        light_direction=None,
        device=None,
        directional_light=False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if isinstance(image_size, (list, tuple)):
            self.height, self.width = image_size
            self.image_size_tensor = torch.tensor(
                [[self.height, self.width]], device=self.device
            )
        else:
            self.image_size_tensor = (
                image_size.to(self.device)
                if isinstance(image_size, torch.Tensor)
                else torch.tensor(image_size, device=self.device)
            )
        if self.image_size_tensor.dim() == 1:
            self.image_size_tensor = self.image_size_tensor.unsqueeze(0)
        self.height = self.image_size_tensor[0, 0].item()
        self.width = self.image_size_tensor[0, 1].item()
        self.batch_size = self.image_size_tensor.size(0)

        self.R = (
            torch.eye(3, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1, -1)
        )
        if rotation is not None:
            R_tensor = (
                rotation.to(self.device)
                if isinstance(rotation, torch.Tensor)
                else torch.tensor(rotation, device=self.device).float()
            )
            if R_tensor.dim() == 2:
                R_tensor = R_tensor.unsqueeze(0).expand(self.batch_size, -1, -1)
            self.R = R_tensor

        self.T = torch.zeros(self.batch_size, 3, dtype=torch.float32, device=self.device)
        if translation is not None:
            T_tensor = (
                translation.to(self.device)
                if isinstance(translation, torch.Tensor)
                else torch.tensor(translation, device=self.device).float()
            )
            if T_tensor.dim() == 1:
                T_tensor = T_tensor.unsqueeze(0).expand(self.batch_size, -1)
            self.T = T_tensor

        if light_direction is not None:
            self.light_tensor = (
                light_direction.to(self.device)
                if isinstance(light_direction, torch.Tensor)
                else torch.tensor(light_direction, device=self.device).float()
            )
        else:
            self.light_tensor = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)

        focal_tensor = (
            focal_length.to(self.device)
            if isinstance(focal_length, torch.Tensor)
            else torch.tensor(focal_length, device=self.device).float()
        )
        if focal_tensor.dim() == 1:
            focal_tensor = focal_tensor.unsqueeze(0).expand(self.batch_size, -1)
        self.focal_length = focal_tensor

        princpt_tensor = (
            principal_point.to(self.device)
            if isinstance(principal_point, torch.Tensor)
            else torch.tensor(principal_point, device=self.device).float()
        )
        if princpt_tensor.dim() == 1:
            princpt_tensor = princpt_tensor.unsqueeze(0).expand(self.batch_size, -1)
        self.principal_point = princpt_tensor

        self.cameras = PerspectiveCameras(
            device=self.device,
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=self.R,
            T=self.T,
            in_ndc=False,
            image_size=self.image_size_tensor,
        ).to(self.device)

        raster_settings = RasterizationSettings(
            image_size=(int(self.height), int(self.width)),
            blur_radius=0.0,
            bin_size=0,
            perspective_correct=True,
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        ).to(self.device)
        if directional_light:
            # Camera-following directional light + ambient floor, for depth/form
            # on the flat-grey template render. Flat AmbientLights washes the grey
            # template out to a formless silhouette; this matches the reference GT
            # renderer (render_hodome_gt.py). Opt-in only — the textured LRM renders
            # keep flat ambient below so their albedo colors stay unshaded.
            self.lights = DirectionalLights(
                device=self.device,
                direction=self.light_tensor,
                ambient_color=((0.3, 0.3, 0.3),),
                diffuse_color=((0.7, 0.7, 0.7),),
                specular_color=((0.2, 0.2, 0.2),),
            )
        else:
            self.lights = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))
        self.shader = SoftPhongShader(
            cameras=self.cameras, lights=self.lights
        ).to(self.device)
        self.renderer = MeshRendererWithFragments(
            rasterizer=self.rasterizer, shader=self.shader
        ).to(self.device)

    def render(self, verts, faces, texture, scale=None):
        is_batched = verts.dim() > 2
        batch_size = verts.size(0) if is_batched else 1

        verts = (
            verts.to(self.device)
            if isinstance(verts, torch.Tensor)
            else torch.tensor(verts, device=self.device).float()
        )
        if not is_batched:
            verts = verts.unsqueeze(0)

        faces = (
            faces.to(self.device)
            if isinstance(faces, torch.Tensor)
            else torch.tensor(faces, device=self.device).long()
        )
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(batch_size, -1, -1)

        if scale is not None:
            scale_tensor = (
                scale.to(self.device)
                if isinstance(scale, torch.Tensor)
                else torch.tensor(scale, device=self.device).float()
            )
            if scale_tensor.dim() == 0:
                scale_tensor = scale_tensor.expand(batch_size)
            elif scale_tensor.dim() == 1 and len(scale_tensor) != batch_size:
                scale_tensor = scale_tensor[0].expand(batch_size)
            for i in range(batch_size):
                verts[i] = verts[i] * scale_tensor[i]

        # Expand a single per-face atlas across the mesh batch: every view shares the
        # same atlas values, so per-view shading is bit-identical.
        atlas = texture if texture.dim() == 5 else texture.unsqueeze(0)
        if atlas.size(0) == 1 and batch_size > 1:
            atlas = atlas.expand(batch_size, -1, -1, -1, -1).contiguous()
        textures = TexturesAtlas(atlas=atlas)
        mesh = Meshes(
            verts=verts.to(self.device),
            faces=faces.to(self.device),
            textures=textures.to(self.device),
        ).to(self.device)

        with torch.no_grad():
            rendered_output, fragments = self.renderer(mesh)

        zbuf = fragments.zbuf
        depth = zbuf[..., 0]
        valid_mask = depth != -1
        depth_clean = torch.where(valid_mask, depth, torch.zeros_like(depth))
        depth_clean = torch.where(
            torch.isfinite(depth_clean), depth_clean, torch.zeros_like(depth_clean)
        )

        if not is_batched:
            rendered_output = rendered_output[0]
            depth_clean = depth_clean[0]

        return rendered_output, depth_clean


# ---------------------------------------------------------------------------
# pyrender backend (default)
# ---------------------------------------------------------------------------
# EGL offscreen context (headless GPU) — must be set BEFORE `import pyrender`.
# Mirror run_vis.py / viewer.py: pick the EGL device from CUDA_VISIBLE_DEVICES so
# the GL context lands on the same physical GPU the step was assigned.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "EGL_DEVICE_ID" not in os.environ:
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["EGL_DEVICE_ID"] = _cvd.split(",")[0] if _cvd else "0"

import pyrender  # noqa: E402
import trimesh  # noqa: E402
from pyrender.constants import RenderFlags  # noqa: E402

# OpenCV camera (x right, y down, z forward) -> OpenGL/pyrender camera (x right,
# y up, z backward): flip Y and Z.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _cv_to_gl_pose(R_cv, t_cv):
    """OpenCV world->cam extrinsic (R_cv, t_cv) -> pyrender camera node pose
    (cam->world, OpenGL convention)."""
    world_to_cam = np.eye(4, dtype=np.float64)
    world_to_cam[:3, :3] = _CV_TO_GL @ R_cv
    world_to_cam[:3, 3] = _CV_TO_GL @ t_cv
    return np.linalg.inv(world_to_cam)


def _atlas_to_face_colors(texture):
    """Per-face atlas (F,ar,ar,3) or (B,F,ar,ar,3) in [0,1] -> (F,4) uint8 RGBA
    flat per-face color (mean over the atlas tile — closest match to pytorch3d's
    barycentrically-interpolated atlas)."""
    atlas = texture.detach().cpu().numpy() if isinstance(texture, torch.Tensor) else np.asarray(texture)
    if atlas.ndim == 5:  # (B,F,ar,ar,3) — same atlas per view, take the first
        atlas = atlas[0]
    face_rgb = atlas.reshape(atlas.shape[0], -1, 3).mean(axis=1)  # (F,3) in [0,1]
    face_rgba = np.concatenate(
        [np.clip(face_rgb, 0.0, 1.0), np.ones((face_rgb.shape[0], 1))], axis=1
    )
    return (face_rgba * 255.0).astype(np.uint8)


def _as_batched_np(x, batch_size, cols):
    """Normalize a per-view param to a (batch_size, cols) float64 numpy array,
    broadcasting a single (cols,) / (1,cols) row across the batch."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).reshape(-1, cols)
    if x.shape[0] == 1 and batch_size > 1:
        x = np.repeat(x, batch_size, axis=0)
    return x


class PyrenderMesh:
    """Drop-in pyrender replacement for ``_RendererMesh`` — identical constructor
    signature and ``render()`` contract. Renders one static mesh from B moving
    cameras and returns ``(rendered_output[B,H,W,3] float in [0,1], depth[B,H,W]
    float, 0 at background)`` torch tensors on ``device`` (unbatched inputs -> the
    leading B axis is squeezed, mirroring ``_RendererMesh``)."""

    def __init__(
        self,
        image_size,
        focal_length=None,
        principal_point=None,
        rotation=None,
        translation=None,
        light_direction=None,
        device=None,
        directional_light=False,
    ):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # image_size -> (H, W) ints
        if isinstance(image_size, (list, tuple)):
            H, W = int(image_size[0]), int(image_size[1])
            batch_size = 1
        else:
            ims = image_size.detach().cpu().numpy() if isinstance(image_size, torch.Tensor) else np.asarray(image_size)
            ims = ims.reshape(-1, 2)
            H, W = int(ims[0, 0]), int(ims[0, 1])
            batch_size = ims.shape[0]
        self.height, self.width = H, W

        # rotation (B,3,3), translation (B,3)
        R = rotation.detach().cpu().numpy() if isinstance(rotation, torch.Tensor) else np.asarray(rotation, dtype=np.float64)
        R = R.reshape(-1, 3, 3).astype(np.float64)
        batch_size = max(batch_size, R.shape[0])
        if R.shape[0] == 1 and batch_size > 1:
            R = np.repeat(R, batch_size, axis=0)
        self.R = R

        if translation is None:
            self.T = np.zeros((batch_size, 3), dtype=np.float64)
        else:
            self.T = _as_batched_np(translation, batch_size, 3)
        self.batch_size = batch_size

        self.focal_length = _as_batched_np(focal_length, batch_size, 2)
        self.principal_point = _as_batched_np(principal_point, batch_size, 2)

        self.directional_light = directional_light
        if light_direction is not None:
            self.light_dirs = _as_batched_np(light_direction, batch_size, 3)
        else:
            self.light_dirs = np.tile([0.0, 0.0, -1.0], (batch_size, 1))

    def _light_pose(self, direction):
        """4x4 pose whose -Z axis points along ``direction`` (a pyrender
        DirectionalLight emits along its local -Z)."""
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        z = -d  # local -Z should equal the travel direction
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-12)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, z
        return pose

    def render(self, verts, faces, texture, scale=None):
        is_batched = verts.dim() > 2 if isinstance(verts, torch.Tensor) else np.asarray(verts).ndim > 2
        B = self.batch_size

        v = verts.detach().cpu().numpy() if isinstance(verts, torch.Tensor) else np.asarray(verts)
        v = v.astype(np.float64)
        if v.ndim == 3:      # (B,V,3) — mesh identical per view; take the first
            v = v[0]
        if scale is not None:
            s = float(scale.reshape(-1)[0]) if isinstance(scale, torch.Tensor) else float(np.asarray(scale).reshape(-1)[0])
            v = v * s

        f = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)
        f = f.astype(np.int64)
        if f.ndim == 3:
            f = f[0]

        face_colors = _atlas_to_face_colors(texture)

        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        tm.visual.face_colors = face_colors
        # Flat per-face shading (pyrender can't combine face colors with smooth normals).
        # For the ambient render RenderFlags.FLAT ignores normals anyway; for the
        # directional render the per-face normals still shade a dense mesh into 3D form.
        mesh_node = pyrender.Mesh.from_trimesh(tm, smooth=False)
        # Force a matte (non-metallic) material: pyrender's default from_trimesh material
        # is fully metallic (metallicFactor=1) — no diffuse term — so a directional light
        # barely shades it and the meshes come out near-flat. metallicFactor=0 restores
        # Lambertian diffuse so form shows. (Ignored by the FLAT ambient path.)
        for _prim in mesh_node.primitives:
            if _prim.material is not None:
                _prim.material.metallicFactor = 0.0
                _prim.material.roughnessFactor = 1.0

        # White background (matches pytorch3d's SoftPhong white bg). The ambient
        # render is drawn UNLIT (RenderFlags.FLAT -> pure per-face albedo, like the
        # pytorch3d AmbientLights path). The directional render (template / final)
        # keeps a directional light + ambient floor so grey/flat meshes read as 3D.
        if self.directional_light:
            scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0, 1.0])
            flags = RenderFlags.NONE
        else:
            scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0, 1.0])
            flags = RenderFlags.FLAT
        scene.add(mesh_node)
        cam_node = scene.add(pyrender.PerspectiveCamera(yfov=1.0))  # replaced per view
        light_node = (
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))
            if self.directional_light else None
        )

        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        rgb_out = np.zeros((B, self.height, self.width, 3), dtype=np.float32)
        depth_out = np.zeros((B, self.height, self.width), dtype=np.float32)
        try:
            for b in range(B):
                R_cv = np.linalg.inv(self.R[b] @ _FLIP)
                t_cv = self.T[b]
                cam_pose = _cv_to_gl_pose(R_cv, t_cv)

                # Retighten near/far to this view's actual vertex-depth range every view:
                # OpenGL depth-buffer precision is set by how wide a range it spans, and
                # views differ enough (mesh isn't a sphere) that a fixed range hurts it.
                z = (R_cv @ v.T + t_cv[:, None])[2]      # (V,) camera-space z (OpenCV)
                z_pos = z[z > 1e-6]
                if z_pos.size:
                    zmin, zmax = float(z_pos.min()), float(z_pos.max())
                    span = max(zmax - zmin, 1e-6)
                    znear = max(zmin - span * 0.02, 1e-3)
                    zfar = zmax + span * 0.02
                else:
                    znear, zfar = 1e-3, 10.0

                fx, fy = self.focal_length[b]
                cx, cy = self.principal_point[b]
                cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)
                cam_node.camera = cam
                scene.set_pose(cam_node, cam_pose)
                if light_node is not None:
                    # Camera-following light. pyrender lights live in WORLD space, so the
                    # view-space light_direction (±T/‖T‖) can't be used verbatim (it gives
                    # near-uniform n·l — a flat render). Emit along the camera view axis
                    # instead: into the scene (front light, render_final) or toward the
                    # camera (back light, template), by the sign of light_direction vs T.
                    sign = 1.0 if np.dot(self.light_dirs[b], t_cv) >= 0 else -1.0
                    travel = sign * (-cam_pose[:3, 2])   # camera -Z (into scene) in world
                    scene.set_pose(light_node, self._light_pose(travel))

                color, depth = renderer.render(scene, flags=flags)
                rgb_out[b] = color[..., :3].astype(np.float32) / 255.0
                depth_out[b] = depth  # 0 at background, camera-space z elsewhere
        finally:
            renderer.delete()

        rendered_output = torch.from_numpy(rgb_out).to(self.device)
        depth_clean = torch.from_numpy(depth_out).to(self.device)
        if not is_batched:
            rendered_output = rendered_output[0]
            depth_clean = depth_clean[0]
        return rendered_output, depth_clean


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_renderer(backend="pyrender", **kwargs):
    """Build a renderer by backend name. Both backends take the identical
    constructor kwargs, so callers pass ``**kwargs`` through verbatim."""
    if backend == "pytorch3d":
        return _RendererMesh(**kwargs)
    if backend == "pyrender":
        return PyrenderMesh(**kwargs)
    raise ValueError(f"unknown renderer backend {backend!r} (expected 'pyrender' or 'pytorch3d')")

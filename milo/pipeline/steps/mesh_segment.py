"""
pipeline/steps/mesh_segment.py

Vertex classification of the LRM mesh using the segmenter masks (SAM 3 /
Grounded-SAM-2).

Reads rendered PNGs + visible_vertices .npy files from render_segment/renders/
and segmenter masks from render_segment/segments/, then classifies each mesh
vertex as human (1), object (2), or conflicted (3).

Saves: render_segment/vertex_classification_data_multiaxis_boundary_elev_azim_adaptive.npz
       render_segment/segmentation_colored_mesh.obj  (LRM mesh coloured by label:
       grey=bg, green=human, blue=object, red=conflicted)

Runs AFTER img_segment and BEFORE kp2d.

Standalone usage:
    python -m milo.pipeline.steps.mesh_segment --seq_dir /path/to/seq

Module usage:
    from milo.pipeline.steps.mesh_segment import run
    run(seq_dir="/path/to/seq")
"""

import argparse
import os
import glob

import cv2
import numpy as np
import torch
import trimesh
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from pytorch3d.renderer import look_at_view_transform

from milo.pipeline.steps._log import vprint, set_verbose
from milo.pipeline.steps.render import _load_img
from milo.pipeline.steps.renderer_utils import _perspective_projection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_ROTATION_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

_OUTPUT_FILENAME = (
    "vertex_classification_data_multiaxis_boundary_elev_azim_adaptive.npz"
)

SURFACE_DISTANCE_WEIGHT = 4.0


def _compute_viewpoint_quality(obj_visible, total_vert_count, seg_mask_human=None, seg_mask_object=None):
    visibility_ratio = len(obj_visible) / total_vert_count
    mask_quality = 0.0
    if seg_mask_human is not None:
        mask_quality += np.sum(seg_mask_human > 0) / seg_mask_human.size
    if seg_mask_object is not None:
        mask_quality += np.sum(seg_mask_object > 0) / seg_mask_object.size
    quality_score = 0.55 * visibility_ratio + 0.45 * mask_quality
    return np.clip(quality_score, 0.1, 1.0)


def _compute_boundary_distance_scores(seg_mask_human, seg_mask_object, max_distance_pixels=50):
    if seg_mask_human is None or seg_mask_object is None:
        return None, None

    height, width = seg_mask_human.shape
    human_binary = (seg_mask_human > 127).astype(bool)
    object_binary = (seg_mask_object > 127).astype(bool)

    overlap_region = human_binary & object_binary
    total_object_area = np.sum(object_binary)
    overlap_area = np.sum(overlap_region)
    overlap_ratio = overlap_area / total_object_area if total_object_area > 0 else 0
    leniency_factor = overlap_ratio

    kernels = [
        np.ones((3, 3), np.uint8),
        np.ones((5, 5), np.uint8),
        np.ones((7, 7), np.uint8),
    ]
    boundary_mask = np.zeros((height, width), dtype=bool)
    for i, kernel in enumerate(kernels):
        human_dilated = cv2.dilate(human_binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        object_dilated = cv2.dilate(object_binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary_intersection = human_dilated & object_dilated
        if i == 0:
            boundary_mask = boundary_intersection.copy()
        else:
            boundary_mask = boundary_mask | boundary_intersection
    boundary_mask = boundary_mask.astype(bool)

    if not np.any(boundary_mask):
        human_edges = cv2.Canny(human_binary.astype(np.uint8) * 255, 50, 150)
        object_edges = cv2.Canny(object_binary.astype(np.uint8) * 255, 50, 150)
        edge_kernel = np.ones((5, 5), np.uint8)
        human_edge_zone = cv2.dilate(human_edges, edge_kernel, iterations=2).astype(bool)
        object_edge_zone = cv2.dilate(object_edges, edge_kernel, iterations=2).astype(bool)
        boundary_mask = (human_edge_zone & object_edge_zone).astype(bool)

    if not np.any(boundary_mask):
        kernel_simple = np.ones((3, 3), np.uint8)
        human_dilated_simple = cv2.dilate(human_binary.astype(np.uint8), kernel_simple, iterations=1).astype(bool)
        object_dilated_simple = cv2.dilate(object_binary.astype(np.uint8), kernel_simple, iterations=1).astype(bool)
        boundary_mask = (
            (human_binary & object_dilated_simple) | (object_binary & human_dilated_simple)
        ).astype(bool)

    if not np.any(boundary_mask):
        return (
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )

    inverted_boundary = (~boundary_mask).astype(bool)
    distance_from_boundary = distance_transform_edt(inverted_boundary)

    human_distances = distance_from_boundary * human_binary.astype(np.float32)
    object_distances = distance_from_boundary * object_binary.astype(np.float32)

    human_distance_scores = np.zeros((height, width), dtype=np.float32)
    object_distance_scores = np.zeros((height, width), dtype=np.float32)

    max_overall_dist = max(np.max(human_distances), np.max(object_distances))
    if max_overall_dist > 0:
        human_distance_scores = np.clip(human_distances / max_distance_pixels, 0, 1)
        object_distance_scores = np.clip(object_distances / max_distance_pixels, 0, 1)

        human_distance_scores = np.power(human_distance_scores, 0.6)

        min_power, max_power = 0.3, 0.8
        object_power = max_power - (leniency_factor * (max_power - min_power))
        object_distance_scores = np.power(object_distance_scores, object_power)

        if leniency_factor > 0.3:
            base_boost = leniency_factor * 0.2
            object_distance_scores = np.clip(object_distance_scores + base_boost, 0, 1)

    return human_distance_scores, object_distance_scores


def _compute_weighted_boundary_score(distance_scores, x, y, radius=3):
    if distance_scores is None:
        return 0.0
    scores, weights = [], []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < distance_scores.shape[1] and 0 <= ny < distance_scores.shape[0]:
                dist = np.sqrt(dx * dx + dy * dy)
                w = np.exp(-dist / radius)
                scores.append(w * distance_scores[ny, nx])
                weights.append(w)
    if not weights:
        return 0.0
    return np.sum(scores) / np.sum(weights)


def _compute_weighted_mask_score(seg_mask, x, y, radius=2):
    if seg_mask is None:
        return 0.0
    scores, weights = [], []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < seg_mask.shape[1] and 0 <= ny < seg_mask.shape[0]:
                dist = np.sqrt(dx * dx + dy * dy)
                w = np.exp(-dist / radius)
                scores.append(w * seg_mask[ny, nx] / 255.0)
                weights.append(w)
    if not weights:
        return 0.0
    return np.sum(scores) / np.sum(weights)


def _gather_weighted_avg(value_img, xs, ys, radius, scalar_fn, divide_255):
    """Bit-exact vectorised evaluation of ``scalar_fn(value_img, x, y, radius)`` at
    every pixel ``(xs[i], ys[i])``.

    ``scalar_fn`` is :func:`_compute_weighted_mask_score` (``divide_255=True``) or
    :func:`_compute_weighted_boundary_score` (``divide_255=False``). Interior pixels
    (full ``(2*radius+1)**2`` patch in-bounds) are gathered into an ``(N, K)`` matrix
    in the same dx-outer/dy-inner order and reduced with ``np.sum(axis=1)`` — the
    same numpy pairwise summation as the scalar code, so float64 results match
    bit-for-bit. Border pixels fall back to ``scalar_fn`` verbatim.
    """
    H, W = value_img.shape
    n = len(xs)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    # Kernel offsets + weights in the exact dx-outer/dy-inner order of scalar_fn.
    dxs, dys, ws = [], [], []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            dxs.append(dx)
            dys.append(dy)
            ws.append(np.exp(-np.sqrt(dx * dx + dy * dy) / radius))
    ws = np.array(ws, dtype=np.float64)
    w_sum = np.sum(ws)  # == np.sum(weights) for an all-in-bounds patch

    interior = (xs >= radius) & (xs < W - radius) & (ys >= radius) & (ys < H - radius)

    if np.any(interior):
        ix = xs[interior]
        iy = ys[interior]
        K = len(ws)
        vals = np.empty((ix.shape[0], K), dtype=np.float64)
        for c in range(K):
            # Widen to float64 BEFORE multiplying: the scalar path promotes each
            # pixel to float64 for `w * v`, but `float64_scalar * float32_array`
            # demotes to float32. Explicit widening keeps the product bit-identical.
            nb = value_img[iy + dys[c], ix + dxs[c]].astype(np.float64)
            if divide_255:
                vals[:, c] = (ws[c] * nb) / 255.0
            else:
                vals[:, c] = ws[c] * nb
        out[interior] = np.sum(vals, axis=1) / w_sum

    for j in np.where(~interior)[0]:
        out[j] = scalar_fn(value_img, int(xs[j]), int(ys[j]), radius)
    return out


def _process_angle_vertices(
    verts_hom,
    rotation,
    translation,
    obj_visible,
    cam_param_obj,
    seg_mask_pad,
    seg_mask_obj_pad,
    img_height_padded,
    img_width_padded,
    human_distance_scores,
    object_distance_scores,
    vertex_human_viewpoint_count,
    vertex_object_viewpoint_count,
    vertex_human_viewpoint_int_count,
    vertex_object_viewpoint_int_count,
    vertices_in_human_mask,
    vertices_in_object_mask,
    resolution_multiplier,
    device,
    human_bias=None,
    object_bias=None,
    viewpoint_quality_score=1.0,
):
    focal_length = cam_param_obj["focal"] * resolution_multiplier
    principal_point = cam_param_obj["princpt"] * resolution_multiplier

    actual_img_height = int(img_height_padded * resolution_multiplier)
    actual_img_width = int(img_width_padded * resolution_multiplier)

    projected_vertices, depth = _perspective_projection(
        torch.tensor(verts_hom, device=device).unsqueeze(0),
        translation=translation.clone().detach().to(device=device, dtype=torch.double),
        focal_length=torch.tensor(focal_length, device=device).unsqueeze(0),
        camera_center=torch.tensor(principal_point, device=device).unsqueeze(0),
        rotation=rotation.clone().detach().to(device=device, dtype=torch.double),
    )

    projected_2d = projected_vertices[0].cpu().numpy()
    depth_np = depth[0].cpu().numpy().squeeze()

    bound_h = actual_img_height
    bound_w = actual_img_width
    if seg_mask_pad is not None:
        bound_h = min(bound_h, seg_mask_pad.shape[0])
        bound_w = min(bound_w, seg_mask_pad.shape[1])
    if seg_mask_obj_pad is not None:
        bound_h = min(bound_h, seg_mask_obj_pad.shape[0])
        bound_w = min(bound_w, seg_mask_obj_pad.shape[1])

    valid_x = (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < bound_w)
    valid_y = (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < bound_h)
    valid_depth = depth_np > 0
    valid_projection = valid_x & valid_y & valid_depth

    obj_visible_verts = np.zeros_like(valid_projection)
    obj_visible_verts[obj_visible] = 1
    obj_visible_verts = obj_visible_verts.astype(bool)
    visible_indices = np.where(obj_visible_verts & valid_projection)[0]

    boundary_scoring_available = (
        human_distance_scores is not None and object_distance_scores is not None
    )

    visible_human_vertices = set()
    visible_object_vertices = set()

    if len(visible_indices) == 0:
        return visible_human_vertices, visible_object_vertices

    # Integer pixel coords of every visible vertex (astype truncation == int() on
    # the >=0 values guaranteed by valid_projection); the vectorised gather is
    # bit-exact vs the per-vertex scalar path — see _gather_weighted_avg.
    xs = projected_2d[visible_indices, 0].astype(np.int64)
    ys = projected_2d[visible_indices, 1].astype(np.int64)

    # ---- Human segmentation ----
    if seg_mask_pad is not None:
        hum_hit = seg_mask_pad[ys, xs] > 0
        if np.any(hum_hit):
            h_idx = visible_indices[hum_hit]
            hx = xs[hum_hit]
            hy = ys[hum_hit]
            mask_confidence = _gather_weighted_avg(
                seg_mask_pad, hx, hy,
                max(2, int(2 * resolution_multiplier)),
                _compute_weighted_mask_score, divide_255=True,
            )
            if boundary_scoring_available:
                boundary_confidence = _gather_weighted_avg(
                    human_distance_scores, hx, hy,
                    max(3, int(3 * resolution_multiplier)),
                    _compute_weighted_boundary_score, divide_255=False,
                )
                keep = boundary_confidence > 0.05
                combined_confidence = mask_confidence * 0.75 + 0.25 * boundary_confidence
                weight = viewpoint_quality_score * combined_confidence
                if human_bias is not None:
                    weight = weight + SURFACE_DISTANCE_WEIGHT * human_bias[h_idx]
                k_idx = h_idx[keep]
                vertex_human_viewpoint_count[k_idx] += weight[keep]
                vertex_human_viewpoint_int_count[k_idx] += 1
                vertices_in_human_mask.update(int(v) for v in k_idx)
                visible_human_vertices.update(int(v) for v in k_idx)
            else:
                weight = viewpoint_quality_score * mask_confidence
                if human_bias is not None:
                    weight = weight + SURFACE_DISTANCE_WEIGHT * human_bias[h_idx]
                vertex_human_viewpoint_count[h_idx] += weight
                vertex_human_viewpoint_int_count[h_idx] += 1
                vertices_in_human_mask.update(int(v) for v in h_idx)
                visible_human_vertices.update(int(v) for v in h_idx)

    # ---- Object segmentation ----
    if seg_mask_obj_pad is not None:
        obj_hit = seg_mask_obj_pad[ys, xs] > 0
        if np.any(obj_hit):
            o_idx = visible_indices[obj_hit]
            ox = xs[obj_hit]
            oy = ys[obj_hit]
            mask_confidence = seg_mask_obj_pad[oy, ox] / 255.0
            if boundary_scoring_available:
                boundary_confidence = _gather_weighted_avg(
                    object_distance_scores, ox, oy, 3,
                    _compute_weighted_boundary_score, divide_255=False,
                )
                keep = boundary_confidence > 0.05
                combined_confidence = mask_confidence * 0.9 + 0.1 * boundary_confidence
                weight = viewpoint_quality_score * combined_confidence
                if object_bias is not None:
                    weight = weight + SURFACE_DISTANCE_WEIGHT * object_bias[o_idx]
                k_idx = o_idx[keep]
                vertex_object_viewpoint_count[k_idx] += weight[keep]
                vertex_object_viewpoint_int_count[k_idx] += 1
                vertices_in_object_mask.update(int(v) for v in k_idx)
                visible_object_vertices.update(int(v) for v in k_idx)
            else:
                weight = viewpoint_quality_score * mask_confidence
                if object_bias is not None:
                    weight = weight + SURFACE_DISTANCE_WEIGHT * object_bias[o_idx]
                vertex_object_viewpoint_count[o_idx] += weight
                vertex_object_viewpoint_int_count[o_idx] += 1
                vertices_in_object_mask.update(int(v) for v in o_idx)
                visible_object_vertices.update(int(v) for v in o_idx)

    return visible_human_vertices, visible_object_vertices


def _compute_adaptive_thresholds(
    vertex_human_counts,
    vertex_object_counts,
    total_viewpoints_processed,
    viewpoint_quality_scores=None,
    mesh_vertices_count=None,
    surface_preclassification_used=False,
):
    human_counts = np.array(vertex_human_counts)
    object_counts = np.array(vertex_object_counts)
    total_vertices = len(human_counts)

    if mesh_vertices_count is None:
        mesh_vertices_count = total_vertices

    if viewpoint_quality_scores is not None and len(viewpoint_quality_scores) > 0:
        quality_scores = np.array(viewpoint_quality_scores)
        quality_mean = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        quality_cv = quality_std / quality_mean if quality_mean > 0 else 0
        effective_viewpoints = np.sum(quality_scores)
        quality_consistency = 1.0 / (1.0 + quality_cv)
    else:
        quality_mean = 1.0
        quality_consistency = 1.0
        effective_viewpoints = float(total_viewpoints_processed)

    combined_counts = human_counts + object_counts
    nonzero_counts = combined_counts[combined_counts > 0]

    if len(nonzero_counts) > 0:
        min_evidence_threshold = max(0.1, np.percentile(nonzero_counts, 5))
    else:
        min_evidence_threshold = 0.3

    meaningful_human_counts = human_counts[human_counts > min_evidence_threshold]
    meaningful_object_counts = object_counts[object_counts > min_evidence_threshold]

    if len(meaningful_human_counts) > 0 and len(meaningful_object_counts) > 0:
        all_meaningful_evidence = np.concatenate([meaningful_human_counts, meaningful_object_counts])
        base_min_count = np.percentile(all_meaningful_evidence, 10)
        human_p10 = np.percentile(meaningful_human_counts, 10) if len(meaningful_human_counts) > 5 else base_min_count
        object_p10 = np.percentile(meaningful_object_counts, 10) if len(meaningful_object_counts) > 5 else base_min_count
        base_min_count = min(base_min_count, human_p10, object_p10)
    elif len(meaningful_human_counts) > 0:
        base_min_count = np.percentile(meaningful_human_counts, 5)
    elif len(meaningful_object_counts) > 0:
        base_min_count = np.percentile(meaningful_object_counts, 5)
    else:
        base_min_count = min_evidence_threshold * 1.5

    quality_factor = 1.0
    if quality_mean > 1.1:
        quality_factor *= 0.7
    elif quality_mean > 0.9:
        quality_factor *= 0.85
    elif quality_mean < 0.7:
        quality_factor *= 1.15

    viewpoint_density_ratio = effective_viewpoints / total_viewpoints_processed
    if viewpoint_density_ratio > 1.2:
        quality_factor *= 0.8
    elif viewpoint_density_ratio > 1.0:
        quality_factor *= 0.9

    if surface_preclassification_used:
        quality_factor *= 0.75

    if mesh_vertices_count > 30000:
        quality_factor *= 0.8
    elif mesh_vertices_count > 10000:
        quality_factor *= 0.9
    elif mesh_vertices_count < 3000:
        quality_factor *= 1.05

    adaptive_min_count = base_min_count * quality_factor
    absolute_minimum = max(0.2, min_evidence_threshold * 0.8)
    adaptive_min_count = max(absolute_minimum, adaptive_min_count)
    max_reasonable = np.percentile(nonzero_counts, 30) if len(nonzero_counts) > 0 else 2.0
    adaptive_min_count = min(adaptive_min_count, max_reasonable)

    ratio_analysis_threshold = max(
        min_evidence_threshold * 0.5, adaptive_min_count * 0.4
    )
    ratios = []
    confidence_weights = []
    for i in range(total_vertices):
        h = human_counts[i]
        o = object_counts[i]
        t = h + o
        if t > ratio_analysis_threshold:
            ratios.append(h / t)
            confidence_weights.append(min(1.0, t / (effective_viewpoints * 0.1)))

    if len(ratios) > 10:
        ratios = np.array(ratios)
        weights = np.array(confidence_weights)
        ratio_std = np.std(ratios)
        weighted_mean = np.average(ratios, weights=weights)

        if ratio_std < 0.15:
            base_ratio_threshold = max(0.5, weighted_mean * 0.9)
        elif ratio_std > 0.35:
            human_dominant = ratios > 0.5
            if np.sum(human_dominant) > len(ratios) * 0.2:
                base_ratio_threshold = max(0.52, np.percentile(ratios[human_dominant], 15))
            else:
                base_ratio_threshold = max(0.6, np.percentile(ratios, 60))
        else:
            base_ratio_threshold = max(0.52, min(0.68, weighted_mean * 0.95))

        if quality_mean > 1.0 and quality_consistency > 0.7:
            base_ratio_threshold -= 0.05
        elif quality_mean > 1.2:
            base_ratio_threshold -= 0.08
        elif quality_mean < 0.8:
            base_ratio_threshold += 0.02

        adaptive_ratio_threshold = max(0.5, min(0.75, base_ratio_threshold))
    else:
        adaptive_ratio_threshold = 0.58

    # Validation and emergency relaxation
    def _simulate(min_c, ratio_t):
        n_h = n_o = 0
        for i in range(total_vertices):
            h = human_counts[i]
            o = object_counts[i]
            t = h + o
            if h >= min_c:
                if o <= min_evidence_threshold or h / t >= ratio_t:
                    n_h += 1
            if o >= min_c:
                if h <= min_evidence_threshold or o / t >= ratio_t:
                    n_o += 1
        return n_h, n_o

    sim_h, sim_o = _simulate(adaptive_min_count, adaptive_ratio_threshold)
    classification_rate = (sim_h + sim_o) / total_vertices

    if classification_rate < 0.15:
        adaptive_min_count *= 0.7
        adaptive_ratio_threshold = max(0.48, adaptive_ratio_threshold - 0.05)
    elif classification_rate > 0.90:
        adaptive_min_count *= 1.05
        adaptive_ratio_threshold = min(0.72, adaptive_ratio_threshold + 0.01)

    adaptive_min_count = max(0.1, min(adaptive_min_count, 3.0))
    adaptive_ratio_threshold = max(0.48, min(0.75, adaptive_ratio_threshold))

    analysis_info = {
        "quality_mean": quality_mean,
        "quality_consistency": quality_consistency,
        "effective_viewpoints": effective_viewpoints,
        "evidence_coverage_human": len(meaningful_human_counts) / total_vertices,
        "evidence_coverage_object": len(meaningful_object_counts) / total_vertices,
        "classification_rate": classification_rate,
        "base_min_count": base_min_count,
        "quality_adjustments_applied": quality_factor,
        "min_evidence_threshold": min_evidence_threshold,
    }

    return adaptive_min_count, adaptive_ratio_threshold, analysis_info


def _classify_vertices(seq_dir, rotation_angles, device, object_label="object", save_boundary_vis=True):
    """
    Core vertex classification using the segmenter masks + rendered visible_vertices.
    Saves the .npz classification file.
    """
    obj_name = object_label

    for _ext in (".obj", ".glb"):
        _candidate = os.path.join(seq_dir, f"full_img_textured{_ext}")
        if os.path.exists(_candidate):
            h3d_obj_path = _candidate
            break
    else:
        raise FileNotFoundError(
            f"No full_img_textured.obj or full_img_textured.glb found in {seq_dir}"
        )
    _glb = h3d_obj_path.endswith(".glb")
    h3d_obj_mesh = trimesh.load(h3d_obj_path, force="mesh" if _glb else None, process=False)
    h3d_verts = h3d_obj_mesh.vertices
    total_vert_count = len(h3d_verts)
    verts_cam_3d = h3d_verts.copy()

    img_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No *.jpg found in {seq_dir}")
    full_img = _load_img(img_paths[0])

    pad = [
        abs((full_img.shape[0] // 2) * 2 - full_img.shape[0]),
        abs((full_img.shape[1] // 2) * 2 - full_img.shape[1]),
    ]
    img_height_padded = full_img.shape[0] + pad[0]
    img_width_padded = full_img.shape[1] + pad[1]

    # camera intrinsics (metadata.npz optional → default focal for in-the-wild)
    metadata_path = os.path.join(seq_dir, "metadata.npz")
    if os.path.exists(metadata_path):
        metadata = dict(np.load(metadata_path))
        cam_param_obj = {
            "focal": metadata["focal_length"],
            "princpt": metadata["principal_point"],
        }
    else:
        _f = 0.5 * (img_height_padded + img_width_padded)
        cam_param_obj = {
            "focal": np.array([_f, _f], dtype=np.float32),
            "princpt": np.array([img_width_padded / 2.0, img_height_padded / 2.0], dtype=np.float32),
        }
        print(f"[mesh_segment] No metadata.npz — default focal={_f:.1f}, principal=center")

    centered_bounds_min = h3d_verts.min(axis=0)
    centered_bounds_max = h3d_verts.max(axis=0)
    max_extent = max(
        np.linalg.norm(centered_bounds_min), np.linalg.norm(centered_bounds_max)
    )
    safety_margin = 1.0
    distance = max(
        (max_extent * cam_param_obj["focal"][0] * safety_margin) / (img_width_padded / 2.0),
        (max_extent * cam_param_obj["focal"][1] * safety_margin) / (img_height_padded / 2.0),
    ) * 1.1  # match render.py: masks rendered at 1.1× distance

    vertex_human_viewpoint_count = np.zeros(total_vert_count, dtype=np.float32)
    vertex_object_viewpoint_count = np.zeros(total_vert_count, dtype=np.float32)
    vertex_human_viewpoint_int_count = np.zeros(total_vert_count, dtype=np.uint8)
    vertex_object_viewpoint_int_count = np.zeros(total_vert_count, dtype=np.uint8)
    vertices_in_human_mask = set()
    vertices_in_object_mask = set()
    viewpoint_qualities = []
    viewpoint_processed_count = 0
    angle_results = {}

    renders_dir = os.path.join(seq_dir, "render_segment", "renders")
    segments_dir = os.path.join(seq_dir, "render_segment", "segments")

    for elevation in tqdm(rotation_angles, desc="classify", dynamic_ncols=True):
        elevation_centered = elevation - 180
        if elevation_centered <= -90 or elevation_centered >= 90:
            continue

        for azimuth in rotation_angles:
            obj_visible_path = os.path.join(
                renders_dir,
                f"visible_vertices_e{elevation_centered}_a{azimuth}.npy",
            )
            if not os.path.exists(obj_visible_path):
                vprint(f"    Skipping - visible vertices file not found: {obj_visible_path}")
                continue

            render_path = os.path.join(
                renders_dir,
                f"rend_img_obj_e{elevation_centered}_a{azimuth}.png",
            )
            if not os.path.exists(render_path):
                vprint(f"    Skipping - render image not found: {render_path}")
                continue

            seg_mask_path = os.path.join(
                segments_dir, f"rend_img_obj_e{elevation_centered}_a{azimuth}.png"
            )
            if not os.path.exists(seg_mask_path):
                seg_mask_path = os.path.join(
                    segments_dir,
                    f"rend_img_obj_e{elevation_centered}_a{azimuth}_highres.png",
                )

            seg_mask_obj_path = os.path.join(
                segments_dir,
                f"rend_img_{obj_name}_e{elevation_centered}_a{azimuth}.png",
            )
            if not os.path.exists(seg_mask_obj_path):
                seg_mask_obj_path = os.path.join(
                    segments_dir,
                    f"rend_img_{obj_name}_e{elevation_centered}_a{azimuth}_highres.png",
                )

            flag_hum = os.path.exists(seg_mask_path)
            flag_obj = os.path.exists(seg_mask_obj_path)

            if not flag_hum and not flag_obj:
                continue

            seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE) if flag_hum else None
            seg_mask_obj = cv2.imread(seg_mask_obj_path, cv2.IMREAD_GRAYSCALE) if flag_obj else None

            seg_mask_pad = (
                np.pad(seg_mask, ((0, pad[0]), (0, pad[1])), mode="constant", constant_values=0)
                if seg_mask is not None
                else None
            )
            seg_mask_obj_pad = (
                np.pad(seg_mask_obj, ((0, pad[0]), (0, pad[1])), mode="constant", constant_values=0)
                if seg_mask_obj is not None
                else None
            )

            obj_visible = np.load(obj_visible_path)

            quality_score = _compute_viewpoint_quality(
                obj_visible, total_vert_count, seg_mask_pad, seg_mask_obj_pad
            )
            viewpoint_qualities.append(quality_score)
            viewpoint_processed_count += 1

            human_distance_scores, object_distance_scores = _compute_boundary_distance_scores(
                seg_mask_pad, seg_mask_obj_pad, max_distance_pixels=50
            )

            R, T = look_at_view_transform(distance, elevation_centered, azimuth, device=device)
            R = R @ torch.tensor(
                [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
                device=R.device,
            )
            R = torch.linalg.inv(R)

            visible_human_vertices, visible_object_vertices = _process_angle_vertices(
                verts_cam_3d,
                R,
                T,
                obj_visible,
                cam_param_obj,
                seg_mask_pad,
                seg_mask_obj_pad,
                img_height_padded,
                img_width_padded,
                human_distance_scores,
                object_distance_scores,
                vertex_human_viewpoint_count,
                vertex_object_viewpoint_count,
                vertex_human_viewpoint_int_count,
                vertex_object_viewpoint_int_count,
                vertices_in_human_mask,
                vertices_in_object_mask,
                resolution_multiplier=1.0,
                device=device,
                human_bias=None,
                object_bias=None,
                viewpoint_quality_score=quality_score,
            )

            vprint(
                f"    Found {len(visible_human_vertices)} human vertices, "
                f"{len(visible_object_vertices)} object vertices"
            )
            angle_results[f"e{elevation_centered}_a{azimuth}"] = {
                "visible_human_vertices": visible_human_vertices,
                "visible_object_vertices": visible_object_vertices,
            }

    # Final classification
    final_human_vertices = set()
    final_object_vertices = set()

    adaptive_min_count, adaptive_ratio_threshold, _ = _compute_adaptive_thresholds(
        vertex_human_viewpoint_count,
        vertex_object_viewpoint_count,
        viewpoint_processed_count,
        viewpoint_qualities,
        total_vert_count,
    )
    adaptive_ratio_threshold = max(0.65, adaptive_ratio_threshold * 0.85)

    for vertex_idx in vertices_in_human_mask:
        if vertex_human_viewpoint_count[vertex_idx] < adaptive_min_count:
            continue
        human_count = vertex_human_viewpoint_count[vertex_idx]
        object_count = vertex_object_viewpoint_count[vertex_idx]
        if object_count > 0:
            if human_count / (human_count + object_count) < adaptive_ratio_threshold:
                continue
        final_human_vertices.add(vertex_idx)

    for vertex_idx in vertices_in_object_mask:
        if vertex_object_viewpoint_count[vertex_idx] < adaptive_min_count:
            continue
        object_count = vertex_object_viewpoint_count[vertex_idx]
        human_count = vertex_human_viewpoint_count[vertex_idx]
        if human_count > 0:
            if object_count / (object_count + human_count) < adaptive_ratio_threshold:
                continue
        final_object_vertices.add(vertex_idx)

    for vertex_idx in vertices_in_human_mask & vertices_in_object_mask:
        max_human_count = np.max(vertex_human_viewpoint_int_count)
        max_object_count = np.max(vertex_object_viewpoint_int_count)
        if max_human_count > 0 and max_object_count > 0:
            hum_vert_ratio = vertex_human_viewpoint_int_count[vertex_idx] / max_human_count
            obj_vert_ratio = vertex_object_viewpoint_int_count[vertex_idx] / max_object_count
            if hum_vert_ratio < obj_vert_ratio:
                final_object_vertices.add(vertex_idx)
                final_human_vertices.discard(vertex_idx)
        elif max_object_count > 0:
            final_object_vertices.add(vertex_idx)
            final_human_vertices.discard(vertex_idx)
        else:
            if vertex_human_viewpoint_int_count[vertex_idx] < vertex_object_viewpoint_int_count[vertex_idx]:
                final_object_vertices.add(vertex_idx)
                final_human_vertices.discard(vertex_idx)

    conflicted_vertices = final_human_vertices & final_object_vertices

    total_angles_processed = len(angle_results)
    total_combinations = (
        len([e for e in rotation_angles if -90 < e - 180 < 90]) * len(rotation_angles)
    )

    print(
        f"[mesh_segment] {total_vert_count} verts | "
        f"human {len(final_human_vertices)} ({len(final_human_vertices)/total_vert_count*100:.1f}%) "
        f"object {len(final_object_vertices)} ({len(final_object_vertices)/total_vert_count*100:.1f}%) "
        f"conflict {len(conflicted_vertices)} | angles {total_angles_processed}/{total_combinations}"
    )

    if len(final_human_vertices) > 0 or len(final_object_vertices) > 0:
        vertex_labels = np.zeros(total_vert_count, dtype=np.int32)
        if len(final_human_vertices) > 0:
            vertex_labels[list(final_human_vertices)] = 1
        if len(final_object_vertices) > 0:
            vertex_labels[list(final_object_vertices)] = 2
        if len(conflicted_vertices) > 0:
            vertex_labels[list(conflicted_vertices)] = 3

        save_data = {
            "vertex_labels": vertex_labels,
            "final_human_vertex_indices": (
                np.array(list(final_human_vertices)) if final_human_vertices else np.array([])
            ),
            "final_object_vertex_indices": (
                np.array(list(final_object_vertices)) if final_object_vertices else np.array([])
            ),
            "conflicted_vertex_indices": (
                np.array(list(conflicted_vertices)) if conflicted_vertices else np.array([])
            ),
            "human_viewpoint_counts": vertex_human_viewpoint_count,
            "object_viewpoint_counts": vertex_object_viewpoint_count,
            "resolution_multiplier": 1.0,
            "min_viewpoint_count": adaptive_min_count,
            "ratio_threshold": adaptive_ratio_threshold,
            "total_angles_processed": total_angles_processed,
            "total_combinations_possible": total_combinations,
            "surface_preclassification_used": False,
            "surface_distance_weight": 0,
            "boundary_distance_scoring_used": True,
            "elevation_azimuth_rotations_used": True,
        }

        out_path = os.path.join(seq_dir, "render_segment", _OUTPUT_FILENAME)
        np.savez(out_path, **save_data)
        vprint(f"[mesh_segment] Saved → {out_path}")

        # Colored segmentation mesh — quick visual sanity check of the split
        # (matches the reference render_intercap_batched.py).
        seg_colors = np.full((total_vert_count, 4), [128, 128, 128, 255], dtype=np.uint8)
        seg_colors[vertex_labels == 1] = [0, 255, 0, 255]    # human  -> green
        seg_colors[vertex_labels == 2] = [0, 0, 255, 255]    # object -> blue
        seg_colors[vertex_labels == 3] = [255, 0, 0, 255]    # conflicted -> red
        seg_mesh = trimesh.Trimesh(
            vertices=h3d_obj_mesh.vertices.copy(),
            faces=h3d_obj_mesh.faces.copy(),
            process=False,
        )
        seg_mesh.visual.vertex_colors = seg_colors
        seg_out = os.path.join(seq_dir, "render_segment", "segmentation_colored_mesh.obj")
        seg_mesh.export(seg_out)
        vprint(f"[mesh_segment] Saved colored segmentation mesh → {seg_out}")
    else:
        print("[mesh_segment] WARNING: no vertices classified, output not saved.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(seq_dir: str, object_label: str = "object", verbose: bool = False) -> bool:
    """
    Classify LRM mesh vertices using the segmenter masks (SAM 3 / Grounded-SAM-2).

    Args:
        seq_dir: Absolute path to sequence directory.
        object_label: Short single-word object label used in the mask filenames (e.g. "chair").

    Returns:
        True if the step ran, False if skipped.
    """
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)

    renders_dir = os.path.join(seq_dir, "render_segment", "renders")
    if not os.path.isdir(renders_dir) or not glob.glob(
        os.path.join(renders_dir, "visible_vertices_e*_a*.npy")
    ):
        print(f"[mesh_segment] Skipping — renders not found. Run 'render' step first.")
        return False

    segments_dir = os.path.join(seq_dir, "render_segment", "segments")
    if not os.path.isdir(segments_dir) or not os.listdir(segments_dir):
        print(f"[mesh_segment] Skipping — segments dir empty or missing. Run 'img_segment' step first.")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[mesh_segment] Running on {seq_dir}")
    _classify_vertices(
        seq_dir=seq_dir,
        rotation_angles=_DEFAULT_ROTATION_ANGLES,
        device=device,
        object_label=object_label,
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify mesh vertices using the segmenter masks for one sequence"
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory")
    parser.add_argument(
        "--object", default="object",
        help="Short single-word object label used in the mask filenames (e.g. 'chair'). "
             "Default: 'object'.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, object_label=args.object, verbose=args.verbose)

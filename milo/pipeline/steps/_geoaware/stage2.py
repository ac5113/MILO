"""
pipeline/steps/_geoaware/stage2.py   (env: geo-aware)

Stage-2 correspondence pipeline: dense vertex matches between an LRM-generated
(H3D) textured object mesh and the untextured template ("GT") mesh.

For each sequence, loads the Stage-1 per-view MNN ranking
(``render_similarity_to_input.npz``, written by ``stage1.py``) to pick the
top-K H3D views, extracts fused SD+DINO features for all GT renders and those
H3D views, runs chunked FP16 bidirectional per-pixel cosine matching,
vertex-snaps via 2D KD-trees, applies a cosine-threshold + cycle-consistency
check, and keeps the best match per GT vertex across view pairs. Outputs
``final_combined_correspondences.npz`` plus visualizations under
``{seq_dir}/correspondences``.

Flag-gated matching/filtering refinements (adaptive threshold, per-pair
z-score calibration) default to reproducing the pre-refinement pipeline
byte-for-byte; see the ``__main__`` argparse block for the full flag surface.
"""

import os
import sys
import tempfile
import argparse
import contextlib
import datetime
import traceback
import time
import platform
import socket
# Cache/scratch root for SD/DINO model downloads + temp chunks. Configurable via
# MILO_GEO_CACHE; defaults to a repo-local _cache/geo (repo root is 4 levels up).
storage_path = os.environ.get('MILO_GEO_CACHE') or os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '_cache', 'geo'))
os.makedirs(storage_path, exist_ok=True)

# Point tempfile at the cache root instead of the (possibly small) system /tmp
tempfile.tempdir = storage_path

os.environ['TORCH_HOME'] = storage_path
os.environ['XDG_CACHE_HOME'] = storage_path
os.environ['HF_HOME'] = storage_path  # For HuggingFace models
os.environ['TMPDIR'] = storage_path   # For temporary download chunks

# Specific to iopath / detectron2 / odise
os.environ['FVCORE_CACHE'] = storage_path

import torch
from PIL import Image
import cv2
import torch.nn.functional as F
import numpy as np
from matplotlib import cm
import glob
import trimesh
from scipy.spatial import cKDTree

# MILO: resolve GeoAware-SC (model_utils/utils/preprocess_map) from the vendored
# submodule, and the co-located standalone utils_py3d (geo-aware has no pytorch3d).
import os as _os, sys as _sys
_GEO = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "third-party", "GeoAware-SC"))
for _p in (_GEO, _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_CKPT = _os.path.join(_GEO, "results_spair", "best_856.PTH")

from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed

from utils_py3d import look_at_view_transform

device = 'cuda'
set_seed(42)
num_patches = 60
sd_model = sd_aug = extractor_vit = None
aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
aggre_net.load_pretrained_weights(torch.load(_CKPT))

INTERCAP_MAPPING = ['trolley', 'skateboard', 'sports ball', 'umbrella', 'tennis_racquet', 'suitcase', 'chair', 'bottle', 'cup', 'stool']
_OBJ_NAME_OVERRIDE = None   # MILO: set by run() to bypass the InterCap folder-name mapping
_TEMPLATE_OVERRIDE = None   # MILO: set by run() to use an explicit template mesh

def perspective_projection_torch(points: torch.Tensor,
                                  translation: torch.Tensor,
                                  focal_length: torch.Tensor,
                                  camera_center: torch.Tensor,
                                  rotation: torch.Tensor) -> torch.Tensor:
    """GPU pinhole projection: returns ``(N, 2)`` pixel coords (no depth)."""
    pts = points @ rotation.T
    pts = pts + translation.unsqueeze(0)
    K = torch.zeros(3, 3, device=points.device, dtype=points.dtype)
    K[0, 0] = focal_length[0]
    K[1, 1] = focal_length[1]
    K[2, 2] = 1.0
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    pts_div = pts / pts[:, 2:3]
    projected = pts_div @ K.T
    return projected[:, :2]


def compute_bbox_from_mask(mask, tolerance=0.1):
    """Bounding box ``(x0, y0, x1, y1)`` of pixels > 128, padded by ``tolerance``
    (fraction of bbox size) and clamped to the image; ``None`` if mask is empty."""
    ys, xs = np.where(mask > 128)
    if len(xs) == 0:
        return None
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    
    pad_x = int(bbox_w * tolerance)
    pad_y = int(bbox_h * tolerance)
    
    h, w = mask.shape[:2]
    x_start = max(0, x_min - pad_x)
    y_start = max(0, y_min - pad_y)
    x_end = min(w, x_max + pad_x + 1)
    y_end = min(h, y_max + pad_y + 1)
    
    return (x_start, y_start, x_end, y_end)


def compute_resize_transform(orig_w, orig_h, target_res):
    """Return ``(scale_x, scale_y, offset_x, offset_y)`` mapping source-image
    pixel coords into the aspect-preserving, centered ``target_res`` square
    canvas produced by ``resize``."""
    if orig_h <= orig_w:
        new_w = target_res
        new_h = int(np.around(target_res * orig_h / orig_w))
        offset_x = 0
        offset_y = (target_res - new_h) // 2
    else:
        new_h = target_res
        new_w = int(np.around(target_res * orig_w / orig_h))
        offset_x = (target_res - new_w) // 2
        offset_y = 0

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    return scale_x, scale_y, offset_x, offset_y


def project_to_cropped_resized(projected, bbox, crop_scale_x, crop_scale_y,
                                crop_offset_x, crop_offset_y):
    """Map projected vertex coords from original render space into the
    cropped-then-resized image space."""
    out = projected.copy()
    x0, y0 = bbox[0], bbox[1]
    out[:, 0] = (out[:, 0] - x0) * crop_scale_x + crop_offset_x
    out[:, 1] = (out[:, 1] - y0) * crop_scale_y + crop_offset_y
    return out


def compute_rigid_alignment(source_points, target_points):
    """Umeyama similarity fit: returns ``(R, t, scale)`` mapping source to target.

    Filters non-finite rows, guards SVD failure and near-zero variance, and
    returns the identity transform (R=I, t=0, scale=1) when the problem is
    ill-posed.
    """
    A = source_points.copy()
    B = target_points.copy()
    n = A.shape[0]

    valid = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    A = A[valid]
    B = B[valid]
    n = A.shape[0]

    if n < 3:
        print("  WARNING: Too few valid points for rigid alignment, returning identity")
        return np.eye(3), np.zeros(3), 1.0
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    Am = A - centroid_A
    Bm = B - centroid_B

    H = np.dot(Am.T, Bm) / n

    if not np.isfinite(H).all():
        print("  WARNING: H matrix contains NaN/inf, returning identity")
        return np.eye(3), np.zeros(3), 1.0

    try:
        U, s, V = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        print("  WARNING: SVD did not converge, returning identity")
        return np.eye(3), np.zeros(3), 1.0

    R = np.dot(V.T, U.T)
    
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[-1] = -V[-1]
        R = np.dot(V.T, U.T)
    
    varP = np.var(A, axis=0).sum()
    if varP < 1e-12:
        print("  WARNING: Source points have near-zero variance, returning identity")
        return np.eye(3), np.zeros(3), 1.0

    scale = 1.0 / varP * np.sum(s)
    
    t = centroid_B - scale * np.dot(R, centroid_A)
    
    return R, t, scale


def apply_rigid_alignment(points, R, t, scale):
    """Apply a similarity transform ``(R, t, scale)`` (as returned by
    ``compute_rigid_alignment``) to a ``(N, 3)`` point array."""
    return scale * (points @ R.T) + t


# ==========================================================================
# Vectorized local scale computation using batch KD-tree
# ==========================================================================

def compute_local_scales_batch(vertices, sample_indices, k_neighbors=10):
    """Mean k-NN distance (local mesh resolution) per sampled vertex, via one
    KD-tree over all vertices.

    Returns ``({vertex_idx: scale}, default_scale)`` where ``default_scale``
    is the median scale (0.01 if no samples)."""
    if len(sample_indices) == 0:
        return {}, 0.01
    
    tree = cKDTree(vertices)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sample_points = vertices[sample_indices]
    
    # Query k+1 neighbors (first is self with distance 0)
    dists, _ = tree.query(sample_points, k=k_neighbors + 1)
    # Exclude self (column 0)
    mean_dists = dists[:, 1:].mean(axis=1)
    
    scales = dict(zip(sample_indices.tolist(), mean_dists.tolist()))
    default_scale = float(np.median(mean_dists)) if len(mean_dists) > 0 else 0.01
    
    return scales, default_scale


# ============================================================================
# VECTORIZED HELPERS
# ============================================================================

def filter_visible_vertices_vectorized(num_verts, projected_scaled, visible_set,
                                       mask_img, img_size):
    """Filter vertices by visibility set, image bounds, and mask > 128.

    Returns ``(valid_indices, valid_positions, px_int, py_int, valid_mask)``;
    the pixel arrays cover all ``num_verts`` vertices."""
    visible_arr = np.zeros(num_verts, dtype=bool)
    vis_list = list(visible_set)
    if len(vis_list) > 0:
        for vis_idx in vis_list:
            if 0 <= vis_idx < num_verts:
                visible_arr[vis_idx] = True

    px_int = np.round(projected_scaled[:, 0]).astype(np.int32)
    py_int = np.round(projected_scaled[:, 1]).astype(np.int32)

    in_bounds = (
        (px_int >= 0) & (px_int < img_size) &
        (py_int >= 0) & (py_int < img_size)
    )
    candidate = visible_arr & in_bounds

    # Mask check only for candidates (avoids out-of-bounds indexing)
    mask_check = np.zeros(num_verts, dtype=bool)
    cand_idx = np.where(candidate)[0]
    if len(cand_idx) > 0:
        mask_check[cand_idx] = mask_img[py_int[cand_idx], px_int[cand_idx]] > 128

    valid_mask = candidate & mask_check
    valid_indices = np.where(valid_mask)[0]
    valid_positions = projected_scaled[valid_indices]

    return valid_indices, valid_positions, px_int, py_int, valid_mask


def compute_correspondences_vectorized(valid_src_indices, px_int_src, py_int_src,
                                       correspondence_map, confidence_map,
                                       target_tree, valid_tgt_indices,
                                       img_size, cosine_threshold,
                                       nn_distance_threshold=10.0,
                                       threshold_mode='fixed',
                                       adaptive_percentile=0.7,
                                       adaptive_floor=0.35):
    """
    Vectorized correspondence lookup.

    threshold_mode='fixed' (default) uses cosine_threshold as the scalar cutoff
    (legacy behavior). threshold_mode='adaptive' computes a per-call threshold
    as max(quantile(confs, adaptive_percentile), adaptive_floor).
    """
    if len(valid_src_indices) == 0 or len(valid_tgt_indices) == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.float32), 0)

    # Gather correspondence and confidence for all valid source vertices
    src_py = py_int_src[valid_src_indices]
    src_px = px_int_src[valid_src_indices]

    corr_py = correspondence_map[src_py, src_px, 0]
    corr_px = correspondence_map[src_py, src_px, 1]
    confs = confidence_map[src_py, src_px]

    # Select effective threshold: fixed (legacy) or adaptive per-call.
    if threshold_mode == 'adaptive' and confs.size > 0:
        effective_threshold = max(
            float(np.quantile(confs, adaptive_percentile)),
            float(adaptive_floor),
        )
    else:
        effective_threshold = cosine_threshold

    # Filter 1: confidence threshold
    above_thresh = confs >= effective_threshold
    skipped_by_threshold = int(np.sum(~above_thresh))

    # Filter 2: correspondence pixel in bounds
    corr_in_bounds = (
        (corr_px >= 0) & (corr_px < img_size) &
        (corr_py >= 0) & (corr_py < img_size)
    )
    
    keep = above_thresh & corr_in_bounds
    
    if not np.any(keep):
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.float32), skipped_by_threshold)

    kept_src_indices = valid_src_indices[keep]
    kept_corr_px = corr_px[keep].astype(np.float64)
    kept_corr_py = corr_py[keep].astype(np.float64)
    kept_confs = confs[keep]

    # Batch KD-tree query
    query_points = np.stack([kept_corr_px, kept_corr_py], axis=1)
    dists, nearest_tree_idx = target_tree.query(query_points)

    # Filter 3: distance threshold
    within_radius = dists < nn_distance_threshold
    
    src_matched = kept_src_indices[within_radius]
    tgt_matched = valid_tgt_indices[nearest_tree_idx[within_radius]]
    matched_confs = kept_confs[within_radius]

    return src_matched, tgt_matched, matched_confs, skipped_by_threshold


# ==========================================================================
# Fused forward+reverse correspondence with FP16 and larger chunks
# ==========================================================================

def compute_dense_correspondence_both_directions(feat_src_upsampled, feat_tgt_upsampled,
                                                  img_size, chunk_size=4096):
    """Dense per-pixel correspondence in BOTH directions from a single
    similarity-matrix computation per chunk (FP16, large chunks).

    Returns ``(fwd_corr_map, fwd_conf_map, rev_corr_map, rev_conf_map)``;
    corr maps are ``(H, W, 2)`` int32 in (row, col) order, conf maps are
    ``(H, W)`` float32 cosines.
    """
    with torch.no_grad():
        C = feat_src_upsampled.shape[1]
        H = feat_src_upsampled.shape[2]
        W = feat_src_upsampled.shape[3]
        
        feat_src_reshaped = feat_src_upsampled.view(C, H * W).permute(1, 0)
        feat_tgt_reshaped = feat_tgt_upsampled.view(C, H * W).permute(1, 0)
        
        # FP16 for faster matmul and lower memory
        feat_src_norm = F.normalize(feat_src_reshaped, p=2, dim=1).half()
        feat_tgt_norm = F.normalize(feat_tgt_reshaped, p=2, dim=1).half()
        
        # Free FP32 intermediates immediately
        del feat_src_reshaped, feat_tgt_reshaped
        
        num_pixels = H * W

        # Allocate forward results on CPU (pinned for async transfer)
        fwd_indices = torch.empty(num_pixels, dtype=torch.int64, pin_memory=True)
        fwd_confs = torch.empty(num_pixels, dtype=torch.float32, pin_memory=True)
        
        # Allocate reverse running max on GPU for accumulation
        rev_best_vals = torch.full((num_pixels,), -float('inf'), device='cuda', dtype=torch.float16)
        rev_best_indices = torch.zeros(num_pixels, dtype=torch.int64, device='cuda')

        # Pre-transpose target features once (reused every chunk)
        feat_tgt_norm_t = feat_tgt_norm.t().contiguous()

        for i in range(0, num_pixels, chunk_size):
            end_i = min(i + chunk_size, num_pixels)
            # sim shape: (chunk, num_pixels) = src_chunk @ tgt_all.T
            sim = torch.mm(feat_src_norm[i:end_i], feat_tgt_norm_t)
            
            # Forward: row-wise max (src -> tgt)
            fwd_max_vals, fwd_max_idx = sim.max(dim=1)
            fwd_indices[i:end_i] = fwd_max_idx.cpu()
            fwd_confs[i:end_i] = fwd_max_vals.float().cpu()
            
            # Reverse: column-wise max (tgt -> src), accumulated across chunks
            chunk_col_max_vals, chunk_col_max_idx = sim.max(dim=0)
            chunk_col_max_idx_global = chunk_col_max_idx + i
            
            improved = chunk_col_max_vals > rev_best_vals
            rev_best_vals[improved] = chunk_col_max_vals[improved]
            rev_best_indices[improved] = chunk_col_max_idx_global[improved]
            
            del sim  # Free chunk memory immediately
        
        del feat_src_norm, feat_tgt_norm, feat_tgt_norm_t
        
        # Finalize forward
        fwd_all = fwd_indices.numpy()
        fwd_conf_all = fwd_confs.numpy()
        
        fwd_correspondence_map = np.stack([
            (fwd_all // W).reshape(H, W),
            (fwd_all % W).reshape(H, W)
        ], axis=-1).astype(np.int32)
        fwd_confidence_map = fwd_conf_all.reshape(H, W).astype(np.float32)

        # Finalize reverse
        rev_all = rev_best_indices.cpu().numpy()
        rev_conf_all = rev_best_vals.float().cpu().numpy()
        
        del rev_best_vals, rev_best_indices
        
        rev_correspondence_map = np.stack([
            (rev_all // W).reshape(H, W),
            (rev_all % W).reshape(H, W)
        ], axis=-1).astype(np.int32)
        rev_confidence_map = rev_conf_all.reshape(H, W).astype(np.float32)

    return (fwd_correspondence_map, fwd_confidence_map,
            rev_correspondence_map, rev_confidence_map)


# ============================================================================
# CYCLE CONSISTENCY
# ============================================================================

def check_cycle_consistency_with_neighborhoods(forward_map, reverse_map, 
                                               gt_verts, h3d_verts,
                                               gt_mesh_faces, h3d_mesh_faces,
                                               gt_scale_factor=3.0, 
                                               h3d_scale_factor=3.0):
    """Score forward matches by cycle consistency: 'strong' when the reverse
    match lands within ``gt_scale_factor`` x local mesh scale of the original
    GT vertex, else 'weak' (exponential distance penalty) or 'no_reverse'.

    Returns ``{gt_idx: {score, cycle_consistent, consistency_type, ...}}``.
    """
    print("  Computing cycle consistency with neighborhood checking...")

    print("    Computing local mesh scales (batch KD-tree)...")

    # Sample up to 100 vertices to estimate local scales; the rest fall back
    # to the median (default) scale.
    gt_sample_indices = list(forward_map.keys())[:min(100, len(forward_map))]
    gt_local_scales, gt_default_scale = compute_local_scales_batch(
        gt_verts, gt_sample_indices, k_neighbors=10
    )

    h3d_sample_indices = [v['h3d_idx'] for v in list(forward_map.values())[:100]]
    h3d_local_scales, h3d_default_scale = compute_local_scales_batch(
        h3d_verts, h3d_sample_indices, k_neighbors=10
    )
    
    print(f"    GT default scale: {gt_default_scale:.4f}, H3D default scale: {h3d_default_scale:.4f}")
    
    consistency_scores = {}
    
    for gt_idx, forward_data in forward_map.items():
        h3d_idx = forward_data['h3d_idx']
        forward_conf = forward_data.get('confidence', 0.5)
        
        if h3d_idx not in reverse_map:
            consistency_scores[gt_idx] = {
                'cycle_consistent': False,
                'consistency_type': 'no_reverse',
                'score': 0.3,
                'forward_confidence': forward_conf,
                'reverse_confidence': 0.0
            }
            continue
        
        reverse_gt_idx = reverse_map[h3d_idx]['gt_idx']
        reverse_conf = reverse_map[h3d_idx].get('confidence', 0.5)
        
        gt_local_scale = gt_local_scales.get(gt_idx, gt_default_scale)
        gt_neighborhood_radius = gt_local_scale * gt_scale_factor
        
        distance = np.linalg.norm(gt_verts[gt_idx] - gt_verts[reverse_gt_idx])
        
        if distance < gt_neighborhood_radius:
            proximity_score = 1.0 - (distance / gt_neighborhood_radius) * 0.5
            
            consistency_scores[gt_idx] = {
                'cycle_consistent': True,
                'consistency_type': 'strong',
                'score': proximity_score,
                'distance': distance,
                'neighborhood_radius': gt_neighborhood_radius,
                'forward_confidence': forward_conf,
                'reverse_confidence': reverse_conf,
                'reverse_gt_idx': reverse_gt_idx
            }
        else:
            distance_penalty = np.exp(-distance / gt_neighborhood_radius)
            
            consistency_scores[gt_idx] = {
                'cycle_consistent': False,
                'consistency_type': 'weak',
                'score': 0.3 * distance_penalty,
                'distance': distance,
                'neighborhood_radius': gt_neighborhood_radius,
                'forward_confidence': forward_conf,
                'reverse_confidence': reverse_conf,
                'reverse_gt_idx': reverse_gt_idx
            }
    
    if len(consistency_scores) > 0:
        strong = sum(1 for v in consistency_scores.values() if v['consistency_type'] == 'strong')
        weak = sum(1 for v in consistency_scores.values() if v['consistency_type'] == 'weak')
        no_reverse = sum(1 for v in consistency_scores.values() if v['consistency_type'] == 'no_reverse')
        
        print(f"    Cycle consistency: {strong} strong ({strong/len(consistency_scores)*100:.1f}%), "
              f"{weak} weak ({weak/len(consistency_scores)*100:.1f}%), "
              f"{no_reverse} no reverse ({no_reverse/len(consistency_scores)*100:.1f}%)")
    
    return consistency_scores


# ==========================================================================
# Batched mutual NN ratio: compute for multiple H3D views in one pass
# ==========================================================================

def compute_mutual_nn_ratio_batched(feat_gt_norm, h3d_data_list,
                                     gt_fg_indices, max_pixels=10000):
    """MNN ratio of multiple H3D views against one GT view; returns one ratio
    per view.

    GT foreground features (``feat_gt_norm``: (HW, C) normalized, on GPU) are
    subsampled to ``max_pixels`` once; each H3D view's 'feat_flat_norm' (CPU
    tensor, with 'fg_indices') is moved to GPU one at a time so all H3D
    features never coexist on GPU.
    """
    fg_gt = gt_fg_indices
    if len(fg_gt) > max_pixels:
        fg_gt = fg_gt[np.random.choice(len(fg_gt), max_pixels, replace=False)]
    
    if len(fg_gt) == 0:
        return [0.0] * len(h3d_data_list)
    
    feat_gt_fg = feat_gt_norm[fg_gt].half()  # (M, C) on GPU, FP16
    
    results = []
    for h3d_data in h3d_data_list:
        h3d_fg = h3d_data['fg_indices']
        if len(h3d_fg) > max_pixels:
            h3d_fg = h3d_fg[np.random.choice(len(h3d_fg), max_pixels, replace=False)]
        
        if len(h3d_fg) == 0:
            results.append(0.0)
            continue
        
        # Move this H3D view's features to GPU, compute, then free
        feat_h3d_fg = h3d_data['feat_flat_norm'][h3d_fg].half().cuda(non_blocking=True)
        
        sim = torch.mm(feat_gt_fg, feat_h3d_fg.t())
        forward_nn = torch.argmax(sim, dim=1)
        reverse_nn = torch.argmax(sim, dim=0)
        
        # Vectorized mutual check
        mutual_count = int(
            (reverse_nn[forward_nn] == torch.arange(len(forward_nn), device=forward_nn.device)).sum().item()
        )
        mutual_nn_ratio = mutual_count / len(forward_nn)
        results.append(mutual_nn_ratio)
        
        del feat_h3d_fg, sim, forward_nn, reverse_nn
    
    del feat_gt_fg
    return results


def generate_distinct_colors_rgba(n):
    """Return ``n`` visually distinct RGBA tuples (0-255) for matplotlib/3D viz."""
    from matplotlib.colors import hsv_to_rgb
    colors = []
    for i in range(n):
        hue = i / n
        sat = 0.8 + 0.2 * (i % 2)
        val = 0.9 + 0.1 * ((i + 1) % 2)
        rgb = hsv_to_rgb([hue, sat, val])
        colors.append([int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 255])
    return colors


def visualize_pair_correspondences(render_gt_img, render_h3d_img,
                                    projected_gt_scaled, projected_h3d_scaled,
                                    pair_correspondences, save_path,
                                    similarity_score=None,
                                    max_vis=100):
    """Write a side-by-side PNG of one (GT, H3D) render pair with up to
    ``max_vis`` correspondence lines colored by confidence (RdYlGn)."""
    import random
    
    gt_img = np.array(render_gt_img)
    h3d_img = np.array(render_h3d_img)
    h, w = gt_img.shape[:2]

    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
    h3d_bgr = cv2.cvtColor(h3d_img, cv2.COLOR_RGB2BGR)

    canvas = np.concatenate([gt_bgr, h3d_bgr], axis=1)

    all_corrs = list(pair_correspondences.items())
    if len(all_corrs) > max_vis:
        sampled_corrs = random.sample(all_corrs, max_vis)
    else:
        sampled_corrs = all_corrs

    n_corr = len(sampled_corrs)
    if n_corr == 0:
        cv2.imwrite(save_path, canvas)
        return

    confidences = np.array([data['confidence'] for _, data in sampled_corrs])
    min_conf = confidences.min()
    max_conf = confidences.max()
    
    if max_conf - min_conf < 1e-6:
        norm_confidences = np.ones_like(confidences)
    else:
        norm_confidences = (confidences - min_conf) / (max_conf - min_conf)

    colormap = cm.get_cmap('RdYlGn')

    for i, (gt_idx, data) in enumerate(sampled_corrs):
        h3d_idx = data['h3d_representative']
        
        norm_conf = norm_confidences[i]
        rgba = colormap(norm_conf)
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

        gt_px = int(np.round(projected_gt_scaled[gt_idx, 0]))
        gt_py = int(np.round(projected_gt_scaled[gt_idx, 1]))

        h3d_px = int(np.round(projected_h3d_scaled[h3d_idx, 0]))
        h3d_py = int(np.round(projected_h3d_scaled[h3d_idx, 1]))

        gt_px = np.clip(gt_px, 0, w - 1)
        gt_py = np.clip(gt_py, 0, h - 1)
        h3d_px = np.clip(h3d_px, 0, w - 1)
        h3d_py = np.clip(h3d_py, 0, h - 1)

        cv2.circle(canvas, (gt_px, gt_py), 3, color, -1)
        cv2.circle(canvas, (h3d_px + w, h3d_py), 3, color, -1)
        cv2.line(canvas, (gt_px, gt_py), (h3d_px + w, h3d_py), color, 1, cv2.LINE_AA)

    cv2.putText(canvas, 'GT', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, 'H3D', (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f'{n_corr} corrs', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if similarity_score is not None:
        cv2.putText(canvas, f'MNN Ratio: {similarity_score:.4f}', (10, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(canvas, f'Raw Cosine: {min_conf:.3f} - {max_conf:.3f}', (10, h - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, 'Green=High | Red=Low', (10, h - 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(save_path, canvas)


def create_correspondence_lines(gt_positions, h3d_positions, colors, radius=0.003):
    """Build trimesh cylinder primitives connecting paired 3D points for the
    combined 3D visualization. Zero-length pairs are skipped."""
    line_meshes = []
    for i, (gt_pos, h3d_pos) in enumerate(zip(gt_positions, h3d_positions)):
        direction = h3d_pos - gt_pos
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        
        try:
            cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=6)
            
            unit_dir = direction / length
            z_axis = np.array([0, 0, 1])
            
            if np.allclose(unit_dir, -z_axis, atol=1e-6):
                rot_mat = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            elif not np.allclose(unit_dir, z_axis, atol=1e-6):
                rot_axis = np.cross(z_axis, unit_dir)
                rot_norm = np.linalg.norm(rot_axis)
                if rot_norm > 1e-8:
                    rot_axis /= rot_norm
                    rot_angle = np.arccos(np.clip(np.dot(z_axis, unit_dir), -1, 1))
                    rot_mat = trimesh.transformations.rotation_matrix(rot_angle, rot_axis)
                else:
                    rot_mat = np.eye(4)
            else:
                rot_mat = np.eye(4)
            
            cylinder.apply_transform(rot_mat)
            midpoint = (gt_pos + h3d_pos) / 2
            cylinder.vertices += midpoint
            cylinder.visual.vertex_colors = np.tile(colors[i], (len(cylinder.vertices), 1))
            line_meshes.append(cylinder)
        except Exception as e:
            print(f"  Warning: Failed to create cylinder {i}: {e}")
            continue
    
    return line_meshes


def create_combined_visualization(gt_verts, gt_faces, h3d_verts, h3d_faces,
                                   final_correspondences, output_dir):
    """Export the two meshes plus correspondence-line cylinders to
    ``output_dir`` as a combined GLB, for inspection in a 3D viewer."""
    os.makedirs(output_dir, exist_ok=True)
    
    gt_indices = np.array(list(final_correspondences.keys()), dtype=np.int64)
    h3d_indices = np.array([v['h3d_idx'] for v in final_correspondences.values()], dtype=np.int64)
    
    source_pts = gt_verts[gt_indices]
    target_pts = h3d_verts[h3d_indices]
    
    R, t, scale = compute_rigid_alignment(source_pts, target_pts)
    aligned_gt_verts = apply_rigid_alignment(gt_verts, R, t, scale)
    
    print(f"  Rigid alignment: scale={scale:.4f}")
    
    aligned_source = apply_rigid_alignment(source_pts, R, t, scale)
    errors = np.linalg.norm(aligned_source - target_pts, axis=1)
    print(f"  Alignment error: mean={errors.mean():.4f}, median={np.median(errors):.4f}, max={errors.max():.4f}")
    
    n_corr = len(gt_indices)
    colors = generate_distinct_colors_rgba(n_corr)
    
    gt_colors = np.full((len(aligned_gt_verts), 4), [128, 128, 128, 255], dtype=np.uint8)
    for i, gt_idx in enumerate(gt_indices):
        gt_colors[gt_idx] = colors[i]
    
    gt_mesh = trimesh.Trimesh(vertices=aligned_gt_verts, faces=gt_faces, process=False)
    gt_mesh.visual.vertex_colors = gt_colors
    
    h3d_colors = np.full((len(h3d_verts), 4), [128, 128, 128, 255], dtype=np.uint8)
    for i, h3d_idx in enumerate(h3d_indices):
        h3d_colors[h3d_idx] = colors[i]
    
    h3d_mesh = trimesh.Trimesh(vertices=h3d_verts, faces=h3d_faces, process=False)
    h3d_mesh.visual.vertex_colors = h3d_colors
    
    gt_positions = aligned_gt_verts[gt_indices]
    h3d_positions = h3d_verts[h3d_indices]
    line_meshes = create_correspondence_lines(gt_positions, h3d_positions, colors)
    
    print(f"  Created {len(line_meshes)} correspondence lines")
    
    all_meshes = [gt_mesh, h3d_mesh] + line_meshes
    combined = trimesh.util.concatenate(all_meshes)
    
    combined_path = os.path.join(output_dir, 'combined_correspondences.obj')
    combined.export(combined_path)
    print(f"  Saved combined visualization to {combined_path}")
    
    gt_mesh.export(os.path.join(output_dir, 'gt_aligned.obj'))
    h3d_mesh.export(os.path.join(output_dir, 'h3d.obj'))
    
    np.savez(os.path.join(output_dir, 'alignment_params.npz'),
             R=R, t=t, scale=scale,
             mean_error=errors.mean(),
             median_error=np.median(errors),
             max_error=errors.max())
    
    return aligned_gt_verts, R, t, scale


def get_feature_cache_path(render_path, obj_name, seq_dir):
    """Map a render PNG path to its ``.pt`` feature-cache path under
    ``{seq_dir}/cached_features/{relpath-from-render-segment}``."""
    features_base_dir = os.path.join(seq_dir, 'cached_features')
    render_segment_dir = os.path.dirname(os.path.dirname(render_path))
    rel_path = os.path.relpath(render_path, render_segment_dir)
    cache_dir = os.path.join(features_base_dir, os.path.dirname(rel_path))
    filename = os.path.basename(render_path).replace('.png', '.pt')
    return os.path.join(cache_dir, filename)


def load_cached_features(render_path, obj_name, seq_dir, img_size):
    """Load fused SD+DINO features from disk, moved to GPU.

    Returns ``(features_raw, features_upsampled)`` or ``(None, None)`` when
    the cache file is missing or unreadable. Upsamples on the fly when the
    cache only holds the raw feature map.
    """
    cache_path = get_feature_cache_path(render_path, obj_name, seq_dir)
    if not os.path.exists(cache_path):
        return None, None
    try:
        save_dict = torch.load(cache_path, map_location='cpu')
        features = save_dict.get('features_raw', None)
        features_upsampled = save_dict.get('features_upsampled', None)
        if features is not None:
            features = features.cuda()
        if features_upsampled is not None:
            features_upsampled = features_upsampled.cuda()
        if features is not None and features_upsampled is None:
            features_upsampled = F.interpolate(features, size=(img_size, img_size), mode='bilinear', align_corners=False)
        return features, features_upsampled
    except Exception as e:
        print(f"      Warning: Failed to load cached features from {cache_path}: {e}")
        return None, None
        

@torch.no_grad()
def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    """Extract L2-normalized fused SD + DINOv2 descriptors for one image.

    Takes one of ``img`` (PIL) or ``img_path``. Reads pre-cached SD/DINO
    ``.pt`` shards when present, else runs the extractors. SD blocks s3/s4/s5
    and DINO tokens are concatenated and passed through ``aggre_net`` to a
    unified 768-d feature map at ``num_patches x num_patches``.
    """
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    del features_sd, features_dino
    
    desc = aggre_net(desc_gathered)
    del desc_gathered

    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


def is_sequence_fully_processed(seq_dir):
    """True if the flag-aware output dir already has final_combined_correspondences.npz."""
    output_path = os.path.join(seq_dir, _corres_dirname(), 'final_combined_correspondences.npz')
    # output_path = os.path.join(seq_dir, 'correspondences_new', 'final_combined_correspondences.npz')
    return os.path.exists(output_path)


def has_seg_maps(seq_dir):
    """True if the sequence has H3D segmentation maps (render_segment/segments);
    the GT segments dir is not checked."""
    seg_h3d_dir = os.path.join(seq_dir, 'render_segment', 'segments')
    has_h3d = os.path.isdir(seg_h3d_dir) and len(glob.glob(os.path.join(seg_h3d_dir, '**/*.png'), recursive=True)) > 0
    return has_h3d


def divide_batches(root_dir, num_batches):
    """Divide list of all sequences into num_batches, skipping completed ones."""
    img_list = list(glob.iglob(root_dir + '/**/*.jpg', recursive=False))
    all_sequence_dirs = sorted(list(set([os.path.dirname(img_path) for img_path in img_list])))

    print(f"Total sequences found: {len(all_sequence_dirs)}")

    no_seg_maps = [s for s in all_sequence_dirs if not has_seg_maps(s)]
    all_sequence_dirs = [s for s in all_sequence_dirs if has_seg_maps(s)]
    print(f"Skipping {len(no_seg_maps)} sequences with missing seg maps")

    completed = [s for s in all_sequence_dirs if is_sequence_fully_processed(s)]
    sequence_dirs = [s for s in all_sequence_dirs if not is_sequence_fully_processed(s)]

    print(f"Already completed: {len(completed)}, Remaining: {len(sequence_dirs)}")
    
    total_sequences = len(sequence_dirs)
    if total_sequences == 0:
        return []
        
    batch_size = (total_sequences + num_batches - 1) // num_batches
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_sequences)
        batches.append(sequence_dirs[start_idx:end_idx])
    
    return batches


# ============================================================================
# Preload all render data with GPU projection and pre-built KD-trees
# ============================================================================

def _consume_sd_aug_rng(sd_aug):
    """Consume exactly the np.random draw a skipped SD extraction would have made.

    detectron2's ResizeShortestEdge(sample_style='choice').get_transform calls
    np.random.choice(short_edge_length) once per aug application (the draw count
    is independent of image size), so applying the aug to a tiny dummy image
    keeps the global np.random stream bit-identical to the compute path.
    """
    from detectron2.data import transforms as T
    sd_aug(T.AugInput(np.zeros((8, 8, 3), dtype=np.uint8)))


def preload_view_data(renders_by_view, obj_name, seq_dir, img_size, verts,
                      cam_param_obj, distance, num_patches,
                      sd_model, sd_aug, aggre_net, extractor_vit,
                      use_cached_features, fallback_to_compute,
                      view_type='H3D', precomputed_feats=None):
    """Preload images, masks, features, visibility, projections, and 2D
    KD-trees for all views of one type ('H3D' or GT).

    Features are stored on CPU pinned memory and moved to GPU on demand
    during similarity/correspondence computation, keeping peak GPU memory
    low when many views are preloaded. Returns ``{(elev, azim): {...}}``.

    ``precomputed_feats``: optional ``{(elev, azim): raw feature tensor}`` (CPU
    fp32, from stage 1) reused instead of re-extracting; views missing from it
    fall through to the normal compute path.
    """
    preloaded = {}
    
    # Move vertices and camera params to GPU once
    verts_gpu = torch.tensor(verts, dtype=torch.float32, device='cuda')
    focal_gpu = torch.tensor(cam_param_obj['focal'], dtype=torch.float32, device='cuda')
    princpt_gpu = torch.tensor(cam_param_obj['princpt'], dtype=torch.float32, device='cuda')
    
    flip = torch.tensor([[-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, 1.0]], device='cuda')
    
    for (elevation, azimuth), render_path in sorted(renders_by_view.items()):
        # --- Load mask and compute bbox ---
        if view_type == 'H3D':
            mask_path = render_path.replace('renders', 'segments').replace('obj', obj_name.replace(' ', '_'))
        else:
            mask_path = render_path.replace('renders_gt', 'segments_gt').replace('obj', obj_name.replace(' ', '_'))
            # mask_path = render_path.replace('renders_gt_textured', 'segments_gt').replace('obj', obj_name.replace(' ', '_'))
        
        if not os.path.exists(mask_path):
            continue
        
        mask_orig = np.array(Image.open(mask_path).convert('L'))
        bbox = compute_bbox_from_mask(mask_orig, tolerance=0.1)
        if bbox is None:
            continue
        
        x0, y0, x1, y1 = bbox
        crop_w, crop_h = x1 - x0, y1 - y0
        
        render_pil = Image.open(render_path).convert('RGB')
        render_cropped = render_pil.crop((x0, y0, x1, y1))
        mask_cropped = Image.fromarray(mask_orig[y0:y1, x0:x1])
        
        render_resized = resize(render_cropped, target_res=img_size, resize=True, to_pil=True)
        mask_resized = np.array(resize(mask_cropped, target_res=img_size, resize=True, to_pil=True))
        
        sx, sy, ox, oy = compute_resize_transform(crop_w, crop_h, img_size)
        
        # --- Load visibility ---
        visible_path = render_path.replace('rend_img_obj', 'visible_vertices').replace('.png', '.npy')
        if not os.path.exists(visible_path):
            continue
        visible = set(np.load(visible_path).tolist())
        
        # --- Load or compute features (on GPU temporarily for processing) ---
        if precomputed_feats is not None and (elevation, azimuth) in precomputed_feats:
            # Stage-1 handoff: identical crop pipeline + deterministic extraction
            # (fixed shared_noise) make these features bit-identical to a fresh
            # compute. Burn the one np.random draw the skipped extraction would
            # have consumed so the downstream subsampling stream is unchanged.
            feat = precomputed_feats[(elevation, azimuth)].cuda()
            _consume_sd_aug_rng(sd_aug)
            feat_upsampled = F.interpolate(feat, size=(img_size, img_size), mode='bilinear', align_corners=False)
        elif use_cached_features:
            feat, feat_upsampled = load_cached_features(render_path, obj_name, seq_dir, img_size)
            if feat is None:
                if fallback_to_compute:
                    feat = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=render_resized)
                    feat_upsampled = F.interpolate(feat, size=(img_size, img_size), mode='bilinear', align_corners=False)
                else:
                    continue
        else:
            feat = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=render_resized)
            feat_upsampled = F.interpolate(feat, size=(img_size, img_size), mode='bilinear', align_corners=False)
        
        # --- Free raw features immediately (only upsampled is needed) ---
        del feat
        
        # --- GPU-based projection ---
        R_t, T_t = look_at_view_transform(distance, elevation, azimuth, device=device)
        R_t = R_t @ flip.unsqueeze(0)
        R_t = torch.linalg.inv(R_t)
        R_mat = R_t[0]
        T_vec = T_t[0]
        
        projected_gpu = perspective_projection_torch(
            verts_gpu, T_vec, focal_gpu, princpt_gpu, R_mat
        )
        projected_np = projected_gpu.cpu().numpy()
        
        projected_scaled = project_to_cropped_resized(
            projected_np, bbox, sx, sy, ox, oy
        )
        
        # --- Vectorized vertex filtering ---
        valid_indices, valid_positions, px_int, py_int, valid_mask = \
            filter_visible_vertices_vectorized(
                len(verts), projected_scaled, visible, mask_resized, img_size
            )
        
        # --- Pre-build KD-tree on valid 2D positions ---
        if len(valid_indices) > 0:
            kd_tree_2d = cKDTree(valid_positions)
        else:
            kd_tree_2d = None
        
        # --- Pre-normalize features for foreground (on GPU, then move to CPU) ---
        C, H, W = feat_upsampled.shape[1], feat_upsampled.shape[2], feat_upsampled.shape[3]
        foreground_mask = (mask_resized > 128).astype(np.float32)
        fg_indices = np.where(foreground_mask.flatten() > 0.5)[0]
        
        with torch.no_grad():
            feat_flat_norm = F.normalize(
                feat_upsampled.view(C, H * W).permute(1, 0), p=2, dim=1
            )
        
        # Store features on CPU pinned memory; moved back to GPU on demand.
        feat_upsampled_cpu = feat_upsampled.cpu().pin_memory()
        feat_flat_norm_cpu = feat_flat_norm.cpu().pin_memory()

        del feat_upsampled, feat_flat_norm
        
        preloaded[(elevation, azimuth)] = {
            'render': render_resized,
            'mask': mask_resized,
            'bbox': bbox,
            'scale': (sx, sy, ox, oy),
            'visible': visible,
            'path': render_path,
            'feat_upsampled': feat_upsampled_cpu,      # CPU pinned
            'feat_flat_norm': feat_flat_norm_cpu,        # CPU pinned
            'fg_indices': fg_indices,
            'projected_scaled': projected_scaled,
            'valid_indices': valid_indices,
            'valid_positions': valid_positions,
            'px_int': px_int,
            'py_int': py_int,
            'valid_mask': valid_mask,
            'kd_tree_2d': kd_tree_2d,
        }
    
    del verts_gpu, focal_gpu, princpt_gpu
    torch.cuda.empty_cache()

    return preloaded


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_sequence(img_path, sd_model, sd_aug, aggre_net, extractor_vit, save_pair_vis=True):
    """Per-sequence entry point. Sets up log-file teeing, then dispatches.

    Writes all stdout/stderr for this sequence to ``{output_dir}/run.log``
    (where ``{output_dir}`` is the flag-aware output subdirectory), including
    full tracebacks on failure. The console stream is unchanged.
    """
    seq_dir = os.path.dirname(img_path)
    output_dir = os.path.join(seq_dir, _corres_dirname())
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'run.log')

    with _tee_stdio(log_path):
        _log_run_header(img_path, log_path)
        seq_start = time.time()
        try:
            result = _process_sequence_body(img_path, sd_model, sd_aug, aggre_net, extractor_vit,
                                            save_pair_vis=save_pair_vis)
            elapsed = time.time() - seq_start
            print(f"\n[run.log] sequence finished in {elapsed:.1f}s "
                  f"(result={result}) at {datetime.datetime.utcnow().isoformat()}Z")
            return result
        except Exception:
            elapsed = time.time() - seq_start
            print(f"\n[run.log] sequence FAILED after {elapsed:.1f}s "
                  f"at {datetime.datetime.utcnow().isoformat()}Z")
            traceback.print_exc()
            raise


def _process_sequence_body(img_path, sd_model, sd_aug, aggre_net, extractor_vit,
                           h3d_features=None, save_pair_vis=True):
    """Compute and save dense GT<->H3D vertex correspondences for one sequence.

    Returns True on success, False when required inputs are missing or no
    correspondences survive filtering.

    ``h3d_features``: optional stage-1 feature handoff (see preload_view_data).
    ``save_pair_vis=False`` skips the per-pair PNGs (inspection-only output;
    the correspondence npz is unaffected).
    """
    folder_name = os.path.basename(os.path.dirname(img_path))
    obj_name = _OBJ_NAME_OVERRIDE
    if obj_name is None:
        try:
            obj_name = INTERCAP_MAPPING[(int)(folder_name.split('__')[1]) - 1]
        except IndexError:
            print(f"  Could not determine obj_name for {folder_name}, skipping...")
            return False
    seq_dir = os.path.dirname(img_path)

    print(f"\n{'='*80}")
    print(f"Processing: {img_path}")
    print(f"{'='*80}")

    # =========================================================================
    # Load top-K input-similar H3D views
    # =========================================================================
    similarity_path = os.path.join(seq_dir, 'correspondences', 'render_similarity_to_input.npz')
    # similarity_path = os.path.join(seq_dir, 'correspondences_new', 'render_similarity_to_input.npz')
    if not os.path.exists(similarity_path):
        print(f"  Missing render_similarity_to_input.npz at {similarity_path}, skipping...")
        print(f"  Please run the input-image similarity script first!")
        return False

    sim_data = np.load(similarity_path, allow_pickle=True)
    sim_types = sim_data['types']
    sim_elevations = sim_data['elevations'].astype(int)
    sim_azimuths = sim_data['azimuths'].astype(int)
    sim_mnn_ratios = sim_data['mnn_ratios']

    h3d_mask = (sim_types == 'h3d')
    h3d_elevations = sim_elevations[h3d_mask]
    h3d_azimuths = sim_azimuths[h3d_mask]
    h3d_mnn = sim_mnn_ratios[h3d_mask]

    top_k_count = min(TOP_K_INPUT_SIMILAR, len(h3d_elevations))
    top_h3d_views = set()
    for i in range(top_k_count):
        top_h3d_views.add((int(h3d_elevations[i]), int(h3d_azimuths[i])))

    print(f"  Top-{top_k_count} H3D views most similar to input image:")
    for i in range(top_k_count):
        print(f"    #{i+1}  elevation={h3d_elevations[i]:3d}  azimuth={h3d_azimuths[i]:3d}  MNN={h3d_mnn[i]:.4f}")

    # =========================================================================
    # Load meshes and metadata
    # =========================================================================
    renders_h3d_list = list(glob.iglob(os.path.dirname(img_path) + '/render_segment/renders/*.png', recursive=False))
    # renders_h3d_list = list(glob.iglob(os.path.dirname(img_path) + '/render_segment/renders_new/*.png', recursive=False))
    renders_gt_list = list(glob.iglob(os.path.dirname(img_path) + '/render_segment/renders_gt/*.png', recursive=False))
    # renders_gt_list = list(glob.iglob(os.path.dirname(img_path) + '/render_segment/renders_gt_textured/*.png', recursive=False))

    render_size = Image.open(renders_gt_list[0]).size if renders_gt_list else (1280, 720)

    # h3d_mesh_path = os.path.dirname(img_path) + '/full_img_textured.obj'
    h3d_mesh_path = os.path.dirname(img_path) + '/full_img_textured.glb'
    gt_mesh_path = _TEMPLATE_OVERRIDE if _TEMPLATE_OVERRIDE else (os.path.dirname(img_path) + '/obj_gt.obj')

    if not os.path.exists(h3d_mesh_path) or not os.path.exists(gt_mesh_path):
        print(f"  Missing mesh files, skipping...")
        return False

    h3d_mesh = trimesh.load(h3d_mesh_path, process=False, force='mesh')
    h3d_verts = np.array(h3d_mesh.vertices)
    h3d_faces = np.array(h3d_mesh.faces)

    gt_mesh = trimesh.load(gt_mesh_path, process=False)
    gt_verts = np.array(gt_mesh.vertices)
    gt_faces = np.array(gt_mesh.faces)

    centroid = gt_verts.mean(axis=0)
    gt_verts_centered = gt_verts - centroid
    gt_verts = gt_verts_centered

    rot_x_180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
    gt_verts = gt_verts @ rot_x_180.T

    metadata_path = os.path.dirname(img_path) + "/metadata.npz"
    if not os.path.exists(metadata_path):
        print(f"  Missing metadata, skipping...")
        return False
        
    metadata = dict(np.load(metadata_path))

    cam_param_obj = {
        'focal': metadata["focal_length"],
        'princpt': metadata["principal_point"]
    }

    distance_h3d = 3.5

    centered_bounds_gt_min = gt_verts.min(axis=0)
    centered_bounds_gt_max = gt_verts.max(axis=0)
    max_gt_extent = max(
        np.linalg.norm(centered_bounds_gt_min),
        np.linalg.norm(centered_bounds_gt_max)
    )
    safety_margin = 1.0

    distance_gt_width = (max_gt_extent * cam_param_obj['focal'][0] * safety_margin) / (render_size[0] / 2.0)
    distance_gt_height = (max_gt_extent * cam_param_obj['focal'][1] * safety_margin) / (render_size[1] / 2.0)
    distance_gt = max(distance_gt_width, distance_gt_height)

    pair_vis_dir = os.path.join(os.path.dirname(img_path), _corres_dirname(), 'pair_visualizations')
    # pair_vis_dir = os.path.join(os.path.dirname(img_path), 'correspondences_new', 'pair_visualizations')
    os.makedirs(pair_vis_dir, exist_ok=True)

    # =========================================================================
    # Group renders by (elevation, azimuth)
    # =========================================================================
    h3d_renders_by_view = {}
    gt_renders_by_view = {}
    
    for render_h3d_path in sorted(renders_h3d_list):
        fname = os.path.basename(render_h3d_path)
        elevation = int(fname.split('_e')[1].split('_a')[0])
        azimuth = int(fname.split('_a')[1].split('.png')[0])
        key = (elevation, azimuth)
        if key in top_h3d_views:
            h3d_renders_by_view[key] = render_h3d_path
    
    for render_gt_path in sorted(renders_gt_list):
        fname = os.path.basename(render_gt_path)
        elevation = int(fname.split('_e')[1].split('_a')[0])
        azimuth = int(fname.split('_a')[1].split('.png')[0])
        gt_renders_by_view[(elevation, azimuth)] = render_gt_path
    
    print(f"\n  Found {len(gt_renders_by_view)} GT views and "
          f"{len(h3d_renders_by_view)} H3D views (top-{TOP_K_INPUT_SIMILAR} input-similar)")
    
    if len(h3d_renders_by_view) == 0:
        print(f"  No matching H3D views found after filtering, skipping...")
        return False
    
    # =========================================================================
    # Preload ALL H3D view data ONCE (features stored on CPU pinned memory)
    # =========================================================================
    print(f"\n  Preloading all {len(h3d_renders_by_view)} H3D views...")
    _t_preload = time.time()
    h3d_preloaded = preload_view_data(
        h3d_renders_by_view, obj_name, seq_dir, img_size, h3d_verts,
        cam_param_obj, distance_h3d, num_patches,
        sd_model, sd_aug, aggre_net, extractor_vit,
        USE_CACHED_FEATURES, FALLBACK_TO_COMPUTE,
        view_type='H3D', precomputed_feats=h3d_features
    )
    print(f"  Successfully preloaded {len(h3d_preloaded)}/{len(h3d_renders_by_view)} H3D views "
          f"[TIMING] preload H3D views: {time.time() - _t_preload:.2f}s")
    
    if len(h3d_preloaded) == 0:
        print(f"  No valid H3D views after preloading, skipping...")
        return False

    # =========================================================================
    # Accumulate reverse correspondences during the forward pass
    # =========================================================================
    all_pair_correspondences = {}
    reverse_correspondences = {}

    # =========================================================================
    # Precompute GPU tensors for GT projection
    # =========================================================================
    gt_verts_gpu = torch.tensor(gt_verts, dtype=torch.float32, device='cuda')
    focal_gpu = torch.tensor(cam_param_obj['focal'], dtype=torch.float32, device='cuda')
    princpt_gpu = torch.tensor(cam_param_obj['princpt'], dtype=torch.float32, device='cuda')
    flip = torch.tensor([[-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, 1.0]], device='cuda')
    
    # =========================================================================
    # FORWARD + REVERSE CORRESPONDENCE COMPUTATION
    # =========================================================================
    
    # Pre-build sorted list of H3D data for batched MNN
    h3d_sorted_keys = sorted(h3d_preloaded.keys())
    h3d_data_list_for_mnn = [h3d_preloaded[k] for k in h3d_sorted_keys]
    
    _t_match_loop = time.time()
    for gt_view_idx, ((elevation_gt, azimuth_gt), render_gt_path) in enumerate(sorted(gt_renders_by_view.items())):
        print(f"\n  Processing GT view {gt_view_idx+1}/{len(gt_renders_by_view)}: "
              f"elevation {elevation_gt}, azimuth {azimuth_gt} deg")
        
        # --- Load GT render, mask, features ---
        render_gt_mask_path = render_gt_path.replace('renders_gt', 'segments_gt').replace('obj', obj_name.replace(' ', '_'))
        # render_gt_mask_path = render_gt_path.replace('renders_gt_textured', 'segments_gt').replace('obj', obj_name.replace(' ', '_'))
        if not os.path.exists(render_gt_mask_path):
            print(f"    Missing GT mask, skipping...")
            continue
        
        gt_mask_orig = np.array(Image.open(render_gt_mask_path).convert('L'))
        gt_bbox = compute_bbox_from_mask(gt_mask_orig, tolerance=0.1)
        
        if gt_bbox is None:
            print(f"    Empty GT mask, skipping...")
            continue
        
        gt_x0, gt_y0, gt_x1, gt_y1 = gt_bbox
        gt_crop_w, gt_crop_h = gt_x1 - gt_x0, gt_y1 - gt_y0
        
        render_gt_pil = Image.open(render_gt_path).convert('RGB')
        render_gt_cropped = render_gt_pil.crop((gt_x0, gt_y0, gt_x1, gt_y1))
        gt_mask_cropped = Image.fromarray(gt_mask_orig[gt_y0:gt_y1, gt_x0:gt_x1])
        
        render_gt = resize(render_gt_cropped, target_res=img_size, resize=True, to_pil=True)
        render_gt_mask = np.array(resize(gt_mask_cropped, target_res=img_size, resize=True, to_pil=True))
        
        gt_sx, gt_sy, gt_ox, gt_oy = compute_resize_transform(gt_crop_w, gt_crop_h, img_size)
        
        # Load GT visibility
        gt_visible_path = render_gt_path.replace('rend_img_obj', 'visible_vertices').replace('.png', '.npy')
        if not os.path.exists(gt_visible_path):
            print(f"    Missing GT visibility data, skipping...")
            continue
        gt_visible = set(np.load(gt_visible_path).tolist())
        
        # Extract or load GT features
        print(f"    Loading GT features...")
        if USE_CACHED_FEATURES:
            feat_gt, feat_gt_upsampled = load_cached_features(render_gt_path, obj_name, seq_dir, img_size)
            if feat_gt is None:
                if FALLBACK_TO_COMPUTE:
                    print(f"    Cached features not found, computing...")
                    feat_gt = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=render_gt)
                    feat_gt_upsampled = F.interpolate(feat_gt, size=(img_size, img_size), mode='bilinear', align_corners=False)
                else:
                    print(f"    ERROR: Cached features not found and FALLBACK_TO_COMPUTE=False")
                    continue
        else:
            feat_gt = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=render_gt)
            feat_gt_upsampled = F.interpolate(feat_gt, size=(img_size, img_size), mode='bilinear', align_corners=False)
        
        # Free raw GT features immediately
        del feat_gt

        gt_foreground_mask = (render_gt_mask > 128).astype(np.float32)
        
        # GT projection on GPU
        R_t, T_t = look_at_view_transform(distance_gt, elevation_gt, azimuth_gt, device=device)
        R_t = R_t @ flip.unsqueeze(0)
        R_t = torch.linalg.inv(R_t)
        R_mat = R_t[0]
        T_vec = T_t[0]
        
        projected_gt_gpu = perspective_projection_torch(
            gt_verts_gpu, T_vec, focal_gpu, princpt_gpu, R_mat
        )
        projected_gt = projected_gt_gpu.cpu().numpy()
        
        projected_gt_scaled = project_to_cropped_resized(
            projected_gt, gt_bbox, gt_sx, gt_sy, gt_ox, gt_oy
        )
        
        # Vectorized GT vertex filtering
        gt_valid_indices, gt_valid_positions, gt_px_int, gt_py_int, gt_valid_mask = \
            filter_visible_vertices_vectorized(
                len(gt_verts), projected_gt_scaled, gt_visible, render_gt_mask, img_size
            )
        
        # Precompute GT normalized features (kept on GPU for this GT view)
        with torch.no_grad():
            C, H, W = feat_gt_upsampled.shape[1], feat_gt_upsampled.shape[2], feat_gt_upsampled.shape[3]
            feat_gt_flat_norm = F.normalize(
                feat_gt_upsampled.view(C, H * W).permute(1, 0), p=2, dim=1
            )
            gt_foreground_flat = gt_foreground_mask.flatten()
            gt_fg_indices = np.where(gt_foreground_flat > 0.5)[0]
        
        # Build KD-tree for GT vertices (needed for reverse correspondences)
        if len(gt_valid_indices) > 0:
            gt_2d_tree = cKDTree(gt_valid_positions)
        else:
            gt_2d_tree = None
        
        # --- Batched MNN ratio computation across all H3D views ---
        print(f"    Computing mutual NN ratio with {len(h3d_preloaded)} H3D views (batched)...")
        
        mnn_ratios = compute_mutual_nn_ratio_batched(
            feat_gt_flat_norm, h3d_data_list_for_mnn,
            gt_fg_indices, max_pixels=MAX_PIXELS_FOR_SIMILARITY
        )
        
        h3d_similarities = {}
        for key, ratio in zip(h3d_sorted_keys, mnn_ratios):
            h3d_similarities[key] = ratio
            print(f"      H3D e={key[0]} a={key[1]}: MNN={ratio:.4f}")
        
        if len(h3d_similarities) == 0:
            print(f"    No valid H3D views, skipping GT view...")
            del feat_gt_upsampled, feat_gt_flat_norm
            torch.cuda.empty_cache()
            continue
        
        sorted_h3d = sorted(h3d_similarities.items(), key=lambda x: x[1], reverse=True)
        best_h3d_views = sorted_h3d[:min(TOP_K_MATCHES, len(sorted_h3d))]
        
        print(f"    Best {len(best_h3d_views)} H3D matches:")
        for (elev_h3d, azim_h3d), sim in best_h3d_views:
            print(f"      e={elev_h3d} a={azim_h3d}: MNN={sim:.4f}")
        
        # --- Establish correspondences with best matching H3D views ---
        for rank, ((elevation_h3d, azimuth_h3d), similarity_score) in enumerate(best_h3d_views):
            print(f"\n    Pair rank {rank+1}: GT e={elevation_gt} a={azimuth_gt} <-> "
                  f"H3D e={elevation_h3d} a={azimuth_h3d} (MNN={similarity_score:.4f})")
            
            view_key = (elevation_h3d, azimuth_h3d)
            h3d_data = h3d_preloaded[view_key]
            
            # ================================================================
            # Move H3D features to GPU for this pair, then free
            # ================================================================
            h3d_feat_upsampled_gpu = h3d_data['feat_upsampled'].cuda(non_blocking=True)
            
            fwd_corr_map, fwd_conf_map, rev_corr_map, rev_conf_map = \
                compute_dense_correspondence_both_directions(
                    feat_gt_upsampled, h3d_feat_upsampled_gpu, img_size
                )
            
            del h3d_feat_upsampled_gpu
            
            # ================================================================
            # Forward correspondences (GT -> H3D)
            # ================================================================
            h3d_valid_indices = h3d_data['valid_indices']

            if len(h3d_valid_indices) == 0:
                print(f"      No valid H3D vertices, skipping pair...")
                continue
            
            h3d_2d_tree = h3d_data['kd_tree_2d']
            if h3d_2d_tree is None:
                print(f"      No H3D KD-tree available, skipping pair...")
                continue
            
            # Vectorized GT -> H3D
            fwd_src_matched, fwd_tgt_matched, fwd_confs, fwd_skipped = \
                compute_correspondences_vectorized(
                    gt_valid_indices, gt_px_int, gt_py_int,
                    fwd_corr_map, fwd_conf_map,
                    h3d_2d_tree, h3d_valid_indices,
                    img_size, FIXED_THRESHOLD,
                    nn_distance_threshold=CLUSTER_RADIUS_2D,
                    threshold_mode=THRESHOLD_MODE,
                    adaptive_percentile=ADAPTIVE_PERCENTILE,
                    adaptive_floor=ADAPTIVE_FLOOR,
                )
            
            pair_correspondences = {}
            for i in range(len(fwd_src_matched)):
                gt_idx = int(fwd_src_matched[i])
                h3d_idx = int(fwd_tgt_matched[i])
                conf = float(fwd_confs[i])
                
                pair_correspondences[gt_idx] = {
                    'h3d_representative': h3d_idx,
                    'h3d_cluster': [h3d_idx],
                    'cluster_size': 1,
                    'confidence': conf,
                    'global_similarity': similarity_score,
                    'pair_id': (elevation_gt, azimuth_gt, elevation_h3d, azimuth_h3d)
                }
            
            _thresh_label = (f"{FIXED_THRESHOLD}" if THRESHOLD_MODE == 'fixed'
                             else f"adaptive(p={ADAPTIVE_PERCENTILE}, floor={ADAPTIVE_FLOOR})")
            print(f"      Forward: {len(pair_correspondences)} matches "
                  f"({fwd_skipped} below threshold {_thresh_label})")
            
            # ================================================================
            # Reverse correspondences (H3D -> GT)
            # ================================================================
            if gt_2d_tree is not None and len(h3d_valid_indices) > 0:
                rev_src_matched, rev_tgt_matched, rev_confs, _ = \
                    compute_correspondences_vectorized(
                        h3d_valid_indices, h3d_data['px_int'], h3d_data['py_int'],
                        rev_corr_map, rev_conf_map,
                        gt_2d_tree, gt_valid_indices,
                        img_size, FIXED_THRESHOLD,
                        nn_distance_threshold=CLUSTER_RADIUS_2D,
                        threshold_mode=THRESHOLD_MODE,
                        adaptive_percentile=ADAPTIVE_PERCENTILE,
                        adaptive_floor=ADAPTIVE_FLOOR,
                    )
                
                # Merge into global reverse correspondences (keep best confidence)
                new_rev = 0
                updated_rev = 0
                for i in range(len(rev_src_matched)):
                    h3d_idx = int(rev_src_matched[i])
                    gt_idx = int(rev_tgt_matched[i])
                    conf = float(rev_confs[i])
                    
                    if h3d_idx not in reverse_correspondences:
                        reverse_correspondences[h3d_idx] = {
                            'gt_idx': gt_idx,
                            'confidence': conf
                        }
                        new_rev += 1
                    elif conf > reverse_correspondences[h3d_idx]['confidence']:
                        reverse_correspondences[h3d_idx] = {
                            'gt_idx': gt_idx,
                            'confidence': conf
                        }
                        updated_rev += 1
                
                print(f"      Reverse: {len(rev_src_matched)} matches this pair, "
                      f"{len(reverse_correspondences)} total ({new_rev} new, {updated_rev} updated)")
            
            del fwd_corr_map, fwd_conf_map, rev_corr_map, rev_conf_map
            
            # --- Per-pair 2D visualization (inspection only; the aggregation
            # bookkeeping below must run regardless of the viz flag) ---
            if len(pair_correspondences) > 0:
                if save_pair_vis:
                    pair_vis_path = os.path.join(
                        pair_vis_dir,
                        f'corr_gt_e{elevation_gt}_a{azimuth_gt}__h3d_e{elevation_h3d}_a{azimuth_h3d}_rank{rank+1}.png'
                    )
                    visualize_pair_correspondences(
                        render_gt, h3d_data['render'],
                        projected_gt_scaled, h3d_data['projected_scaled'],
                        pair_correspondences, pair_vis_path,
                        similarity_score=similarity_score
                    )
                    print(f"      Saved visualization to {pair_vis_path}")

                pair_key = (elevation_h3d, azimuth_h3d, elevation_gt, azimuth_gt, rank)
                all_pair_correspondences[pair_key] = pair_correspondences

        # Clear GT features for this view
        del feat_gt_upsampled, feat_gt_flat_norm
        print(f"    Done with GT view e={elevation_gt} a={azimuth_gt}")

    # Free preloaded H3D data and projection tensors
    del h3d_preloaded, h3d_data_list_for_mnn
    del gt_verts_gpu, focal_gpu, princpt_gpu
    torch.cuda.empty_cache()

    if len(all_pair_correspondences) == 0:
        print(f"  No correspondences found, skipping...")
        return False

    # =========================================================================
    # SIMPLE BEST-MATCH AGGREGATION
    # =========================================================================
    
    print(f"\n[TIMING] match loop over {len(gt_renders_by_view)} GT views: "
          f"{time.time() - _t_match_loop:.2f}s")
    _t_aggregate = time.time()
    print(f"\n  Aggregating correspondences: keeping best match per GT vertex...")

    # Optional per-pair (per-view-pair) z-score calibration of the per-vertex
    # confidence, to avoid systematically demoting dim template views during
    # multi-view aggregation below. Legacy default ('none') leaves calibrated
    # confidences identical to raw.
    pair_calibration = {}
    if CALIBRATION == 'zscore':
        for pair_key, pair_correspondences in all_pair_correspondences.items():
            confs_in_pair = np.array(
                [d['confidence'] for d in pair_correspondences.values()],
                dtype=np.float64,
            )
            if confs_in_pair.size >= 2:
                mu = float(confs_in_pair.mean())
                sigma = float(confs_in_pair.std())
                if sigma <= 1e-8:
                    sigma = 1.0
                pair_calibration[pair_key] = (mu, sigma)
            else:
                pair_calibration[pair_key] = (0.0, 1.0)

    def _calibrated_conf(pair_key, raw_conf):
        if CALIBRATION != 'zscore':
            return raw_conf
        mu, sigma = pair_calibration.get(pair_key, (0.0, 1.0))
        return (raw_conf - mu) / sigma

    gt_candidates = {}

    for pair_key, pair_correspondences in all_pair_correspondences.items():
        for gt_idx, data in pair_correspondences.items():
            if gt_idx not in gt_candidates:
                gt_candidates[gt_idx] = []

            cal_conf = _calibrated_conf(pair_key, data['confidence'])

            gt_candidates[gt_idx].append({
                'h3d_idx': data['h3d_representative'],
                'confidence': data['confidence'],
                'global_similarity': data['global_similarity'],
                'score': cal_conf * data['global_similarity'],
                'pair_id': data['pair_id']
            })

    forward_correspondences = {}

    for gt_idx, candidates in gt_candidates.items():
        best = max(candidates, key=lambda c: c['score'])
        forward_correspondences[gt_idx] = {
            'h3d_idx': best['h3d_idx'],
            'confidence': best['confidence'],
            'global_similarity': best['global_similarity'],
            'score': best['score'],
            'num_candidates': len(candidates),
            'pair_id': best['pair_id']
        }

    print(f"  Forward correspondences: {len(forward_correspondences)} GT -> H3D matches")
    print(f"  Average raw cosine confidence: {np.mean([v['confidence'] for v in forward_correspondences.values()]):.4f}")
    print(f"  Average MNN global similarity: {np.mean([v['global_similarity'] for v in forward_correspondences.values()]):.4f}")
    print(f"  Average combined score: {np.mean([v['score'] for v in forward_correspondences.values()]):.4f}")
    print(f"  Average candidates per vertex: {np.mean([v['num_candidates'] for v in forward_correspondences.values()]):.2f}")

    # =========================================================================
    # CYCLE CONSISTENCY
    # =========================================================================
    
    print(f"\n  Reverse correspondences collected during forward pass: {len(reverse_correspondences)}")

    consistency_scores = check_cycle_consistency_with_neighborhoods(
        forward_correspondences,
        reverse_correspondences,
        gt_verts,
        h3d_verts,
        gt_faces,
        h3d_faces,
        gt_scale_factor=GT_NEIGHBORHOOD_SCALE,
        h3d_scale_factor=H3D_NEIGHBORHOOD_SCALE
    )

    final_correspondences = {}
    
    for gt_idx, forward_data in forward_correspondences.items():
        if gt_idx not in consistency_scores:
            continue
        
        consistency = consistency_scores[gt_idx]
        combined_score = forward_data['score'] * consistency['score']
        
        final_correspondences[gt_idx] = {
            'h3d_idx': forward_data['h3d_idx'],
            'confidence': forward_data['confidence'],
            'global_similarity': forward_data['global_similarity'],
            'cycle_score': consistency['score'],
            'cycle_consistent': consistency['cycle_consistent'],
            'consistency_type': consistency['consistency_type'],
            'score': combined_score
        }

    high_quality = {
        gt_idx: data
        for gt_idx, data in final_correspondences.items()
        if data['cycle_score'] > 0.5
    }

    print(f"  Final correspondences after cycle consistency: {len(high_quality)}/{len(forward_correspondences)}")
    print(f"[TIMING] aggregate + cycle consistency: {time.time() - _t_aggregate:.2f}s")

    # =========================================================================
    # SAVE RESULTS AND VISUALIZE
    # =========================================================================
    
    output_dir = os.path.join(os.path.dirname(img_path), _corres_dirname())
    # output_dir = os.path.join(os.path.dirname(img_path), 'correspondences_new')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'final_combined_correspondences.npz')

    gt_indices = np.array(list(high_quality.keys()), dtype=np.int64)
    h3d_indices = np.array([v['h3d_idx'] for v in high_quality.values()], dtype=np.int64)
    confidences_arr = np.array([v['confidence'] for v in high_quality.values()])
    global_similarities_arr = np.array([v['global_similarity'] for v in high_quality.values()])
    cycle_scores = np.array([v['cycle_score'] for v in high_quality.values()])
    scores = np.array([v['score'] for v in high_quality.values()])

    np.savez(output_path,
             gt_indices=gt_indices,
             h3d_indices=h3d_indices,
             confidences=confidences_arr,
             global_similarities=global_similarities_arr,
             cycle_scores=cycle_scores,
             scores=scores)

    print(f"  Saved final correspondences to {output_path}")

    # --- Create 3D visualization ---
    print(f"\n  Creating 3D visualization with rigid alignment...")
    visualization_dir = os.path.join(output_dir, 'visualization')
    
    aligned_gt_verts, align_R, align_t, align_scale = create_combined_visualization(
        gt_verts, gt_faces,
        h3d_verts, h3d_faces,
        high_quality,
        visualization_dir
    )

    print(f"\n  Correspondence Statistics:")
    print(f"    Total GT vertices: {len(gt_verts)}")
    print(f"    GT vertices with correspondences: {len(high_quality)}")
    print(f"    Coverage: {len(high_quality)/len(gt_verts)*100:.1f}%")
    if len(confidences_arr) > 0:
        print(f"    Average raw cosine confidence: {confidences_arr.mean():.3f}")
        print(f"    Average MNN global similarity: {global_similarities_arr.mean():.3f}")
        print(f"    Average cycle consistency score: {cycle_scores.mean():.3f}")
        print(f"    Cycle consistency breakdown:")
        strong_count = sum(1 for v in high_quality.values() if v['consistency_type'] == 'strong')
        weak_count = sum(1 for v in high_quality.values() if v['consistency_type'] == 'weak')
        total_hq = len(high_quality)
        print(f"      - Strong: {strong_count} ({strong_count/total_hq*100:.1f}%)")
        print(f"      - Weak: {weak_count} ({weak_count/total_hq*100:.1f}%)")

    # Single-line summary for easy grep/diff across runs. Unparsed fields are 'nan'.
    _cov = len(high_quality) / max(len(gt_verts), 1) * 100.0
    _rc = float(confidences_arr.mean()) if len(confidences_arr) > 0 else float('nan')
    _gs = float(global_similarities_arr.mean()) if len(confidences_arr) > 0 else float('nan')
    _cs = float(cycle_scores.mean()) if len(confidences_arr) > 0 else float('nan')
    print(
        f"[SUMMARY] seq={folder_name} "
        f"subdir={_corres_dirname()} "
        f"gt_verts={len(gt_verts)} "
        f"matched={len(high_quality)} "
        f"coverage_pct={_cov:.2f} "
        f"raw_cos_mean={_rc:.4f} "
        f"mnn_sim_mean={_gs:.4f} "
        f"cycle_mean={_cs:.4f} "
        f"threshold_mode={THRESHOLD_MODE} "
        f"fixed_threshold={FIXED_THRESHOLD} "
        f"adaptive_p={ADAPTIVE_PERCENTILE} "
        f"adaptive_floor={ADAPTIVE_FLOOR} "
        f"calibration={CALIBRATION}"
    )

    return True

img_size = 480

# Matching / memory-management settings
MAX_PIXELS_FOR_SIMILARITY = 10000
TOP_K_MATCHES = 3
# USE_CACHED_FEATURES = True
USE_CACHED_FEATURES = False
FALLBACK_TO_COMPUTE = True
CLUSTER_RADIUS_2D = 15.0
GT_NEIGHBORHOOD_SCALE = 3.0
H3D_NEIGHBORHOOD_SCALE = 3.0

RAW_COSINE_THRESHOLD = 0.6

TOP_K_INPUT_SIMILAR = 5

# M1 flag-gated additions. Defaults reproduce the legacy pipeline byte-for-byte.
# Set in __main__ from argparse; any module-level consumer below reads these.
THRESHOLD_MODE = 'fixed'           # {'fixed', 'adaptive'}
FIXED_THRESHOLD = RAW_COSINE_THRESHOLD
ADAPTIVE_PERCENTILE = 0.7
ADAPTIVE_FLOOR = 0.35
CALIBRATION = 'none'               # {'none', 'zscore'}


class _Tee:
    """Write to multiple streams. Used to duplicate stdout/stderr into a log file.

    Absorbs errors from the file stream (e.g. disk full) so logging never
    crashes the pipeline; writes to the primary console stream still go
    through.
    """

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        try:
            self._secondary.write(data)
        except Exception:
            pass

    def flush(self):
        self._primary.flush()
        try:
            self._secondary.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._primary, 'isatty', lambda: False)()

    def fileno(self):
        return self._primary.fileno()


@contextlib.contextmanager
def _tee_stdio(log_path):
    """Duplicate sys.stdout and sys.stderr into ``log_path`` for the block.

    Preserves original streams on exit even when exceptions propagate. The
    log file is opened in append mode so re-running a sequence keeps prior
    logs under the same output dir.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fp = open(log_path, 'a', buffering=1)  # line-buffered
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(saved_stdout, log_fp)
    sys.stderr = _Tee(saved_stderr, log_fp)
    try:
        yield log_fp
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass


def _log_run_header(img_path, log_path):
    """Print a detailed header block to the current (teed) stdout.

    Includes run timestamp, host, active M1 flag state, and resolved paths.
    Goes into both the console and ``log_path`` because stdout is teed.
    """
    seq_dir = os.path.dirname(img_path)
    folder_name = os.path.basename(seq_dir)
    banner = '#' * 100
    print(banner)
    print(f"# RUN LOG - {folder_name}")
    print(f"#   timestamp (UTC): {datetime.datetime.utcnow().isoformat()}Z")
    print(f"#   host:            {socket.gethostname()} ({platform.platform()})")
    print(f"#   python:          {sys.version.split()[0]}")
    print(f"#   pid:             {os.getpid()}")
    print(f"#   cwd:             {os.getcwd()}")
    print(f"#   script:          {os.path.abspath(__file__)}")
    print(f"#   img_path:        {img_path}")
    print(f"#   seq_dir:         {seq_dir}")
    print(f"#   output_subdir:   {_corres_dirname()}/")
    print(f"#   log_path:        {log_path}")
    print(f"#   flags:")
    print(f"#     THRESHOLD_MODE      = {THRESHOLD_MODE}")
    print(f"#     FIXED_THRESHOLD     = {FIXED_THRESHOLD}")
    print(f"#     ADAPTIVE_PERCENTILE = {ADAPTIVE_PERCENTILE}")
    print(f"#     ADAPTIVE_FLOOR      = {ADAPTIVE_FLOOR}")
    print(f"#     CALIBRATION         = {CALIBRATION}")
    print(f"#     RAW_COSINE_THRESHOLD (reference) = {RAW_COSINE_THRESHOLD}")
    print(f"#     CLUSTER_RADIUS_2D   = {CLUSTER_RADIUS_2D}")
    print(f"#     TOP_K_MATCHES       = {TOP_K_MATCHES}")
    print(f"#     TOP_K_INPUT_SIMILAR = {TOP_K_INPUT_SIMILAR}")
    print(f"#     GT_NEIGHBORHOOD_SCALE  = {GT_NEIGHBORHOOD_SCALE}")
    print(f"#     H3D_NEIGHBORHOOD_SCALE = {H3D_NEIGHBORHOOD_SCALE}")
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            dev = _torch.cuda.current_device()
            print(f"#   cuda device:     {_torch.cuda.get_device_name(dev)} (idx={dev})")
            print(f"#   cuda cap:        {_torch.cuda.get_device_capability(dev)}")
            print(f"#   torch version:   {_torch.__version__}")
    except Exception as _e:
        print(f"#   cuda info unavailable: {_e}")
    print(banner)
    print()


def _corres_dirname():
    """Output subdirectory name for this run.

    Legacy (all M1 flags at defaults) -> ``'correspondences'`` (byte-identical
    to pre-M1 behavior). Non-legacy runs get a suffixed dirname so different
    configurations don't overwrite each other.
    """
    legacy = (
        THRESHOLD_MODE == 'fixed'
        and float(FIXED_THRESHOLD) == float(RAW_COSINE_THRESHOLD)
        and CALIBRATION == 'none'
    )
    if legacy:
        return 'correspondences'
    parts = []
    if THRESHOLD_MODE == 'adaptive':
        parts.append(f"adaptive-p{ADAPTIVE_PERCENTILE}-f{ADAPTIVE_FLOOR}")
    elif float(FIXED_THRESHOLD) != float(RAW_COSINE_THRESHOLD):
        parts.append(f"fixed-{FIXED_THRESHOLD}")
    if CALIBRATION != 'none':
        parts.append(f"cal-{CALIBRATION}")
    return 'correspondences__' + '_'.join(parts)


def run(seq_dir, obj_name, template, overwrite=False, models=None,
        h3d_features=None, save_pair_vis=True):
    """MILO single-sequence entry (geo-aware env): dense template<->LRM correspondences.

    Loads SD + DINO once (or reuses ``models`` from stage1.load_models()),
    processes one sequence, writes correspondences/final_combined_correspondences.npz.

    ``h3d_features``: optional stage-1 raw-feature handoff for the top-K H3D
    views. ``save_pair_vis=False`` skips the per-pair inspection PNGs.
    """
    global _OBJ_NAME_OVERRIDE, _TEMPLATE_OVERRIDE
    out_path = os.path.join(seq_dir, _corres_dirname(), 'final_combined_correspondences.npz')
    if not overwrite and os.path.exists(out_path):
        print(f"[correspond] stage2 output exists, skipping {os.path.basename(seq_dir)}")
        return False
    jpgs = sorted(p for p in glob.glob(os.path.join(seq_dir, '*.jpg'))
                  if not (p.endswith('_human.jpg') or p.endswith('_object.jpg')))
    if not jpgs:
        print(f"[correspond] no input .jpg in {seq_dir}")
        return False
    _OBJ_NAME_OVERRIDE = obj_name
    _TEMPLATE_OVERRIDE = template
    if models is None:
        print("[correspond] stage2: loading SD + DINO models...")
        sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches * 16,
                                      num_timesteps=50, block_indices=[2, 5, 8, 11])
        extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
        aggre_net_run = aggre_net
    else:
        sd_model, sd_aug = models['sd_model'], models['sd_aug']
        extractor_vit, aggre_net_run = models['extractor_vit'], models['aggre_net']
        # Restore the exact np/python RNG state a legacy inline stage-2 load
        # (load_model + ViTExtractor, which reseed then consume some draws)
        # would have left behind, so stage 2's subsampling stream is
        # bit-identical to the unshared-load flow.
        import random as _pyrandom
        np.random.set_state(models['rng_stage2'][0])
        _pyrandom.setstate(models['rng_stage2'][1])
    return _process_sequence_body(jpgs[0], sd_model, sd_aug, aggre_net_run, extractor_vit,
                                  h3d_features=h3d_features, save_pair_vis=save_pair_vis)


def main(sequence_dirs=None, save_pair_vis=True):
    """Main function to process correspondences."""
    print("="*80)
    print("Correspondence Computation Script (Optimized)")
    print("="*80)

    # Load models
    print("Loading models...")
    sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
    print("Models loaded successfully!")
    
    img_list = []
    for seq_dir in sequence_dirs:
        jpg_files = glob.glob(os.path.join(seq_dir, '*.jpg'))
        if jpg_files:
            img_list.append(sorted(jpg_files)[0])
    
    print(f"\nFound {len(img_list)} sequences to process")
    print(f"Configuration:")
    print(f"  - Use cached features: {USE_CACHED_FEATURES}")
    print(f"  - Fallback to compute: {FALLBACK_TO_COMPUTE}")
    print(f"  - Max pixels for similarity: {MAX_PIXELS_FOR_SIMILARITY}")
    print(f"  - Top-K H3D matches per GT view: {TOP_K_MATCHES}")
    print(f"  - Top-K input-similar H3D views: {TOP_K_INPUT_SIMILAR}")
    print(f"  - 2D cluster radius / NN threshold: {CLUSTER_RADIUS_2D} pixels")
    if THRESHOLD_MODE == 'fixed':
        print(f"  - Cosine threshold: fixed @ {FIXED_THRESHOLD}")
    else:
        print(f"  - Cosine threshold: adaptive (percentile={ADAPTIVE_PERCENTILE}, "
              f"floor={ADAPTIVE_FLOOR})")
    print(f"  - Calibration: {CALIBRATION}")
    print(f"  - Output subdirectory: {_corres_dirname()}/")
    print(f"  - GT neighborhood scale: {GT_NEIGHBORHOOD_SCALE}x local mesh scale")
    print(f"  - H3D neighborhood scale: {H3D_NEIGHBORHOOD_SCALE}x local mesh scale")
    print()
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    for img_idx, img_path in enumerate(sorted(img_list)):
        try:
            success = process_sequence(img_path, sd_model, sd_aug, aggre_net, extractor_vit,
                                       save_pair_vis=save_pair_vis)
            if success:
                total_processed += 1
            else:
                total_skipped += 1
                
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            total_errors += 1
            continue
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Total errors: {total_errors}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correspondences for multi-view renders")
    parser.add_argument('--idx', type=int, default=None, help='Batch index (0-based)')
    parser.add_argument('--num_batches', type=int, default=4,
                       help='Number of batches to divide the dataset into')

    # M1 flag surface. Every default reproduces the legacy pipeline.
    parser.add_argument('--threshold_mode', choices=['fixed', 'adaptive'], default='fixed',
                        help="Threshold rule. 'fixed' uses --fixed_threshold (legacy).")
    parser.add_argument('--fixed_threshold', type=float, default=RAW_COSINE_THRESHOLD,
                        help='Cosine cutoff used when --threshold_mode=fixed.')
    parser.add_argument('--adaptive_percentile', type=float, default=0.7,
                        help='Quantile for --threshold_mode=adaptive. Keep scores >= quantile.')
    parser.add_argument('--adaptive_floor', type=float, default=0.35,
                        help='Hard floor on cosine under --threshold_mode=adaptive.')
    parser.add_argument('--calibration', choices=['none', 'zscore'], default='none',
                        help="Per-pair confidence calibration before multi-view aggregation.")
    parser.add_argument('--legacy', action='store_true',
                        help='Force all M1 flags to their legacy defaults, warning about overrides.')
    parser.add_argument('--skip_pair_vis', action='store_true',
                        help='Skip the per-pair visualization PNGs (inspection only; npz unaffected).')

    args = parser.parse_args()

    if args.legacy:
        legacy_defaults = {
            'threshold_mode': 'fixed',
            'fixed_threshold': RAW_COSINE_THRESHOLD,
            'adaptive_percentile': 0.7,
            'adaptive_floor': 0.35,
            'calibration': 'none',
        }
        for name, legacy_val in legacy_defaults.items():
            current = getattr(args, name)
            if current != legacy_val:
                print(f"[--legacy] overriding --{name}={current} -> {legacy_val}")
                setattr(args, name, legacy_val)

    THRESHOLD_MODE = args.threshold_mode
    FIXED_THRESHOLD = args.fixed_threshold
    ADAPTIVE_PERCENTILE = args.adaptive_percentile
    ADAPTIVE_FLOOR = args.adaptive_floor
    CALIBRATION = args.calibration

    root_dir = os.environ.get("MILO_GEO_TEST_ROOT", "/path/to/data_seq")

    if args.idx is not None:
        batches = divide_batches(
            root_dir=root_dir,
            num_batches=args.num_batches
        )
        
        if len(batches) == 0:
            print("No sequences to process")
            exit(0)
        
        if args.idx >= len(batches):
            print(f"Error: Batch index {args.idx} out of range (0-{len(batches)-1})")
            exit(1)
        
        sequence_dirs = batches[args.idx]
        print(f"\nProcessing batch {args.idx}/{len(batches)-1} with {len(sequence_dirs)} sequences")
        print(f"Sequences in this batch: {[os.path.basename(s) for s in sequence_dirs[:5]]}{'...' if len(sequence_dirs) > 5 else ''}\n")
        
        main(sequence_dirs=sequence_dirs, save_pair_vis=not args.skip_pair_vis)
    else:
        print("Processing all sequences (no batch index specified)")
        main(sequence_dirs=None, save_pair_vis=not args.skip_pair_vis)

"""
pipeline/steps/_align_utils.py

Numpy Kabsch / ICP helpers for the template_align step. All transforms use the
convention x' = R @ (s * x) + t and are returned as (s, R, t).
"""

import numpy as np
from scipy.spatial import cKDTree


def kabsch_with_scale(source_pts, target_pts, allow_scale=True):
    """Optimal similarity transform via Kabsch: returns (s, R, t) with
    target ≈ R @ (s * source) + t. s is fixed to 1 when allow_scale=False."""
    src_centroid = source_pts.mean(axis=0)
    tgt_centroid = target_pts.mean(axis=0)

    src_c = source_pts - src_centroid
    tgt_c = target_pts - tgt_centroid

    if allow_scale:
        s = np.sqrt((tgt_c ** 2).sum()) / np.sqrt((src_c ** 2).sum())
    else:
        s = 1.0

    src_c_s = src_c * s

    H = src_c_s.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    t = tgt_centroid - R @ (src_centroid * s)

    return s, R, t


def kabsch_with_scale_weighted(source_pts, target_pts, weights, allow_scale=True):
    """Per-point-weighted variant of kabsch_with_scale. Returns (s, R, t)."""
    weights = weights / weights.sum()

    src_centroid = (source_pts * weights[:, None]).sum(axis=0)
    tgt_centroid = (target_pts * weights[:, None]).sum(axis=0)

    src_c = source_pts - src_centroid
    tgt_c = target_pts - tgt_centroid

    if allow_scale:
        src_norm_sq = (weights[:, None] * src_c ** 2).sum()
        tgt_norm_sq = (weights[:, None] * tgt_c ** 2).sum()
        s = np.sqrt(tgt_norm_sq / src_norm_sq)
    else:
        s = 1.0

    src_c_s = src_c * s

    H = (src_c_s * weights[:, None]).T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    t = tgt_centroid - R @ (src_centroid * s)

    return s, R, t


def apply_transform(vertices, s, R, t):
    """Apply similarity transform: s * R @ v + t"""
    return (R @ (vertices * s).T).T + t


def iterative_icp_correspondence_fitting(
    gt_verts,
    gt_corr_idx,
    h3d_corr_idx,
    h3d_verts,
    pred_obj_verts,
    corr_weights=None,
    correspondence_weight_start=2.0,
    correspondence_weight_end=2.0,
    icp_weight_start=10.0,
    icp_weight_end=10.0,
    max_iters=20,
    tolerance=1e-5,
    icp_distance_threshold=None,
    allow_scale=True,
    verbose=False,
):
    """
    Refine alignment by iteratively minimizing both:
      1. Correspondence constraint: align gt_corr_idx vertices to h3d_corr_idx positions
      2. ICP constraint: align all GT vertices to nearest neighbors in pred_obj

    correspondence_weight and icp_weight are linearly interpolated from *_start
    at iter 0 to *_end at iter (max_iters - 1). icp_distance_threshold (if set)
    anneals from 5x to 1x over the iterations. Returns (aligned_verts, history).
    """
    current_verts = gt_verts.copy()

    corr_target_pts = h3d_verts[h3d_corr_idx]
    pred_tree = cKDTree(pred_obj_verts)

    if corr_weights is None:
        corr_weights = np.ones(len(gt_corr_idx))
    corr_weights = corr_weights / corr_weights.max()

    history = {
        'corr_errors': [],
        'icp_errors': [],
        'total_errors': [],
        'num_icp_pairs': [],
        'corr_weight_t': [],
        'icp_weight_t': [],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Starting iterative refinement (correspondence + ICP)")
        print(f"  corr_w: {correspondence_weight_start} -> {correspondence_weight_end}")
        print(f"  icp_w:  {icp_weight_start} -> {icp_weight_end}")
        print("=" * 60)

    prev_total_error = float('inf')

    for iter_idx in range(max_iters):
        frac = iter_idx / max(max_iters - 1, 1)
        corr_w_t = correspondence_weight_start + (correspondence_weight_end - correspondence_weight_start) * frac
        icp_w_t = icp_weight_start + (icp_weight_end - icp_weight_start) * frac

        distances, nearest_indices = pred_tree.query(current_verts, k=1)

        if icp_distance_threshold is not None:
            current_thresh = icp_distance_threshold * (5.0 * (1 - frac) + frac)
            valid_mask = distances < current_thresh
            icp_src_idx = np.where(valid_mask)[0]
            icp_tgt_pts = pred_obj_verts[nearest_indices[valid_mask]]
        else:
            icp_src_idx = np.arange(len(current_verts))
            icp_tgt_pts = pred_obj_verts[nearest_indices]

        icp_src_pts = current_verts[icp_src_idx]

        corr_src = current_verts[gt_corr_idx]
        corr_tgt = corr_target_pts

        n_corr = len(corr_src)
        n_icp = len(icp_src_pts)

        weights = np.concatenate([
            corr_weights * corr_w_t,
            np.full(n_icp, icp_w_t)
        ])
        weights = weights / weights.sum()

        combined_src = np.vstack([corr_src, icp_src_pts])
        combined_tgt = np.vstack([corr_tgt, icp_tgt_pts])

        src_centroid = (combined_src * weights[:, None]).sum(axis=0)
        tgt_centroid = (combined_tgt * weights[:, None]).sum(axis=0)

        src_c = combined_src - src_centroid
        tgt_c = combined_tgt - tgt_centroid

        if allow_scale:
            src_norm_sq = (weights[:, None] * src_c ** 2).sum()
            tgt_norm_sq = (weights[:, None] * tgt_c ** 2).sum()
            s = np.sqrt(tgt_norm_sq / src_norm_sq)
        else:
            s = 1.0

        src_c_s = src_c * s

        H = (src_c_s * weights[:, None]).T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, d]) @ U.T

        t = tgt_centroid - R @ (src_centroid * s)

        current_verts = apply_transform(current_verts, s, R, t)

        corr_err = np.linalg.norm(current_verts[gt_corr_idx] - corr_target_pts, axis=1)
        icp_err = np.linalg.norm(current_verts[icp_src_idx] - icp_tgt_pts, axis=1)

        corr_mean_err = corr_err.mean()
        icp_mean_err = icp_err.mean()
        total_error = corr_w_t * corr_mean_err + icp_w_t * icp_mean_err

        history['corr_errors'].append(corr_mean_err)
        history['icp_errors'].append(icp_mean_err)
        history['total_errors'].append(total_error)
        history['num_icp_pairs'].append(n_icp)
        history['corr_weight_t'].append(corr_w_t)
        history['icp_weight_t'].append(icp_w_t)

        if verbose:
            print(f"Iter {iter_idx + 1:2d} | Corr: {corr_mean_err:.4f} (n={n_corr}, w={corr_w_t:.2f}) | "
                  f"ICP: {icp_mean_err:.4f} (n={n_icp}, w={icp_w_t:.2f}) | "
                  f"Total: {total_error:.4f} | Scale: {s:.4f}")

        error_change = abs(prev_total_error - total_error)
        if error_change < tolerance:
            if verbose:
                print(f"Converged after {iter_idx + 1} iterations (change: {error_change:.6f})")
            break

        prev_total_error = total_error

    return current_verts, history


def icp_only_refinement(
    gt_verts,
    pred_obj_verts,
    icp_weight=1.0,
    max_iters=50,
    tolerance=1e-6,
    icp_distance_threshold=None,
    allow_scale=True,
    verbose=False,
):
    """
    ICP-only refinement for when no correspondences are available.

    The GT mesh is first centroid-aligned to the predicted object and given an
    initial bbox-extent scale (geometric mean of per-axis ratios, clipped to
    [0.1, 10]), then refined with point-to-point ICP (same annealed
    icp_distance_threshold as above). Returns (aligned_verts, history).
    """
    V = len(gt_verts)
    current_verts = np.ascontiguousarray(gt_verts.copy(), dtype=np.float64)

    gt_centroid = current_verts.mean(axis=0)
    pred_centroid = pred_obj_verts.mean(axis=0)
    current_verts += (pred_centroid - gt_centroid)
    if verbose:
        print(f"  Centroid shift applied: {np.linalg.norm(pred_centroid - gt_centroid):.4f}")

    gt_extent = current_verts.max(axis=0) - current_verts.min(axis=0)
    pred_extent = pred_obj_verts.max(axis=0) - pred_obj_verts.min(axis=0)
    axis_ratios = pred_extent / np.maximum(gt_extent, 1e-12)
    init_scale = np.exp(np.log(np.maximum(axis_ratios, 1e-12)).mean())
    init_scale = np.clip(init_scale, 0.1, 10.0)
    centroid_now = current_verts.mean(axis=0)
    current_verts = centroid_now + (current_verts - centroid_now) * init_scale
    if verbose:
        print(f"  Initial bbox scale applied: {init_scale:.4f}")

    pred_tree = cKDTree(pred_obj_verts)

    history = {
        'icp_errors': [],
        'total_errors': [],
        'num_icp_pairs': [],
        'scale': [],
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("Starting ICP-only refinement (point-to-point)")
        print("=" * 60)

    prev_total_error = float('inf')
    cumulative_s = init_scale

    for iter_idx in range(max_iters):
        distances, nearest_indices = pred_tree.query(current_verts, k=1)

        if icp_distance_threshold is not None:
            frac = iter_idx / max(max_iters - 1, 1)
            current_thresh = icp_distance_threshold * (5.0 * (1 - frac) + frac)
            valid_mask = distances < current_thresh
            icp_src_idx = np.where(valid_mask)[0]
        else:
            icp_src_idx = np.arange(V)
            valid_mask = None

        n_icp = len(icp_src_idx)
        if n_icp < 3:
            if verbose:
                print(f"  Iter {iter_idx + 1}: only {n_icp} valid ICP pairs, stopping.")
            break

        icp_src_pts = current_verts[icp_src_idx]
        icp_nn_idx = nearest_indices[icp_src_idx] if valid_mask is not None else nearest_indices
        icp_tgt_pts = pred_obj_verts[icp_nn_idx]

        combined_w = np.ones(n_icp) * icp_weight
        combined_w /= combined_w.sum()

        src_centroid = (icp_src_pts * combined_w[:, None]).sum(axis=0)
        tgt_centroid = (icp_tgt_pts * combined_w[:, None]).sum(axis=0)

        src_c = icp_src_pts - src_centroid
        tgt_c = icp_tgt_pts - tgt_centroid

        s = 1.0
        if allow_scale:
            src_norm_sq = (combined_w[:, None] * src_c ** 2).sum()
            tgt_norm_sq = (combined_w[:, None] * tgt_c ** 2).sum()
            s = np.sqrt(tgt_norm_sq / max(src_norm_sq, 1e-12))

        H = (src_c * (s * combined_w[:, None])).T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        t = tgt_centroid - R @ (src_centroid * s)

        current_verts = apply_transform(current_verts, s, R, t)
        cumulative_s *= s

        icp_dists_post, _ = pred_tree.query(current_verts[icp_src_idx], k=1)
        icp_mean_err = icp_dists_post.mean()
        total_error = icp_weight * icp_mean_err

        history['icp_errors'].append(icp_mean_err)
        history['total_errors'].append(total_error)
        history['num_icp_pairs'].append(n_icp)
        history['scale'].append(cumulative_s)

        if verbose:
            print(f"Iter {iter_idx + 1:2d} | ICP: {icp_mean_err:.4f} (n={n_icp}) | "
                  f"Total: {total_error:.4f} | Scale: {cumulative_s:.4f}")

        error_change = abs(prev_total_error - total_error)
        if error_change < tolerance:
            if verbose:
                print(f"Converged after {iter_idx + 1} iterations (change: {error_change:.6f})")
            break
        prev_total_error = total_error

    return current_verts, history

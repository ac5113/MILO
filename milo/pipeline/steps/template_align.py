"""
pipeline/steps/template_align.py

Optional template-alignment final step (paper Sec 3.6). Registers an object
TEMPLATE mesh to the segmented LRM object point cloud using the geometry-aware
semantic correspondences from `correspond`, via a weighted Sim(3) Kabsch fit
followed by iterative correspondence + ICP refinement (ICP-only fallback when
no correspondences are found).

Inputs (per seq_dir):
  - correspondences/final_combined_correspondences.npz   (from `correspond`)
  - the template mesh (--template; same mesh `correspond` used)
  - fit/meshes_smooth_obj.obj                             (fit-frame LRM mesh; corr targets)
  - filtered_h3d_obj_pc.obj                               (isolated object pc; ICP target)
Outputs (-> correspondences/):
  - alignment_transform.npz   (s, R, t, gt_centroid, rot_x_180, errors, chamfer)
  - aligned_template.obj      (template aligned into the LRM/camera frame)

Standalone usage:
    python -m milo.pipeline.steps.template_align --seq_dir /path/to/seq \
        --template /path/to/template.obj
"""

import argparse
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from milo.pipeline.steps._align_utils import (
    kabsch_with_scale,
    kabsch_with_scale_weighted,
    apply_transform,
    iterative_icp_correspondence_fitting,
    icp_only_refinement,
)
from milo.pipeline.steps._common import flip_to_lrm_frame as _flip_to_lrm_frame
from milo.pipeline.steps._log import vprint, set_verbose


def _find(seq_dir, *names):
    for n in names:
        p = os.path.join(seq_dir, n)
        if os.path.exists(p):
            return p
    return None


def _ensure_vertex_colors(mesh):
    try:
        _ = mesh.visual.vertex_colors[0]
    except (AttributeError, IndexError):
        mesh.visual.vertex_colors = np.tile(
            np.array([200, 200, 200, 255], dtype=np.uint8), (len(mesh.vertices), 1)
        )


def run(seq_dir: str, template: str, verbose: bool = False) -> bool:
    """Align the template mesh to the segmented LRM object for one sequence."""
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    out_dir = os.path.join(seq_dir, "correspondences")
    out_npz = os.path.join(out_dir, "alignment_transform.npz")

    corr_path = os.path.join(out_dir, "final_combined_correspondences.npz")
    # Correspondence target must live in the SAME frame as the ICP target / the
    # segmented object. The fit applies a rot_x_180 to the LRM mesh, so the fit's
    # meshes_smooth_obj.obj (same vertex ordering as full_img_textured, so the corr
    # indices transfer) is the right target — using full_img_textured would leave the
    # aligned template in the LRM frame, 180°-off from segmented_object.obj.
    h3d_mesh_path = _find(seq_dir, "fit/meshes_smooth_obj.obj",
                          "full_img_textured.glb", "full_img_textured.obj")
    pred_obj_path = _find(seq_dir, "filtered_h3d_obj_pc.obj")

    if not os.path.isfile(template):
        print(f"[template_align] Skipping — template not found: {template}")
        return False
    if pred_obj_path is None:
        print(f"[template_align] Skipping — filtered_h3d_obj_pc.obj not found in {seq_dir} (run isolate first).")
        return False

    # --- load correspondences (optional; ICP-only fallback if absent/empty) ---
    gt_corr_idx = np.array([], dtype=np.int64)
    h3d_corr_idx = np.array([], dtype=np.int64)
    weights = None
    if os.path.exists(corr_path):
        d = np.load(corr_path)
        gt_corr_idx = d["gt_indices"].astype(np.int64)
        h3d_corr_idx = d["h3d_indices"].astype(np.int64)
        weights = d["scores"] if "scores" in d else (d["confidences"] if "confidences" in d else None)
    n_corr = len(gt_corr_idx)
    use_icp_only = (n_corr < 3) or (h3d_mesh_path is None)
    vprint(f"[template_align] {os.path.basename(seq_dir)} — {n_corr} correspondences"
          f"{' (ICP-only fallback)' if use_icp_only else ''}")

    # --- template mesh: center + rot_x_180 (must match `correspond`/render convention) ---
    gt_mesh = trimesh.load(template, process=False)
    centroid = np.asarray(gt_mesh.vertices).mean(axis=0)
    rot_x_180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
    gt_verts = (np.asarray(gt_mesh.vertices) - centroid) @ rot_x_180.T

    pred_obj_verts = np.asarray(trimesh.load(pred_obj_path, process=False).vertices)

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------- ICP-only fallback -----------------------------
    if use_icp_only:
        aligned, _ = icp_only_refinement(
            gt_verts=gt_verts, pred_obj_verts=pred_obj_verts,
            icp_weight=10.0, max_iters=100, tolerance=1e-6,
            icp_distance_threshold=0.5, allow_scale=True, verbose=verbose,
        )
        pred_tree = cKDTree(pred_obj_verts)
        aln_tree = cKDTree(aligned)
        d2p, _ = pred_tree.query(aligned, k=1)
        dfp, _ = aln_tree.query(pred_obj_verts, k=1)
        chamfer = (d2p.mean() + dfp.mean()) / 2
        out = gt_mesh.copy(); out.vertices = aligned
        _flip_to_lrm_frame(out)
        out.export(os.path.join(out_dir, "aligned_template.obj"))
        np.savez(out_npz, fallback_icp_only=True, gt_centroid=centroid, rot_x_180=rot_x_180,
                 chamfer_sym=chamfer, icp_error_mean=d2p.mean())
        print(f"[template_align] ICP-only chamfer(sym)={chamfer:.4f} → {out_dir}/aligned_template.obj")
        return True

    # --------------------------- correspondence path ----------------------------
    h3d_verts = np.asarray(trimesh.load(h3d_mesh_path, process=False, force="mesh").vertices)
    gt_pts, h3d_pts = gt_verts[gt_corr_idx], h3d_verts[h3d_corr_idx]

    # Stage 1: weighted Kabsch (Sim(3) Procrustes)
    if weights is not None:
        s1, R1, t1 = kabsch_with_scale_weighted(gt_pts, h3d_pts, weights, allow_scale=True)
    else:
        s1, R1, t1 = kabsch_with_scale(gt_pts, h3d_pts, allow_scale=True)
    aligned1 = apply_transform(gt_verts, s1, R1, t1)
    err1 = np.linalg.norm(h3d_pts - aligned1[gt_corr_idx], axis=1)

    # Stage 2: iterative correspondence + ICP refinement
    aligned2, _ = iterative_icp_correspondence_fitting(
        gt_verts=aligned1, gt_corr_idx=gt_corr_idx, h3d_corr_idx=h3d_corr_idx,
        h3d_verts=h3d_verts, pred_obj_verts=pred_obj_verts, corr_weights=weights,
        correspondence_weight_start=2.0, correspondence_weight_end=2.0,
        icp_weight_start=10.0, icp_weight_end=10.0,
        max_iters=50, tolerance=1e-6, icp_distance_threshold=0.5, allow_scale=True,
        verbose=verbose,
    )
    err2 = np.linalg.norm(h3d_pts - aligned2[gt_corr_idx], axis=1)
    pred_tree = cKDTree(pred_obj_verts)
    aln_tree = cKDTree(aligned2)
    d2p, _ = pred_tree.query(aligned2, k=1)
    dfp, _ = aln_tree.query(pred_obj_verts, k=1)
    chamfer = (d2p.mean() + dfp.mean()) / 2

    out = gt_mesh.copy(); out.vertices = aligned2
    _ensure_vertex_colors(out)
    out.visual.vertex_colors[gt_corr_idx] = [0, 128, 255, 255]
    _flip_to_lrm_frame(out)
    out.export(os.path.join(out_dir, "aligned_template.obj"))
    np.savez(out_npz, s1=s1, R1=R1, t1=t1, gt_centroid=centroid, rot_x_180=rot_x_180,
             stage1_error_mean=err1.mean(), stage2_error_mean=err2.mean(), chamfer_sym=chamfer)
    print(f"[template_align] stage1_err={err1.mean():.4f} stage2_err={err2.mean():.4f} "
          f"chamfer(sym)={chamfer:.4f} → {out_dir}/aligned_template.obj")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align an object template to the segmented LRM object for one sequence")
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--template", required=True, help="Path to the object template mesh (same one `correspond` used)")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, template=args.template, verbose=args.verbose)

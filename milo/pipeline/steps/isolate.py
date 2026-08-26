"""
pipeline/steps/isolate.py

Isolate the object vertices from the LRM mesh using the vertex classification
labels. By default it keeps every segmented object vertex; pass --filter to run
the optional outlier/cluster point-cloud filtering pipeline.

Final outputs at the top of the sequence folder (rotated 180° about X back into the
LRM / full_img_textured.glb orientation; the fit saves its meshes 180°-flipped):
  fitted_human.obj      — the optimized SMPL-H human mesh (from <seq>/fit/)
  segmented_object.obj  — the isolated object mesh (submesh of the LRM object with faces)
Plus filtered_h3d_obj_pc.obj (object point cloud, kept in the fit frame) +
filtered_h3d_obj_indices.npy.
Reads the fit's meshes from <seq_dir>/fit/ by default.

Standalone usage:
    python -m milo.pipeline.steps.isolate --seq_dir /path/to/seq

Module usage:
    from milo.pipeline.steps.isolate import run
    run(seq_dir="...", fit_output_dir="...")
"""

import argparse
import os
import time

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from milo.pipeline.config import fit_dir
from milo.pipeline.steps._common import flip_to_lrm_frame as _flip_to_lrm_frame
from milo.pipeline.steps._log import vprint, set_verbose


# ---------------------------------------------------------------------------
# PointCloudFilterPipeline (copied verbatim from isolate_hodome.py)
# ---------------------------------------------------------------------------

class PointCloudFilterPipeline:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.filtering_stats = {}

    def statistical_outlier_removal(self, points, indices=None, nb_neighbors=20, std_ratio=2.0):
        if self.verbose:
            vprint("Applying statistical outlier removal...")
            t0 = time.time()
        n = len(points)
        if indices is None:
            indices = np.arange(n)
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_dist = np.mean(distances[:, 1:], axis=1)
        threshold = np.mean(mean_dist) + std_ratio * np.std(mean_dist)
        mask = mean_dist < threshold
        filtered_points, filtered_indices = points[mask], indices[mask]
        removed = n - len(filtered_points)
        self.filtering_stats["statistical_outliers"] = {
            "removed": removed, "remaining": len(filtered_points),
            "percentage_removed": removed / n * 100,
        }
        if self.verbose:
            vprint(f"  Removed {removed} outliers ({removed/n*100:.1f}%) in {time.time()-t0:.2f}s")
        return filtered_points, filtered_indices

    def radius_based_filtering(self, points, indices=None, radius=0.1, min_neighbors=5):
        if self.verbose:
            vprint("Applying radius-based filtering...")
            t0 = time.time()
        n = len(points)
        if indices is None:
            indices = np.arange(n)
        nbrs = NearestNeighbors(radius=radius).fit(points)
        _, nbr_indices = nbrs.radius_neighbors(points)
        neighbor_counts = np.array([len(nb) - 1 for nb in nbr_indices])
        mask = neighbor_counts >= min_neighbors
        filtered_points, filtered_indices = points[mask], indices[mask]
        removed = n - len(filtered_points)
        self.filtering_stats["radius_based"] = {
            "removed": removed, "remaining": len(filtered_points),
            "percentage_removed": removed / n * 100,
        }
        if self.verbose:
            vprint(f"  Removed {removed} sparse points ({removed/n*100:.1f}%) in {time.time()-t0:.2f}s")
        return filtered_points, filtered_indices

    def neighborhood_consistency_filter(self, points, indices=None, radius=0.05, max_rel_density_diff=1.0):
        if self.verbose:
            vprint("Applying neighborhood consistency filtering...")
            t0 = time.time()
        n = len(points)
        if indices is None:
            indices = np.arange(n)
        nbrs = NearestNeighbors(radius=radius).fit(points)
        _, nbr_indices = nbrs.radius_neighbors(points)
        local_density = np.array([len(idx) for idx in nbr_indices], dtype=np.float32)
        mean_neighbor_density = np.zeros(n, dtype=np.float32)
        for i, idx in enumerate(nbr_indices):
            mean_neighbor_density[i] = local_density[i] if len(idx) <= 1 else np.mean(local_density[idx])
        rel_diff = np.abs(local_density - mean_neighbor_density) / (mean_neighbor_density + 1e-6)
        mask = rel_diff <= max_rel_density_diff
        filtered_points, filtered_indices = points[mask], indices[mask]
        removed = n - len(filtered_points)
        self.filtering_stats["neighborhood_consistency"] = {
            "removed": removed, "remaining": len(filtered_points),
            "percentage_removed": removed / n * 100,
        }
        if self.verbose:
            vprint(f"  Removed {removed} inconsistent points ({removed/n*100:.1f}%) in {time.time()-t0:.2f}s")
        return filtered_points, filtered_indices

    def density_based_filtering(self, points, indices=None,
                                eps=0.05, min_samples=15, keep_largest_cluster=True,
                                second_stage_eps=None, second_stage_min_samples=None):
        if self.verbose:
            vprint("Applying density-based filtering (DBSCAN)...")
            t0 = time.time()
        n = len(points)
        if indices is None:
            indices = np.arange(n)

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters == 0:
            if self.verbose:
                vprint("  No clusters found by DBSCAN.")
            self.filtering_stats["density_based"] = {
                "removed": n, "remaining": 0, "percentage_removed": 100.0, "clusters_found": 0,
            }
            return np.empty((0, 3), dtype=points.dtype), np.empty(0, dtype=indices.dtype)

        cluster_labels = np.array([l for l in unique_labels if l != -1])

        if keep_largest_cluster:
            largest = cluster_labels[np.argmax([np.sum(labels == cl) for cl in cluster_labels])]
            first_stage_mask = labels == largest
        else:
            first_stage_mask = np.isin(labels, cluster_labels)

        filtered_points = points[first_stage_mask]
        filtered_indices = indices[first_stage_mask]

        if second_stage_eps is not None and second_stage_min_samples is not None and len(filtered_points) > 0:
            if self.verbose:
                vprint("  Running second-stage DBSCAN...")
            c2 = DBSCAN(eps=second_stage_eps, min_samples=second_stage_min_samples).fit(filtered_points)
            l2 = c2.labels_
            cl2 = [l for l in np.unique(l2) if l != -1]
            if cl2:
                largest2 = cl2[np.argmax([np.sum(l2 == cl) for cl in cl2])]
                mask2 = l2 == largest2
                filtered_points = filtered_points[mask2]
                filtered_indices = filtered_indices[mask2]

        removed = n - len(filtered_points)
        self.filtering_stats["density_based"] = {
            "removed": removed, "remaining": len(filtered_points),
            "percentage_removed": removed / n * 100, "clusters_found": int(n_clusters),
        }
        if self.verbose:
            vprint(f"  Removed {removed} sparse points ({removed/n*100:.1f}%) in {time.time()-t0:.2f}s")
        return filtered_points, filtered_indices

    def filter_pipeline(self, points,
                        stat_nb_neighbors=20, stat_std_ratio=2.0,
                        radius=0.1, min_neighbors=5,
                        apply_neighborhood_consistency=True, neigh_radius=0.05, max_rel_density_diff=1.0,
                        dbscan_eps=0.05, dbscan_min_samples=15, keep_largest_cluster=True,
                        second_stage_eps=0.03, second_stage_min_samples=10,
                        apply_statistical=True, apply_radius=True, apply_density=True):
        if self.verbose:
            vprint(f"Starting point cloud filtering pipeline...\nInitial: {len(points)} points")
        filtered_points = points.copy()
        filtered_indices = np.arange(len(points))
        self.filtering_stats = {"initial_count": len(points)}

        if apply_statistical:
            filtered_points, filtered_indices = self.statistical_outlier_removal(
                filtered_points, filtered_indices, stat_nb_neighbors, stat_std_ratio)
        if apply_radius:
            filtered_points, filtered_indices = self.radius_based_filtering(
                filtered_points, filtered_indices, radius, min_neighbors)
        if apply_neighborhood_consistency and len(filtered_points) > 0:
            filtered_points, filtered_indices = self.neighborhood_consistency_filter(
                filtered_points, filtered_indices, neigh_radius, max_rel_density_diff)
        if apply_density and len(filtered_points) > 0:
            filtered_points, filtered_indices = self.density_based_filtering(
                filtered_points, filtered_indices,
                eps=dbscan_eps, min_samples=dbscan_min_samples,
                keep_largest_cluster=keep_largest_cluster,
                second_stage_eps=second_stage_eps,
                second_stage_min_samples=second_stage_min_samples)

        total_removed = len(points) - len(filtered_points)
        self.filtering_stats["final_count"] = len(filtered_points)
        self.filtering_stats["total_removed"] = total_removed
        self.filtering_stats["total_percentage_removed"] = total_removed / len(points) * 100
        if self.verbose:
            vprint(f"Pipeline complete! Final: {len(filtered_points)} "
                  f"(removed {total_removed}, {total_removed/len(points)*100:.1f}%)")
        return filtered_points, filtered_indices


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------

def _output_path(seq_dir: str) -> str:
    return os.path.join(seq_dir, "filtered_h3d_obj_pc.obj")


def run(
    seq_dir: str,
    fit_output_dir: str = None,
    filter_pc: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Isolate object point cloud for one sequence.

    Args:
        seq_dir: Path to sequence directory.
        fit_output_dir: Dir holding the fit's meshes_smooth_obj.obj.
            Defaults to <seq_dir>/fit (the consolidated fit output).
        filter_pc: Apply the point-cloud outlier/cluster filtering pipeline.
            Default False — use all segmented object vertices directly.

    Returns:
        True if the step ran, False if skipped.
    """
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    out_path = _output_path(seq_dir)

    if fit_output_dir is None:
        fit_output_dir = fit_dir(seq_dir)   # consolidated: <seq_dir>/fit

    h3d_mesh_path = os.path.join(fit_output_dir, "meshes_smooth_obj.obj")
    classification_path = os.path.join(
        seq_dir, "render_segment",
        "vertex_classification_data_multiaxis_boundary_elev_azim_adaptive.npz",
    )
    for p in [h3d_mesh_path, classification_path]:
        if not os.path.exists(p):
            print(f"[isolate] Skipping — missing file: {p}")
            return False

    h3d_mesh = trimesh.load(h3d_mesh_path, process=False)
    mesh_classification = dict(np.load(classification_path))

    vertex_labels = mesh_classification["vertex_labels"]
    obj_mask = vertex_labels == 2
    if np.sum(obj_mask) == 0:
        obj_mask = vertex_labels != 1
    h3d_obj_indices = np.where(obj_mask)[0]
    h3d_obj_verts = h3d_mesh.vertices[obj_mask]

    if filter_pc:
        pipeline = PointCloudFilterPipeline(verbose=True)
        filtered_verts, filtered_indices = pipeline.filter_pipeline(
            h3d_obj_verts,
            stat_nb_neighbors=20, stat_std_ratio=2.0,
            radius=0.08, min_neighbors=8,
            apply_neighborhood_consistency=True, neigh_radius=0.05, max_rel_density_diff=0.7,
            dbscan_eps=0.05, dbscan_min_samples=15, keep_largest_cluster=True,
            second_stage_eps=0.03, second_stage_min_samples=10,
        )
    else:
        print(f"[isolate] Filtering disabled — using all {len(h3d_obj_verts)} segmented "
              f"object vertices (pass --filter to enable outlier/cluster filtering).")
        filtered_verts = h3d_obj_verts
        filtered_indices = np.arange(len(h3d_obj_verts))

    original_mesh_indices = h3d_obj_indices[filtered_indices]
    trimesh.PointCloud(vertices=filtered_verts).export(out_path)
    np.save(os.path.join(seq_dir, "filtered_h3d_obj_indices.npy"), original_mesh_indices)

    # Segmented object MESH: keep only the faces whose three vertices are ALL
    # object vertices, then drop the now-unreferenced (human) vertices and remap.
    # (update_vertices alone corrupts the boundary faces instead of dropping them.)
    keep = np.zeros(len(h3d_mesh.vertices), dtype=bool)
    keep[original_mesh_indices] = True
    obj_mesh = h3d_mesh.copy()
    obj_mesh.update_faces(keep[obj_mesh.faces].all(axis=1))
    obj_mesh.remove_unreferenced_vertices()
    obj_mesh.visual = trimesh.visual.ColorVisuals(
        obj_mesh,
        vertex_colors=np.full((len(obj_mesh.vertices), 4), [255, 69, 0, 255], dtype=np.uint8),
    )
    object_mesh_path = os.path.join(seq_dir, "segmented_object.obj")
    _flip_to_lrm_frame(obj_mesh)
    obj_mesh.export(object_mesh_path)

    # Final deliverable alongside segmented_object.obj: the fit's SMPL-H human mesh.
    human_mesh_path = os.path.join(seq_dir, "fitted_human.obj")
    hum_src = os.path.join(fit_output_dir, "meshes_smooth_hum.obj")
    human_saved = os.path.exists(hum_src)
    if human_saved:
        hum_mesh = trimesh.load(hum_src, process=False)
        hum_mesh.visual = trimesh.visual.ColorVisuals(
            hum_mesh,
            vertex_colors=np.full((len(hum_mesh.vertices), 4), [67, 135, 240, 255], dtype=np.uint8),
        )
        _flip_to_lrm_frame(hum_mesh)
        hum_mesh.export(human_mesh_path)

    print("[isolate] Final human and object meshes saved here:")
    print(f"    human  -> {human_mesh_path}" if human_saved
          else f"    human  -> (fit output not found: {hum_src})")
    print(f"    object -> {object_mesh_path}  "
          f"({len(obj_mesh.vertices)} verts, {len(obj_mesh.faces)} faces)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Isolate object point cloud for one sequence"
    )
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--fit_output_dir", default=None)
    parser.add_argument("--filter", action="store_true",
                        help="Apply the point-cloud outlier/cluster filtering "
                             "(default: off — use all segmented object vertices).")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    run(
        seq_dir=args.seq_dir,
        fit_output_dir=args.fit_output_dir,
        filter_pc=args.filter,
        verbose=args.verbose,
    )

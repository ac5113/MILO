"""
Human-object reconstruction metrics (CD_h / CD_o / CD_comb per sequence):

  icp       — object = filtered_h3d_obj_pc.obj (LRM point cloud). Human-aligned
              ICP (Fast-Robust-ICP) + SLERP partial-align.
  template  — object = aligned_template.obj. A single Sim(3) Procrustes of the
              combined human+object to GT; needs aligned_template to share the
              GT object's topology.

Per-sequence inputs (resolved at the top of the folder, then under
intermediate_results/ — for the template also correspondences/):
  fitted_human.obj, filtered_h3d_obj_pc.obj [icp], aligned_template.obj
  [template], gt_human.obj (SMPL-H, vertex-corresponded), gt_object.obj (faces).

The icp metric flips the object PC by [1,-1,-1] (--flip_object, on by default):
the final meshes are in the LRM frame while filtered_h3d_obj_pc.obj stays in
the fit frame (180° about X). The template object needs no flip.

FRICP (built by scripts/install_milo.sh) is needed for the icp metric only;
override with --fricp_bin / $FRICP_BIN.

Usage:
    python milo/eval/eval_results.py --data_root demo [--save_mesh]
    bash scripts/eval_results.sh --data_root demo
"""
import argparse
import glob
import json
import os
import subprocess
import tempfile

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as ScipyRotation, Slerp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_FRICP = os.path.join(REPO_ROOT, "third-party", "Fast-Robust-ICP", "build", "FRICP")

# Default per-sequence filenames (override on the CLI).
PRED_HUMAN = "fitted_human.obj"
PRED_OBJECT = "filtered_h3d_obj_pc.obj"
TEMPLATE_OBJECT = "aligned_template.obj"
GT_HUMAN = "gt_human.obj"
GT_OBJECT = "gt_object.obj"


def icp_align(source, target, fricp_bin):
    """Align `source` points to `target` with Fast-Robust-ICP (Robust ICP, the
    FRICP default). Returns the aligned source vertices, or None if FRICP fails
    (the caller then keeps the pre-ICP alignment, matching the reference)."""
    with tempfile.TemporaryDirectory(prefix="fricp_") as tmp:
        source_path = os.path.join(tmp, "source.obj")
        target_path = os.path.join(tmp, "target.obj")
        out_dir = os.path.join(tmp, "out") + os.sep
        os.makedirs(out_dir, exist_ok=True)

        trimesh.PointCloud(source).export(source_path)
        trimesh.PointCloud(target).export(target_path)

        cmd = [fricp_bin, target_path, source_path, out_dir]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[eval] FRICP failed (rc={e.returncode}) — keeping pre-ICP alignment.\n"
                  f"       stderr: {e.stderr.strip()[:300]}")
            return None

        aligned_path = os.path.join(out_dir, "m3reg_pc.ply")
        if not os.path.exists(aligned_path):
            print("[eval] FRICP produced no output — keeping pre-ICP alignment.")
            return None
        aligned = trimesh.load(aligned_path)
        if aligned is None or not hasattr(aligned, "vertices") or len(aligned.vertices) == 0:
            print("[eval] FRICP output unreadable — keeping pre-ICP alignment.")
            return None
        return np.array(aligned.vertices)


def chamfer_distance(x, y):
    """Bidirectional Chamfer distance between two point sets."""
    min_y_to_x = cKDTree(x).query(y, k=1, workers=-1)[0]
    min_x_to_y = cKDTree(y).query(x, k=1, workers=-1)[0]
    return np.mean(min_y_to_x) + np.mean(min_x_to_y)


def rigid_transform_3D(A, B):
    """Sim(3) Procrustes: scale c, rotation R, translation t mapping A onto B."""
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[-1] = -V[-1]
        R = np.dot(np.transpose(V), np.transpose(U))
    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)
    t = -np.dot(c * R, centroid_A) + centroid_B
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    return np.transpose(np.dot(c * R, np.transpose(A))) + t


def eval_chamfer_distance_hum_align(pred_human, target_human, human_faces,
                                    pred_object, target_object, object_faces,
                                    fricp_bin, sample_num=6000, is_mesh=True, seed=None,
                                    partial_align_alpha=0.6, rng=None, return_mesh=False):
    pred_human, target_human, human_faces = pred_human.copy(), target_human.copy(), human_faces.copy()
    pred_object, target_object, object_faces = pred_object.copy(), target_object.copy(), object_faces.copy()
    batch_size = pred_human.shape[0]

    np.random.seed(seed)
    if rng is None:
        rng = np.random.default_rng(seed)

    sampled_obj_pred = np.zeros((batch_size, sample_num, 3))
    sampled_obj_target = np.zeros((batch_size, sample_num, 3))

    for j in range(batch_size):
        if is_mesh:
            pred_mesh = trimesh.Trimesh(pred_object[j], object_faces, process=False, maintain_order=True)
            target_mesh = trimesh.Trimesh(target_object[j], object_faces, process=False, maintain_order=True)
            target_verts = target_mesh.vertices if len(target_mesh.vertices) == sample_num else target_mesh.sample(sample_num)
            pred_verts = pred_mesh.vertices if len(pred_mesh.vertices) == sample_num else pred_mesh.sample(sample_num)
        else:
            pred_mesh = trimesh.PointCloud(pred_object[j])
            target_mesh = trimesh.Trimesh(target_object[j], object_faces, process=False, maintain_order=True)

            target_verts = target_mesh.vertices if len(target_mesh.vertices) == sample_num else target_mesh.sample(sample_num)
            if len(pred_mesh.vertices) == sample_num:
                pred_verts = pred_mesh.vertices
            elif len(pred_mesh.vertices) < sample_num:
                pred_verts = pred_mesh.vertices[rng.choice(np.arange(len(pred_mesh.vertices)), sample_num)]
            else:
                pred_verts = pred_mesh.vertices[rng.choice(np.arange(len(pred_mesh.vertices)), sample_num, replace=False)]

        sampled_obj_pred[j] = pred_verts
        sampled_obj_target[j] = target_verts

    pred_object = sampled_obj_pred
    target_object = sampled_obj_target

    for j in range(batch_size):
        pred_mesh = np.concatenate((pred_human[j], pred_object[j]))
        target_mesh = np.concatenate((target_human[j], target_object[j]))

        s, R, t = rigid_transform_3D(pred_human[j], target_human[j])
        pred_mesh = np.transpose(np.dot(s * R, np.transpose(pred_mesh))) + t
        icp_out = icp_align(pred_mesh, target_mesh, fricp_bin)
        if icp_out is not None and np.isnan(icp_out).sum() == 0:
            pred_mesh = icp_out

        if partial_align_alpha > 0.0:
            pred_hum_post_icp = pred_mesh[:len(pred_human[j])]
            c_res, R_res, _ = rigid_transform_3D(pred_hum_post_icp, target_human[j])
            c_partial = 1.0 + partial_align_alpha * (c_res - 1.0)
            r_identity = ScipyRotation.from_matrix(np.eye(3))
            r_residual = ScipyRotation.from_matrix(R_res)
            slerp = Slerp([0, 1], ScipyRotation.concatenate([r_identity, r_residual]))
            R_partial = slerp(partial_align_alpha).as_matrix()
            centroid_pred = np.mean(pred_hum_post_icp, axis=0)
            centroid_tgt = np.mean(target_human[j], axis=0)
            pred_mesh = (
                np.transpose(np.dot(c_partial * R_partial, np.transpose(pred_mesh - centroid_pred)))
                + centroid_pred
                + partial_align_alpha * (centroid_tgt - centroid_pred)
            )

        pred_human[j], pred_object[j] = pred_mesh[:len(pred_human[j]), :], pred_mesh[len(pred_human[j]):, :]
        target_human[j], target_object[j] = target_mesh[:len(target_human[j]), :], target_mesh[len(target_human[j]):, :]

    human_chamfer_dist = []
    for j in range(batch_size):
        human_chamfer_dist.append([chamfer_distance(target_human[j], pred_human[j])])
    object_chamfer_dist = []
    for j in range(batch_size):
        object_chamfer_dist.append([chamfer_distance(target_object[j], pred_object[j])])
    comb_chamfer_dist = []
    for j in range(batch_size):
        comb_chamfer_dist.append([chamfer_distance(target_mesh, pred_mesh)])

    h = np.array(human_chamfer_dist)
    o = np.array(object_chamfer_dist)
    c = np.array(comb_chamfer_dist)
    if return_mesh:
        aligned_pred = np.concatenate((pred_human[0], pred_object[0]))  # batch is always 1 here
        return h, o, c, aligned_pred
    return h, o, c


def eval_chamfer_distance_template(pred_human, target_human, pred_object, target_object,
                                   return_mesh=False):
    """Template metric: one Sim(3) Procrustes of the combined human+object to GT,
    then bidirectional CD. No sampling — pred/GT object share the template
    topology, so the concatenated align is an ordered Kabsch."""
    pred_mesh = np.concatenate((pred_human, pred_object))
    target_mesh = np.concatenate((target_human, target_object))

    pred_mesh = rigid_align(pred_mesh, target_mesh)

    n_h = len(pred_human)
    pred_h, pred_o = pred_mesh[:n_h], pred_mesh[n_h:]
    tgt_h, tgt_o = target_mesh[:len(target_human)], target_mesh[len(target_human):]

    h = np.array([[chamfer_distance(tgt_h, pred_h)]])
    o = np.array([[chamfer_distance(tgt_o, pred_o)]])
    c = np.array([[chamfer_distance(target_mesh, pred_mesh)]])
    if return_mesh:
        return h, o, c, pred_mesh
    return h, o, c


def _resolve(seq_dir, *relpaths):
    """First existing of seq_dir/<relpath> over the given candidates, else None."""
    for rel in relpaths:
        p = os.path.join(seq_dir, rel)
        if os.path.exists(p):
            return p
    return None


def _save_combined(eval_dir, fname, verts):
    os.makedirs(eval_dir, exist_ok=True)
    out = os.path.join(eval_dir, fname)
    trimesh.PointCloud(np.asarray(verts)).export(out)
    print(f"[eval]   saved {out}")


def _eval_one(seq_dir, args, fricp_bin, fricp_ok, rng):
    """Return a list of per-metric result dicts (icp / template) for one folder."""
    name = os.path.basename(seq_dir)
    inter = "intermediate_results"

    def res(n):  # top-level, then intermediate_results/
        return _resolve(seq_dir, n, os.path.join(inter, n))

    pred_human_p = res(args.pred_human)
    gt_human_p = res(args.gt_human)
    gt_object_p = res(args.gt_object)
    icp_obj_p = res(args.pred_object)
    tmpl_obj_p = _resolve(
        seq_dir, args.template_object,
        os.path.join(inter, args.template_object),
        os.path.join("correspondences", args.template_object),
        os.path.join(inter, "correspondences", args.template_object),
    )

    # Not a sequence folder at all → skip silently.
    if not any([pred_human_p, gt_human_p, gt_object_p, icp_obj_p, tmpl_obj_p]):
        return []
    # Need the human + GT and at least one object source.
    core = {"pred_human": pred_human_p, "gt_human": gt_human_p, "gt_object": gt_object_p}
    if not all(core.values()):
        print(f"[eval] {name}: SKIP — missing {[k for k, v in core.items() if not v]}")
        return []
    if not (icp_obj_p or tmpl_obj_p):
        print(f"[eval] {name}: SKIP — no object prediction "
              f"({args.pred_object} / {args.template_object}).")
        return []

    pred_human_mesh = trimesh.load(pred_human_p, process=False)
    gt_human_mesh = trimesh.load(gt_human_p, process=False)
    gt_object_mesh = trimesh.load(gt_object_p, process=False)
    pred_human = np.array(pred_human_mesh.vertices)
    gt_human = np.array(gt_human_mesh.vertices)
    human_faces = np.array(pred_human_mesh.faces)
    gt_obj = np.array(gt_object_mesh.vertices)
    gt_obj_faces = np.array(getattr(gt_object_mesh, "faces", []))

    if len(pred_human) != len(gt_human):
        print(f"[eval] {name}: SKIP — pred_human {len(pred_human)} vs gt_human {len(gt_human)} "
              f"verts (the human Procrustes needs matching topology).")
        return []

    eval_dir = os.path.join(seq_dir, "eval")
    results = []

    # --- icp metric: object = filtered_h3d_obj_pc.obj (fit frame → flip) ---------
    if icp_obj_p:
        if not fricp_ok:
            print(f"[eval] {name}: skipping 'icp' metric — FRICP binary unavailable.")
        elif gt_obj_faces.size == 0:
            print(f"[eval] {name}: skipping 'icp' metric — gt_object has no faces (needed for sampling).")
        else:
            pred_obj = np.array(trimesh.load(icp_obj_p, process=False).vertices)
            if args.flip_object:
                pred_obj = np.array([1, -1, -1])[None, :] * pred_obj
            if len(pred_obj) == 0:
                print(f"[eval] {name}: skipping 'icp' metric — empty object point cloud.")
            else:
                sample_num = args.sample_num if args.sample_num > 0 else len(pred_obj)
                out = eval_chamfer_distance_hum_align(
                    pred_human[np.newaxis, :], gt_human[np.newaxis, :], human_faces,
                    pred_obj[np.newaxis, :], gt_obj[np.newaxis, :], gt_obj_faces,
                    fricp_bin=fricp_bin, sample_num=sample_num, is_mesh=False,
                    seed=args.seed, rng=rng, return_mesh=args.save_mesh,
                )
                h, o, c = out[:3]
                results.append({"folder_name": name, "metric": "icp",
                                "CD_h": float(h.mean()), "CD_o": float(o.mean()),
                                "CD_comb": float(c.mean()), "n_obj_points": int(sample_num)})
                if args.save_mesh:
                    _save_combined(eval_dir, "combined_aligned_icp.obj", out[3])

    # --- template metric: object = aligned_template.obj (LRM frame → no flip) -----
    if tmpl_obj_p:
        pred_tmpl = np.array(trimesh.load(tmpl_obj_p, process=False).vertices)
        if len(pred_tmpl) != len(gt_obj):
            print(f"[eval] {name}: skipping 'template' metric — aligned_template {len(pred_tmpl)} "
                  f"vs gt_object {len(gt_obj)} verts (need matching template topology).")
        else:
            out = eval_chamfer_distance_template(pred_human, gt_human, pred_tmpl, gt_obj,
                                                 return_mesh=args.save_mesh)
            h, o, c = out[:3]
            results.append({"folder_name": name, "metric": "template",
                            "CD_h": float(h.mean()), "CD_o": float(o.mean()),
                            "CD_comb": float(c.mean()), "n_obj_points": int(len(gt_obj))})
            if args.save_mesh:
                _save_combined(eval_dir, "combined_aligned_template.obj", out[3])

    return results


def _print_table(rows):
    if not rows:
        print("\nNo sequences evaluated (no folder had fitted/gt meshes plus an object "
              "prediction — filtered_h3d_obj_pc.obj or aligned_template.obj).")
        return
    name_w = max(8, max(len(r["folder_name"]) for r in rows))
    met_w = max(8, max(len(r["metric"]) for r in rows))
    header = f"{'Sequence':<{name_w}}  {'Metric':<{met_w}}  {'CD_h':>9}  {'CD_o':>9}  {'CD_comb':>9}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        print(f"{r['folder_name']:<{name_w}}  {r['metric']:<{met_w}}  "
              f"{r['CD_h']:>9.4f}  {r['CD_o']:>9.4f}  {r['CD_comb']:>9.4f}")
    print(sep)
    for metric in sorted(set(r["metric"] for r in rows)):
        mr = [r for r in rows if r["metric"] == metric]
        mh = np.mean([r["CD_h"] for r in mr])
        mo = np.mean([r["CD_o"] for r in mr])
        mc = np.mean([r["CD_comb"] for r in mr])
        label = f"MEAN {metric} (n={len(mr)})"
        print(f"{label:<{name_w + 2 + met_w}}  {mh:>9.4f}  {mo:>9.4f}  {mc:>9.4f}")
    print(sep)
    print("CD in scene units (meters); lower is better.\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="demo",
                        help="Root holding sequence subfolders (each with the pred + gt .obj files). "
                             "The root itself is also evaluated if it contains them.")
    parser.add_argument("--pred_human", default=PRED_HUMAN)
    parser.add_argument("--pred_object", default=PRED_OBJECT, help="icp-metric object (point cloud).")
    parser.add_argument("--template_object", default=TEMPLATE_OBJECT,
                        help="template-metric object (aligned template mesh).")
    parser.add_argument("--gt_human", default=GT_HUMAN)
    parser.add_argument("--gt_object", default=GT_OBJECT)
    parser.add_argument("--fricp_bin", default=None,
                        help=f"Path to the FRICP binary (default: {DEFAULT_FRICP} or $FRICP_BIN). "
                             f"Needed for the icp metric only.")
    parser.add_argument("--sample_num", type=int, default=0,
                        help="icp-metric object points to sample (0 = predicted point-cloud size).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flip_object", action=argparse.BooleanOptionalAction, default=True,
                        help="Flip the icp-metric object PC by [1,-1,-1] into the human's frame "
                             "(in-memory only; see module docstring). The template object is "
                             "never flipped.")
    parser.add_argument("--save_mesh", action="store_true",
                        help="Write the predicted combined mesh aligned to GT into <seq>/eval/ "
                             "(combined_aligned_icp.obj / combined_aligned_template.obj), one per metric.")
    parser.add_argument("--save_json", default=None, help="Optional path to write per-sequence results.")
    return parser.parse_args()


def main():
    args = parse_args()

    fricp_bin = args.fricp_bin or os.environ.get("FRICP_BIN") or DEFAULT_FRICP
    fricp_ok = os.path.isfile(fricp_bin)
    if not fricp_ok:
        print(f"[eval] NOTE: FRICP binary not found at {fricp_bin} — the 'icp' metric will be skipped.\n"
              f"       Build it with scripts/install_milo.sh (or pass --fricp_bin / set FRICP_BIN). "
              f"The 'template' metric needs no FRICP.")

    data_root = os.path.abspath(args.data_root)
    candidates = [data_root] + sorted(
        d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    for seq_dir in candidates:
        for res in _eval_one(seq_dir, args, fricp_bin, fricp_ok, rng):
            print(f"[eval] {res['folder_name']} [{res['metric']}]: CD_h={res['CD_h']:.4f}  "
                  f"CD_o={res['CD_o']:.4f}  CD_comb={res['CD_comb']:.4f}  ({res['n_obj_points']} obj pts)")
            rows.append(res)

    _print_table(rows)

    if args.save_json and rows:
        with open(args.save_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Results written to {args.save_json}")


if __name__ == "__main__":
    main()

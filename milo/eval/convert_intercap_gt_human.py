"""Produce the InterCap eval GT human from the dataset-shipped SMPL-X fit.

Converts InterCap's per-frame SMPL-X GT (`sequences/.../Mesh/<frame>_second.obj`,
10475 v) to SMPL-H (6890 v, the topology the eval's ordered Procrustes needs) via
the SMPL-X `transfer_model` recipe (deformation transfer, then a gendered LBFGS
parameter fit; https://github.com/vchoutas/smplx/tree/main/transfer_model) and
writes `gt_human.obj` into the prepared out_root. One SMPL-X mesh is shared by
all cameras of a (subj,obj,seg,frame), so only unique meshes are fitted.

`prepare_dataset.py intercap` calls `generate_gt_human()` from here; the file is
also runnable standalone (re-fit only gt_human, or partition across GPUs with
--num_batches/--idx).

Assets: the registration-gated `smplx2smplh_deftrafo_setup.pkl` goes under
`_DATA/transfer_data/` (scripts/setup_transfer_data.sh, or --transfer_data); the
gendered SMPL-H model comes from `_DATA/body_models/smplh/<gender>/model.npz`
(scripts/download_models.sh).

Usage (GPU; after prepare_dataset.py intercap):
  python milo/eval/convert_intercap_gt_human.py \
      --raw_root /path/to/InterCap --out_root <data_root>
"""

import argparse
import os
import pickle

import numpy as np
import torch
import trimesh

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TRANSFER_DATA = os.path.join(REPO_ROOT, "_DATA", "transfer_data")
SMPLH_DIR = os.path.join(REPO_ROOT, "_DATA", "body_models", "smplh")

# InterCap subject -> gender (fixed per subject; SMPL-H has no neutral).
INTERCAP_GENDERS = {
    "01": "male", "02": "male", "03": "female", "04": "male", "05": "male",
    "06": "female", "07": "female", "08": "female", "09": "female", "10": "male",
}


def parse_split(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(line.split("/"))  # subj/obj/seg/Frames_CamN/color/frame
    return entries


def load_def_matrix(path):
    """SMPL-X -> SMPL-H deformation-transfer matrix (6890, 10475)."""
    with open(path, "rb") as f:
        setup = pickle.load(f, encoding="latin1")
    mtx = setup.get("mtx", setup.get("matrix"))
    if hasattr(mtx, "todense"):
        mtx = mtx.todense()
    mtx = np.asarray(mtx, dtype=np.float32)
    if mtx.shape[1] == 10475 * 2:  # vertex+normal columns -> keep the vertex half
        mtx = mtx[:, :10475]
    return torch.tensor(mtx, dtype=torch.float32)


def verts(path):
    return np.asarray(trimesh.load(path, process=False, maintain_order=True).vertices, np.float64)


def fit_smplh(target, model, maxiters, summary_steps=0):
    """LBFGS-fit gendered SMPL-H params to target verts (B,6890,3); return fitted verts."""
    B, dev = target.shape[0], target.device
    p = {k: torch.zeros([B, n], device=dev, requires_grad=True)
         for k, n in (("global_orient", 3), ("body_pose", 63), ("betas", 10),
                      ("left_hand_pose", 45), ("right_hand_pose", 45), ("transl", 3))}
    opt = torch.optim.LBFGS(list(p.values()), lr=1.0, max_iter=20,
                            tolerance_grad=1e-6, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        out = model(return_verts=True, **p)
        loss = torch.nn.functional.mse_loss(out.vertices, target)
        loss.backward()
        return loss

    for i in range(maxiters // 20):
        loss = opt.step(closure)
        if summary_steps and (i + 1) % summary_steps == 0:
            print(f"    lbfgs {(i + 1) * 20}/{maxiters} loss={loss.item():.6e}", flush=True)
    with torch.no_grad():
        return model(return_verts=True, **{k: v.detach() for k, v in p.items()}).vertices


def generate_gt_human(entries, raw_root, out_root, transfer_data=DEFAULT_TRANSFER_DATA,
                      gpu=0, maxiters=200, batch=32, num_batches=1, idx=0, overwrite=False):
    """Write gt_human.obj into each sequence dir under out_root, from InterCap's
    shipped SMPL-X GT fitted to SMPL-H. `entries` are split parts
    (subj/obj/seg/Frames_CamN/color/frame). Called by prepare_dataset.py and by
    main() below. Returns the number of gt_human.obj files written."""
    # unique (subj,obj,seg,frame) -> camera dir-names (one SMPL-X mesh per segment-frame)
    groups = {}
    for subj, obj, seg, cam, _, frame in entries:
        groups.setdefault((subj, obj, seg, frame), []).append(cam)
    keys = sorted(groups)[idx::num_batches]  # this process's slice

    def seq_dir(subj, obj, seg, cam, frame):
        return os.path.join(out_root, "__".join([subj, obj, seg, cam, "color", frame]))

    def mesh_done(k):
        subj, obj, seg, frame = k
        return all(os.path.exists(os.path.join(seq_dir(subj, obj, seg, cam, frame), "gt_human.obj"))
                   for cam in groups[k])

    pending = keys if overwrite else [k for k in keys if not mesh_done(k)]
    print(f"[gt_human] {len(keys)} unique meshes (slice {idx}/{num_batches}), "
          f"{len(pending)} pending", flush=True)
    if not pending:  # nothing to fit — don't require smplx / the gated deftrafo asset
        return 0

    # load the fit dependencies only when there is work to do
    from smplx import SMPLH
    from smplx.utils import Struct
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    deftrafo = os.path.join(transfer_data, "smplx2smplh_deftrafo_setup.pkl")
    if not os.path.exists(deftrafo):
        raise SystemExit(
            f"[gt_human] missing {deftrafo}\n"
            f"  the SMPL-X->SMPL+H deformation-transfer setup is registration-gated "
            f"(not in the smplx git repo). Get it via scripts/setup_transfer_data.sh "
            f"or from https://smpl-x.is.tue.mpg.de/ (see the module docstring).")

    def_matrix = load_def_matrix(deftrafo).to(dev)
    models = {}  # (gender, batch_size) -> smplh model

    def get_model(gender, bs):
        # Fit target uses the gendered SMPL-H that MILO already ships
        # (_DATA/body_models/smplh/<gender>/model.npz, from download_models.sh) —
        # no separate transfer_data body model needed. The npz carries no MANO hand
        # PCA, so zero-fill it for use_pca=False / flat hands (egoego/ImhdBodyModel style).
        if (gender, bs) not in models:
            npz = os.path.join(SMPLH_DIR, gender, "model.npz")
            if not os.path.exists(npz):
                raise SystemExit(f"[gt_human] missing SMPL-H model {npz}\n"
                                 f"  run scripts/download_models.sh (SMPL-H).")
            ds = Struct(**dict(np.load(npz, allow_pickle=True)))
            ds.hands_componentsl = np.zeros((0)); ds.hands_componentsr = np.zeros((0))
            ds.hands_meanl = np.zeros((15 * 3)); ds.hands_meanr = np.zeros((15 * 3))
            models[(gender, bs)] = SMPLH(
                npz, model_type="smplh", data_struct=ds, num_betas=10, batch_size=bs,
                use_pca=False, flat_hand_mean=True).to(dev)
        return models[(gender, bs)]

    # group pending meshes by gender, fit in chunks
    by_gender = {}
    for k in pending:
        by_gender.setdefault(INTERCAP_GENDERS.get(k[0], "neutral"), []).append(k)

    written, skipped_cams = 0, 0
    smplh_faces = None
    for gender, ks in by_gender.items():
        if gender not in ("male", "female"):
            print(f"[warn] subject gender {gender!r} unsupported by SMPL-H; skipping {len(ks)} meshes")
            continue
        for c0 in range(0, len(ks), batch):
            chunk = ks[c0:c0 + batch]
            smplx_v, ship_obj_mean = [], []
            for subj, obj, seg, frame in chunk:
                base = os.path.join(raw_root, "sequences", subj, obj, seg, "Mesh")
                smplx_v.append(verts(os.path.join(base, f"{frame}_second.obj")))
                ship_obj_mean.append(verts(os.path.join(base, f"{frame}_second_obj.ply")).mean(0))
            tv = torch.tensor(np.stack(smplx_v), dtype=torch.float32).to(dev)
            target = torch.einsum("mn,bni->bmi", [def_matrix, tv])  # (B,6890,3) world frame
            model = get_model(gender, len(chunk))
            fitted = fit_smplh(target, model, maxiters).cpu().numpy().astype(np.float64)
            if smplh_faces is None:
                smplh_faces = np.asarray(model.faces, dtype=np.int64)

            for j, (subj, obj, seg, frame) in enumerate(chunk):
                world_h = fitted[j]
                for cam in groups[(subj, obj, seg, frame)]:
                    sd = seq_dir(subj, obj, seg, cam, frame)
                    gt_obj_p = os.path.join(sd, "gt_object.obj")
                    if not os.path.exists(gt_obj_p):
                        skipped_cams += 1
                        continue
                    # place the world-frame human into gt_object's frame (mine = shipped + t;
                    # t == 0 when gt_object is the shipped world-frame mesh, as prepare writes)
                    t = verts(gt_obj_p).mean(0) - ship_obj_mean[j]
                    trimesh.Trimesh(vertices=world_h + t, faces=smplh_faces, process=False) \
                        .export(os.path.join(sd, "gt_human.obj"))
                    written += 1
            print(f"[gt_human] {gender} {c0 + len(chunk)}/{len(ks)} fitted "
                  f"({written} gt_human written)", flush=True)

    print(f"[gt_human] done: {written} gt_human.obj written"
          + (f", {skipped_cams} cams skipped (no gt_object)" if skipped_cams else ""), flush=True)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw_root", required=True, help="InterCap release root")
    ap.add_argument("--out_root", required=True, help="Prepared data_root (from prepare_dataset.py)")
    ap.add_argument("--split", default=os.path.join(REPO_ROOT, "milo", "eval", "splits", "intercap_test.txt"))
    ap.add_argument("--transfer_data", default=DEFAULT_TRANSFER_DATA,
                    help="dir holding smplx2smplh_deftrafo_setup.pkl; "
                         "default _DATA/transfer_data (see scripts/setup_transfer_data.sh)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--maxiters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32, help="meshes per gendered LBFGS fit")
    ap.add_argument("--num_batches", type=int, default=1, help="partition unique meshes across processes")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    generate_gt_human(parse_split(args.split), args.raw_root, args.out_root,
                      transfer_data=args.transfer_data, gpu=args.gpu, maxiters=args.maxiters,
                      batch=args.batch, num_batches=args.num_batches, idx=args.idx,
                      overwrite=args.overwrite)


if __name__ == "__main__":
    main()

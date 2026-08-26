"""
pipeline/steps/init_smpl.py

Single-image SMPL-H initialization. Runs, on the one input image:
  - ViTDet (Detectron2)  -> person bounding box
  - HMR2 (4D-Humans)     -> body pose / betas / global orient + weak-persp camera
  - ViTPose (wholebody)  -> 133 2D keypoints (body + hands)
  - HaMeR                -> per-hand MANO pose (15 joints)

and writes a single <seq_dir>/milo_init.npz holding the SMPL-H init the fit
consumes. The body camera translation uses focal = 0.5*(H+W) with the principal
point at the image center, matching the fit's default intrinsics.

Standalone usage:
    python -m milo.pipeline.steps.init_smpl --seq_dir /path/to/seq

Module usage:
    from milo.pipeline.steps.init_smpl import run
    run(seq_dir="/path/to/seq")
"""

import argparse
import glob
import os

from milo.pipeline.steps._common import (
    build_vitdet_detector,
    detect_people,
    get_device,
    inject_third_party_paths,
    suppress_stdout,
)
from milo.pipeline.steps._log import set_verbose

inject_third_party_paths()


def _input_image(seq_dir: str) -> str:
    """First *.jpg in seq_dir that is not a mask (_human/_object)."""
    jpgs = sorted(
        f for f in glob.glob(os.path.join(seq_dir, "*.jpg"))
        if not (f.endswith("_human.jpg") or f.endswith("_object.jpg"))
    )
    return jpgs[0] if jpgs else None


def run(seq_dir: str, hamer_checkpoint: str = None, verbose: bool = False) -> bool:
    """Run single-image SMPL-H init for one sequence → <seq_dir>/milo_init.npz."""
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    out_path = os.path.join(seq_dir, "milo_init.npz")

    img_path = _input_image(seq_dir)
    if img_path is None:
        print(f"[init_smpl] Skipping — no input .jpg in {seq_dir}")
        return False

    import cv2
    import numpy as np
    import torch

    # HMR2 (body)
    from hmr2.models import load_hmr2, download_models as download_hmr2, DEFAULT_CHECKPOINT as HMR2_CKPT
    from hmr2.datasets.vitdet_dataset import ViTDetDataset as HMR2Dataset
    # HaMeR (hands)
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT as HAMER_CKPT, download_models as download_hamer
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset as HaMeRDataset
    from vitpose_model import ViTPoseModel

    device = get_device()

    # ---- load models (skip downloads when checkpoints already present;
    # hamer's download_models would otherwise re-fetch a 5.7GB demo tarball) ----
    if not os.path.exists(HMR2_CKPT):
        download_hmr2()
    hmr2_model, hmr2_cfg = load_hmr2(HMR2_CKPT)
    hmr2_model = hmr2_model.to(device).eval()

    _hamer_ckpt = hamer_checkpoint or HAMER_CKPT
    if not os.path.exists(_hamer_ckpt):
        download_hamer(CACHE_DIR_HAMER)
    hamer_model, hamer_cfg = load_hamer(_hamer_ckpt)
    hamer_model = hamer_model.to(device).eval()

    cpm = ViTPoseModel(device)

    detector = build_vitdet_detector()

    img_cv2 = cv2.imread(img_path)
    H, W = img_cv2.shape[:2]
    img_rgb = img_cv2[:, :, ::-1]

    # ---- person detection (ViTDet) ----
    boxes_all, scores = detect_people(detector, img_cv2)

    def _save(betas, body_pose_aa, global_orient_aa, left_aa, right_aa, cam_trans,
              vitpose_kp, is_valid):
        np.savez(
            out_path,
            betas=np.asarray(betas, np.float32),
            body_pose=np.asarray(body_pose_aa, np.float32),        # (23,3)
            global_orient=np.asarray(global_orient_aa, np.float32),  # (3,)
            left_hand_pose=np.asarray(left_aa, np.float32),        # (15,3)
            right_hand_pose=np.asarray(right_aa, np.float32),      # (15,3)
            cam_trans=np.asarray(cam_trans, np.float32),           # (3,) PHALP convention
            vitpose_keypoints=np.asarray(vitpose_kp, np.float32),  # (133,3)
            img_h=H, img_w=W, valid=bool(is_valid),
        )
        print(f"[init_smpl] Done → {out_path} (valid={is_valid})")

    if len(boxes_all) == 0:
        print(f"[init_smpl] No person detected in {os.path.basename(img_path)} — writing fallback init.")
        eye15 = np.repeat(np.eye(3)[None], 15, axis=0)
        _to_aa = lambda R: np.stack([cv2.Rodrigues(x)[0].squeeze() for x in R], axis=0)
        left_aa = _to_aa(eye15); left_aa[:, 1:] *= -1
        _save(np.zeros(10), np.zeros((23, 3)), np.array([np.pi, 0, 0]),
              left_aa, _to_aa(eye15), np.array([0, 0, 2.5]),
              np.zeros((133, 3)), is_valid=False)
        return True

    best = int(scores.argmax())
    person_box = boxes_all[best:best + 1]  # (1,4) xyxy

    # ---- HMR2 body ----
    hds = HMR2Dataset(hmr2_cfg, img_cv2, person_box)
    with suppress_stdout():  # mute third-party ViTDetDataset per-crop debug print
        hbatch = recursive_to(
            next(iter(torch.utils.data.DataLoader(hds, batch_size=1, shuffle=False))), device
        )
    with torch.no_grad():
        hout = hmr2_model(hbatch)
    body_pose = hout["pred_smpl_params"]["body_pose"][0].cpu().numpy()        # (23,3,3)
    global_orient = hout["pred_smpl_params"]["global_orient"][0].cpu().numpy()  # (1,3,3)
    betas = hout["pred_smpl_params"]["betas"][0].cpu().numpy()                # (10,)
    pred_cam = hout["pred_cam"][0].cpu().numpy()                              # (3,) [s,tx,ty]
    box_center = hbatch["box_center"][0].cpu().numpy()                        # (2,)
    box_size = float(hbatch["box_size"][0].cpu().numpy())                     # scalar (crop side, px)

    # weak-perspective -> full-image translation (export_phalp convention; focal=0.5*(H+W))
    focal = 0.5 * (H + W)
    s, ctx, cty = pred_cam
    tz = 2 * focal / (box_size * s + 1e-6)
    tx = ctx + tz / focal * (box_center[0] - W / 2)
    ty = cty + tz / focal * (box_center[1] - H / 2)
    cam_trans = np.array([tx, ty, tz])

    # ---- ViTPose 133 wholebody keypoints ----
    vitposes_out = cpm.predict_pose(
        img_rgb, [np.concatenate([person_box, scores[best:best + 1, None]], axis=1)]
    )
    vitpose_kp = vitposes_out[0]["keypoints"]  # (133,3)

    # ---- HaMeR hands (default identity if not detected) ----
    left_hand_pose = np.repeat(np.eye(3)[None], 15, axis=0)
    right_hand_pose = np.repeat(np.eye(3)[None], 15, axis=0)

    hand_bboxes, right_flags = [], []
    for hand_kp, is_right in [(vitpose_kp[-42:-21], 0), (vitpose_kp[-21:], 1)]:
        v = hand_kp[:, 2] > 0.5
        if v.sum() > 3:
            hand_bboxes.append([hand_kp[v, 0].min(), hand_kp[v, 1].min(),
                                hand_kp[v, 0].max(), hand_kp[v, 1].max()])
            right_flags.append(is_right)

    if hand_bboxes:
        boxes = np.stack(hand_bboxes)
        rights = np.stack(right_flags)
        hand_ds = HaMeRDataset(hamer_cfg, img_rgb, boxes, rights, rescale_factor=2.0)
        with suppress_stdout():  # mute third-party ViTDetDataset per-crop debug print
            hand_batches = list(
                torch.utils.data.DataLoader(hand_ds, batch_size=len(hand_bboxes), shuffle=False)
            )
        for batch in hand_batches:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = hamer_model(batch)
            box_center2 = batch["box_center"]
            box_size2 = batch["box_size"]
            rt = batch["right"]
            pk2d = out["pred_keypoints_2d"].clone()
            pk2d[:, :, 0] = (2 * rt[:, None] - 1) * pk2d[:, :, 0]
            pk2d = (pk2d * box_size2[:, None, None] + box_center2[:, None]).cpu().numpy()
            hand_pose = out["pred_mano_params"]["hand_pose"].cpu().numpy()  # (N,15,3,3)
        for i in range(len(hand_bboxes)):
            if right_flags[i]:
                right_hand_pose = hand_pose[i]
                vitpose_kp[-21:, :2] = pk2d[i]
            else:
                left_hand_pose = hand_pose[i]
                vitpose_kp[-42:-21, :2] = pk2d[i]

    # ---- rotation matrices -> axis-angle (with left-hand flip; export_phalp convention) ----
    _to_aa = lambda R: np.stack([cv2.Rodrigues(x)[0].squeeze() for x in R], axis=0)
    body_pose_aa = _to_aa(body_pose)                       # (23,3)
    global_orient_aa = cv2.Rodrigues(global_orient.squeeze())[0].squeeze()  # (3,)
    left_hand_pose_aa = _to_aa(left_hand_pose)             # (15,3)
    left_hand_pose_aa[:, 1:] *= -1                          # l-r flip in aa
    right_hand_pose_aa = _to_aa(right_hand_pose)           # (15,3)

    _save(betas, body_pose_aa, global_orient_aa, left_hand_pose_aa, right_hand_pose_aa,
          cam_trans, vitpose_kp, is_valid=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image SMPL-H init (PHALP-free)")
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="HaMeR checkpoint (default: bundled)")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, hamer_checkpoint=args.checkpoint, verbose=args.verbose)

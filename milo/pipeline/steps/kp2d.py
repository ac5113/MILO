"""
milo/pipeline/steps/kp2d.py

2D keypoint step. Runs ViTDet person detection + ViTPose wholebody once per
rendered view, then produces:
  - keypoints/<base>.npy                (25,3) COCO25 body keypoints
  - keypoints_viz/<base>.png            body skeleton overlay
  - kp2d_hand/<base>_{left,right}.npy   (N,21,2) HaMeR hand keypoints
  - kp2d_hand/<base>_kp2d.jpg           hand keypoint overlay

`triangulate` consumes these outputs.

Standalone usage:
    python -m milo.pipeline.steps.kp2d --seq_dir /path/to/seq
"""

import argparse
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

# COCO25 skeleton edges for the overlay viz.
_KEYPOINT_EDGES = [
    [1, 0], [0, 15], [0, 16], [15, 17], [16, 18],
    [1, 2], [1, 5], [1, 8],
    [2, 3], [3, 4], [2, 17],
    [5, 6], [6, 7], [5, 18],
    [8, 9], [8, 12],
    [9, 10], [10, 11], [11, 22], [11, 24],
    [12, 13], [13, 14], [14, 19], [14, 21],
    [19, 20], [22, 23],
]


def _remap_wholebody_to_coco25(all_kp, np):
    """133 ViTPose wholebody keypoints → (25,3) OpenPose COCO25 body."""
    kp = np.zeros((25, 3))
    kp[0] = all_kp[0]
    kp[1, :2] = (all_kp[5, :2] + all_kp[6, :2]) / 2
    kp[1, 2] = (all_kp[5, 2] + all_kp[6, 2]) / 2
    kp[2] = all_kp[6];  kp[3] = all_kp[8];  kp[4] = all_kp[10]
    kp[5] = all_kp[5];  kp[6] = all_kp[7];  kp[7] = all_kp[9]
    kp[8, :2] = (all_kp[11, :2] + all_kp[12, :2]) / 2
    kp[8, 2] = (all_kp[11, 2] + all_kp[12, 2]) / 2
    kp[9] = all_kp[12]; kp[10] = all_kp[14]; kp[11] = all_kp[16]
    kp[12] = all_kp[11]; kp[13] = all_kp[13]; kp[14] = all_kp[15]
    kp[15] = all_kp[2]; kp[16] = all_kp[1]; kp[17] = all_kp[4]; kp[18] = all_kp[3]
    kp[19:22] = all_kp[17:20]
    kp[22:25] = all_kp[20:23]
    return kp


def run(
    seq_dir: str,
    checkpoint: str = None,
    rescale_factor: float = 2.0,
    save_mesh: bool = False,
    verbose: bool = False,
) -> bool:
    """Run merged ViTPose-body + HaMeR-hands 2D keypoint extraction for one sequence."""
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    kp_dir = os.path.join(seq_dir, "keypoints")
    viz_dir = os.path.join(seq_dir, "keypoints_viz")
    hand_dir = os.path.join(seq_dir, "kp2d_hand")

    renders_dir = os.path.join(seq_dir, "render_segment", "renders")
    if not os.path.isdir(renders_dir):
        print(f"[kp2d] Skipping — renders dir not found: {renders_dir}")
        return False

    import cv2
    import numpy as np
    import torch

    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT, download_models
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from vitpose_model import ViTPoseModel

    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT

    download_models(CACHE_DIR_HAMER)
    hamer_model, model_cfg = load_hamer(checkpoint)
    device = get_device()
    hamer_model = hamer_model.to(device).eval()

    detector = build_vitdet_detector()
    cpm = ViTPoseModel(device)

    os.makedirs(kp_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(hand_dir, exist_ok=True)

    palette = np.array([
        [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
        [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
        [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
        [128, 128, 128], [255, 255, 0], [0, 255, 255], [255, 0, 255], [128, 0, 128],
    ])
    kp_colors = palette[[16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 14, 14, 14, 14, 14, 14]]
    link_colors = palette[[0, 0, 0, 0, 0, 9, 9, 9, 7, 7, 0, 9, 9, 0, 16, 16, 16, 16, 14, 14, 16, 16, 14, 14, 14, 14]]

    render_files = sorted(
        f for f in os.listdir(renders_dir)
        if "rend_img_obj_e" in f and f.endswith((".png", ".jpg", ".jpeg")) and ".npy" not in f
    )

    print(f"[kp2d] Running on {seq_dir}")
    for render_name in render_files:
        base = render_name.rsplit(".", 1)[0]
        img_cv2 = cv2.imread(os.path.join(renders_dir, render_name))
        if img_cv2 is None:
            continue
        img_rgb = img_cv2[:, :, ::-1]

        # ---- single ViTDet + ViTPose pass ----
        pred_bboxes, pred_scores = detect_people(detector, img_cv2)
        vitposes_out = []
        if len(pred_bboxes) > 0:
            vitposes_out = cpm.predict_pose(
                img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
            )

        # ================= BODY (COCO25) =================
        kp2d_body = np.zeros((25, 3))
        if vitposes_out and vitposes_out[0]:
            kp2d_body = _remap_wholebody_to_coco25(vitposes_out[0]["keypoints"], np)
        np.save(os.path.join(kp_dir, f"{base}.npy"), kp2d_body)

        vis = img_cv2.copy()
        h, w = vis.shape[:2]
        for kid, (kpt, sc) in enumerate(zip(kp2d_body[:, :2], kp2d_body[:, 2])):
            if sc > 0.3:
                cv2.circle(vis, (int(kpt[0]), int(kpt[1])), 4, tuple(int(c) for c in kp_colors[kid]), -1)
        for sk_id, sk in enumerate(_KEYPOINT_EDGES):
            x1, y1, s1 = int(kp2d_body[sk[0], 0]), int(kp2d_body[sk[0], 1]), kp2d_body[sk[0], 2]
            x2, y2, s2 = int(kp2d_body[sk[1], 0]), int(kp2d_body[sk[1], 1]), kp2d_body[sk[1], 2]
            if s1 > 0.3 and s2 > 0.3 and 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
                cv2.line(vis, (x1, y1), (x2, y2), tuple(int(c) for c in link_colors[sk_id]), 2)
        cv2.imwrite(os.path.join(viz_dir, render_name), vis)

        # ================= HANDS (HaMeR) =================
        bboxes, is_right = [], []
        for vitposes in vitposes_out:
            for hand_kp, hand_right in [(vitposes["keypoints"][-42:-21], 0), (vitposes["keypoints"][-21:], 1)]:
                valid = hand_kp[:, 2] > 0.5
                if valid.sum() > 3:
                    bboxes.append([hand_kp[valid, 0].min(), hand_kp[valid, 1].min(),
                                   hand_kp[valid, 0].max(), hand_kp[valid, 1].max()])
                    is_right.append(hand_right)
        if not bboxes:
            continue

        hds = ViTDetDataset(model_cfg, img_cv2, np.stack(bboxes), np.stack(is_right),
                            rescale_factor=rescale_factor)
        all_kp2d = []
        with suppress_stdout():  # mute third-party ViTDetDataset per-crop debug print
            hand_batches = list(
                torch.utils.data.DataLoader(hds, batch_size=8, shuffle=False, num_workers=0)
            )
        for batch in hand_batches:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = hamer_model(batch)
            right_batch = batch["right"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            pred_kp2d = out["pred_keypoints_2d"].detach().clone()
            pred_kp2d[:, :, 0] = (2 * right_batch[:, None] - 1) * pred_kp2d[:, :, 0]
            pred_kp2d = (pred_kp2d * box_size[:, None, None] + box_center[:, None]).cpu().numpy()
            for n in range(batch["img"].shape[0]):
                all_kp2d.append({"is_right": right_batch[n].item(), "keypoints": pred_kp2d[n]})

        if all_kp2d:
            img_kp = img_cv2.copy()
            for d in all_kp2d:
                color = (0, 0, 255) if d["is_right"] == 1 else (255, 0, 0)
                for kp in d["keypoints"]:
                    cv2.circle(img_kp, (int(kp[0]), int(kp[1])), 3, color, -1)
            cv2.imwrite(os.path.join(hand_dir, f"{base}_kp2d.jpg"), img_kp)
            right_kps = [d["keypoints"] for d in all_kp2d if d["is_right"] == 1]
            left_kps = [d["keypoints"] for d in all_kp2d if d["is_right"] == 0]
            if right_kps:
                np.save(os.path.join(hand_dir, f"{base}_right.npy"), np.stack(right_kps))
            if left_kps:
                np.save(os.path.join(hand_dir, f"{base}_left.npy"), np.stack(left_kps))

    print(f"[kp2d] Done → {kp_dir} + {hand_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merged ViTPose-body + HaMeR-hands 2D keypoints for one sequence")
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir,
        checkpoint=args.checkpoint, rescale_factor=args.rescale_factor, verbose=args.verbose)

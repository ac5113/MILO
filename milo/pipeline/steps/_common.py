"""
pipeline/steps/_common.py

Torch-free-at-import shared helpers: input-image discovery, and the vendored
ViTDet + ViTPose boilerplate used by the kp2d / init_smpl steps.
"""

import contextlib
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))

# Substrings that mark a file as a mask / viz output rather than the input image.
MASK_SKIP = ("human", "object", "mask", "viz", "seg", "depth", "normal")


def find_image(seq_dir: str):
    """First input image in seq_dir (skips mask / viz files), or None."""
    candidates = [
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        for p in sorted(glob.glob(os.path.join(seq_dir, ext)))
        if not any(s in os.path.basename(p).lower() for s in MASK_SKIP)
    ]
    return candidates[0] if candidates else None


def flip_to_lrm_frame(mesh):
    """Rotate a fit-frame mesh 180° about X (negate y, z) so the exported
    deliverable matches the LRM mesh (full_img_textured.glb) orientation — the
    fit saves its meshes 180°-flipped from the LRM frame. A proper rotation,
    so winding / normals are kept."""
    mesh.vertices[:, 1] *= -1
    mesh.vertices[:, 2] *= -1
    return mesh

# ViTDet (Detectron2) checkpoint bundled with the hamer demo config.
_VITDET_CKPT = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/"
    "f328730692/model_final_f05665.pkl"
)


def inject_third_party_paths() -> None:
    """Put the vendored hamer + ViTPose packages on sys.path so `hamer.*`,
    `hmr2.*` and `from vitpose_model import ViTPoseModel` resolve."""
    for _p in (
        os.path.join(REPO_ROOT, "third-party", "hamer"),
        os.path.join(REPO_ROOT, "third-party", "ViTPose"),
    ):
        if _p not in sys.path:
            sys.path.insert(0, _p)


def get_device():
    import torch
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@contextlib.contextmanager
def suppress_stdout():
    """Mute stdout inside the block. Used to silence the per-crop `downsampling_factor=`
    debug print that hamer/hmr2's vendored ViTDetDataset.__getitem__ emits, without
    editing (unshippable) the third-party libraries. Scope it to the dataloader fetch
    only — that is the sole place __getitem__ runs (all callers use num_workers=0)."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def build_vitdet_detector(score_thresh: float = 0.25):
    """Build the ViTDet (cascade_mask_rcnn_vitdet_h) person detector that ships
    with hamer. `inject_third_party_paths()` must have run first."""
    from pathlib import Path
    from detectron2.config import LazyConfig
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    import hamer as _hamer_pkg

    cfg_path = Path(_hamer_pkg.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    cfg = LazyConfig.load(str(cfg_path))
    cfg.train.init_checkpoint = _VITDET_CKPT
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = score_thresh
    return DefaultPredictor_Lazy(cfg)


def detect_people(detector, img_bgr, score_thresh: float = 0.5):
    """Run the detector and return (boxes_xyxy (N,4), scores (N,)) for persons (class 0)."""
    inst = detector(img_bgr)["instances"]
    valid = (inst.pred_classes == 0) & (inst.scores > score_thresh)
    return (
        inst.pred_boxes.tensor[valid].cpu().numpy(),
        inst.scores[valid].cpu().numpy(),
    )

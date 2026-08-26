"""
pipeline/steps/image_segment.py

Text-prompted human / object segmentation for both pipeline stages:
  - input masks  (--input_masks): the input image -> image_human.png / image_object.png
    at the seq root (the auto_masks step; run_lrm consumes them)
  - render views (default):       render_segment/renders/ -> per-view person and
    object masks in render_segment/segments/ (the img_segment step)

Backends (--segmenter):
  - sam3  (default) — SAM 3 concept segmentation (pip-installed in this env; see
    docs/INSTALL.md). One text prompt per class: --human_prompt and --object_prompt
    (elaborate noun phrases ground best, e.g. "a grey trolley suitcase").
  - gsam2 — Grounded-SAM-2 (GroundingDINO + SAM 2.1, third-party/Grounded-SAM-2).
    Uses the terse --object label in a combined "person. <object>." caption.

Standalone usage:
    python -m milo.pipeline.steps.image_segment --seq_dir /path/to/seq --object "chair"

Module usage:
    from milo.pipeline.steps.image_segment import run, run_input_masks
    run(seq_dir="/path/to/seq", object_label="chair")
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

from milo.pipeline.steps._common import find_image as _find_image
from milo.pipeline.steps._log import vprint, set_verbose

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_GSAM2_DIR = os.path.join(REPO_ROOT, "third-party", "Grounded-SAM-2")

DEFAULT_HUMAN_PROMPT = "a person"


def _output_dir(seq_dir: str) -> str:
    return os.path.join(seq_dir, "render_segment", "segments")


def _check_sam3() -> None:
    """Fail fast in the torch-free wrapper (before a GPU subprocess is spawned)
    when the SAM 3 backend is not installed in this env."""
    missing = [m for m in ("sam3", "ftfy") if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            f"SAM 3 backend not installed in this env (missing: {', '.join(missing)}). "
            "Install it with:\n"
            "  pip install ftfy==6.1.1\n"
            '  pip install --no-deps "git+https://github.com/facebookresearch/sam3.git"\n'
            "(--no-deps is deliberate — see docs/INSTALL.md), or use --segmenter gsam2."
        )


def _run_driver(driver_code: str, tag: str) -> bool:
    """Write a driver to a temp file, run it in the current interpreter, clean up."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(driver_code)
        driver_path = f.name
    try:
        result = subprocess.call([sys.executable, driver_path], env=os.environ.copy())
    finally:
        os.unlink(driver_path)
    if result != 0:
        print(f"[{tag}] Step failed with exit code {result}")
        return False
    return True


# ---------------------------------------------------------------------------
# SAM 3 driver template
# One template for both stages (mode: "input" | "renders"): the model is built
# once per invocation; per image set_image runs once (the heavy call) and each
# set_text_prompt only swaps text features and re-runs grounding on the cached
# image features. Instances per prompt are unioned into one binary mask.
# NOTE: keep runtime f-strings/dict literals out of this template — it is
# .format()-ed, so any literal brace would need {{}} escaping.
# ---------------------------------------------------------------------------
_SAM3_DRIVER_TEMPLATE = r"""
import os, numpy as np, torch, cv2
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

mode          = {mode!r}
seq_dir       = {seq_dir!r}
object_prompt = {object_prompt!r}
human_prompt  = {human_prompt!r}
obj_key       = {obj_key!r}
img_path      = {img_path!r}
human_out     = {human_out!r}
object_out    = {object_out!r}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam3_image_model()
model.to(DEVICE)
processor = Sam3Processor(model)

torch.inference_mode().__enter__()
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()


def segment_image(path):
    # -> (human_mask, object_mask): bool HxW arrays, or None when a prompt
    # grounded no instances in the image.
    state = processor.set_image(Image.open(path).convert("RGB"))
    results = []
    for prompt in (human_prompt, object_prompt):
        out = processor.set_text_prompt(prompt=prompt, state=state)
        masks = out["masks"]                       # [N, 1, H, W] bool instances
        if masks.shape[0] > 0:
            results.append(masks.any(dim=0).squeeze(0).cpu().numpy().astype(bool))
        else:
            results.append(None)
    return results


if mode == "input":
    human_mask, object_mask = segment_image(img_path)
    # Only write a mask that actually captured something. Writing an empty mask would
    # (a) make the auto_masks step look "done" and block a retry with a better prompt,
    # and (b) hand run_lrm a zero alpha. An empty result usually means the prompt did not
    # match — warn and skip it.
    for kind, mask, out_path, prompt in (
        ("human", human_mask, human_out, human_prompt),
        ("object", object_mask, object_out, object_prompt),
    ):
        if mask is not None and mask.any():
            cv2.imwrite(out_path, mask.astype(np.uint8) * 255)
            print("[sam3] Input " + kind + " mask -> " + out_path)
        else:
            print("[sam3] WARNING: " + kind + " mask empty — no match for prompt "
                  + repr(prompt) + " in the input image; not writing " + out_path
                  + " (provide it manually or refine the prompt).")
else:
    renders_dir = os.path.join(seq_dir, "render_segment", "renders")
    save_path   = os.path.join(seq_dir, "render_segment", "segments")
    os.makedirs(save_path, exist_ok=True)

    n_views, n_human, n_object, misses = 0, 0, 0, []
    for render_name in tqdm(sorted(os.listdir(renders_dir)), dynamic_ncols=True):
        if ".npy" in render_name:
            continue
        n_views += 1
        human_mask, object_mask = segment_image(os.path.join(renders_dir, render_name))
        if human_mask is not None:
            cv2.imwrite(os.path.join(save_path, render_name),
                        human_mask.astype(np.uint8) * 255)
            n_human += 1
        else:
            misses.append(render_name + " (human)")
        if object_mask is not None:
            cv2.imwrite(os.path.join(save_path, render_name.replace("obj", obj_key)),
                        object_mask.astype(np.uint8) * 255)
            n_object += 1
        else:
            misses.append(render_name + " (object)")

    summary = "[sam3] views=" + str(n_views) + " human=" + str(n_human) + " object=" + str(n_object)
    if misses:
        summary += "  misses: " + ", ".join(misses)
    print(summary)
    print("[sam3] Done -> " + save_path)
"""


# ---------------------------------------------------------------------------
# Grounded-SAM-2 driver templates. GSAM2 uses checkpoint paths relative to the
# Grounded-SAM-2 directory, so the drivers chdir into _GSAM2_DIR; seq_dir /
# obj_name are format()-ed in below.
# ---------------------------------------------------------------------------
_GSAM2_DRIVER_TEMPLATE = r"""
import sys, os, glob, importlib.util, cv2, torch, numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from tqdm import tqdm

gsam2_dir = {gsam2_dir!r}
os.chdir(gsam2_dir)
sys.path.insert(0, gsam2_dir)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

SAM2_CHECKPOINT       = os.path.join(gsam2_dir, "checkpoints", "sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = os.path.join(gsam2_dir, "grounding_dino", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(gsam2_dir, "gdino_checkpoints", "groundingdino_swint_ogc.pth")
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model     = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

seq_dir  = {seq_dir!r}
obj_name = {obj_name!r}

renders_dir = os.path.join(seq_dir, "render_segment", "renders")
save_path   = os.path.join(seq_dir, "render_segment", "segments")
os.makedirs(save_path, exist_ok=True)

for render_name in tqdm(sorted(os.listdir(renders_dir)), dynamic_ncols=True):
    render_path = os.path.join(renders_dir, render_name)
    if ".npy" in render_name:
        continue
    image_source, image = load_image(render_path)
    text = f"person. {{obj_name}}."

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    except AssertionError:
        continue

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    for i, cls in enumerate(labels):
        pred_mask = (masks[i] * 255).astype(np.uint8)
        if cls == "person":
            cv2.imwrite(os.path.join(save_path, render_name), pred_mask)
        else:
            cv2.imwrite(
                os.path.join(save_path, render_name.replace("obj", obj_name.replace(" ", "_"))),
                pred_mask
            )

print(f"[gsam2] Done → {{save_path}}")
"""


_GSAM2_INPUT_DRIVER_TEMPLATE = r"""
import sys, os, cv2, torch, numpy as np
from torchvision.ops import box_convert

gsam2_dir = {gsam2_dir!r}
os.chdir(gsam2_dir)
sys.path.insert(0, gsam2_dir)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

SAM2_CHECKPOINT       = os.path.join(gsam2_dir, "checkpoints", "sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = os.path.join(gsam2_dir, "grounding_dino", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(gsam2_dir, "gdino_checkpoints", "groundingdino_swint_ogc.pth")
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model     = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

img_path   = {img_path!r}
obj_name   = {obj_name!r}
human_out  = {human_out!r}
object_out = {object_out!r}

image_source, image = load_image(img_path)
text = f"person. {{obj_name}}."
sam2_predictor.set_image(image_source)
boxes, confidences, labels = predict(
    model=grounding_model, image=image, caption=text,
    box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=DEVICE
)

h, w, _ = image_source.shape
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()

human_mask  = np.zeros((h, w), dtype=bool)
object_mask = np.zeros((h, w), dtype=bool)
if len(boxes):
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    try:
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes, multimask_output=False)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        for i, cls in enumerate(labels):
            m = masks[i].astype(bool)
            if cls == "person":
                human_mask |= m
            else:
                object_mask |= m
    except AssertionError:
        print("[gsam2] WARNING: SAM2 predict failed (AssertionError) — masks may be incomplete.")

# Only write a mask that actually captured something. Writing an empty mask would (a) make
# the auto_masks step look "done" and block a retry with a better prompt, and (b) hand run_lrm
# a zero alpha. An empty result usually means the caption did not match — warn and skip it.
for kind, mask, out in (("human", human_mask, human_out), ("object", object_mask, object_out)):
    if mask.any():
        cv2.imwrite(out, (mask.astype(np.uint8) * 255))
        print(f"[gsam2] Input {{kind}} mask -> {{out}}")
    else:
        print(f"[gsam2] WARNING: {{kind}} mask empty — no match for {{text!r}} in the input "
              f"image; not writing {{out}} (provide it manually or refine --object).")
"""


def run(seq_dir: str, object_label: str = "object", segmenter: str = "sam3",
        object_prompt: str = None, human_prompt: str = DEFAULT_HUMAN_PROMPT,
        verbose: bool = False) -> bool:
    """
    Segment the rendered views of one sequence into per-view person / object masks.

    Args:
        seq_dir: Path to sequence directory.
        object_label: Short (single-word) object label — keys the object-mask
            filenames (rend_img_<label>_*.png) and, for gsam2, the detector caption.
        segmenter: "sam3" (default) or "gsam2".
        object_prompt: SAM 3 text prompt for the object (default: object_label).
            Elaborate noun phrases ground best (e.g. "a grey trolley suitcase").
        human_prompt: SAM 3 text prompt for the person (default: "a person").

    Returns:
        True if the step ran, False if skipped / failed.
    """
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    out_dir = _output_dir(seq_dir)

    renders_dir = os.path.join(seq_dir, "render_segment", "renders")
    if not os.path.isdir(renders_dir):
        print(f"[{segmenter}] Skipping — renders dir not found: {renders_dir}")
        return False

    if segmenter == "sam3":
        _check_sam3()
        object_prompt = object_prompt or object_label
        vprint(f"[sam3] object_prompt={object_prompt!r} human_prompt={human_prompt!r}")
        driver_code = _SAM3_DRIVER_TEMPLATE.format(
            mode="renders", seq_dir=seq_dir,
            object_prompt=object_prompt, human_prompt=human_prompt,
            obj_key=object_label.replace(" ", "_"),
            img_path="", human_out="", object_out="",
        )
    else:
        vprint(f"[gsam2] obj_name={object_label!r}")
        driver_code = _GSAM2_DRIVER_TEMPLATE.format(
            gsam2_dir=_GSAM2_DIR, seq_dir=seq_dir, obj_name=object_label,
        )

    print(f"[{segmenter}] Running on {seq_dir}")
    if not _run_driver(driver_code, segmenter):
        return False
    print(f"[{segmenter}] Done → {out_dir}")
    return True


def run_input_masks(seq_dir: str, object_label: str = "object", segmenter: str = "sam3",
                    object_prompt: str = None, human_prompt: str = DEFAULT_HUMAN_PROMPT,
                    verbose: bool = False) -> bool:
    """
    Generate image_human.png / image_object.png from the input image.

    Used to create the initial masks a sequence needs (the run_lrm step consumes them)
    when they are not provided. Same backend / prompt semantics as run().
    Returns True if it ran, False if skipped / failed.
    """
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    img_path = _find_image(seq_dir)
    if img_path is None:
        print(f"[{segmenter}] Skipping input masks — no input image in {seq_dir}")
        return False

    human_out = os.path.join(seq_dir, "image_human.png")
    object_out = os.path.join(seq_dir, "image_object.png")

    if segmenter == "sam3":
        _check_sam3()
        object_prompt = object_prompt or object_label
        vprint(f"[sam3] object_prompt={object_prompt!r} human_prompt={human_prompt!r}")
        driver_code = _SAM3_DRIVER_TEMPLATE.format(
            mode="input", seq_dir=seq_dir,
            object_prompt=object_prompt, human_prompt=human_prompt,
            obj_key=object_label.replace(" ", "_"),
            img_path=img_path, human_out=human_out, object_out=object_out,
        )
    else:
        vprint(f"[gsam2] obj_name={object_label!r}")
        driver_code = _GSAM2_INPUT_DRIVER_TEMPLATE.format(
            gsam2_dir=_GSAM2_DIR, img_path=img_path, obj_name=object_label,
            human_out=human_out, object_out=object_out,
        )

    print(f"[{segmenter}] Generating input masks for {seq_dir}")
    if not _run_driver(driver_code, segmenter):
        return False
    print(f"[{segmenter}] Done → {os.path.join(seq_dir, 'image_{human,object}.png')}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run human/object segmentation for one sequence (SAM 3 or Grounded-SAM-2)"
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory")
    parser.add_argument(
        "--object", default="object",
        help="Short single-word object label (e.g. 'chair') — keys the object-mask "
             "filenames and the gsam2 detector caption. Default: 'object'.",
    )
    parser.add_argument(
        "--segmenter", default="sam3", choices=["sam3", "gsam2"],
        help="Segmentation backend (default: sam3).",
    )
    parser.add_argument(
        "--object_prompt", default=None,
        help="SAM 3 text prompt for the object (default: the --object label). "
             "An elaborate noun phrase grounds best (e.g. 'a grey trolley suitcase'). "
             "Ignored by --segmenter gsam2.",
    )
    parser.add_argument(
        "--human_prompt", default=DEFAULT_HUMAN_PROMPT,
        help=f"SAM 3 text prompt for the person (default: {DEFAULT_HUMAN_PROMPT!r}). "
             "Ignored by --segmenter gsam2.",
    )
    parser.add_argument(
        "--input_masks", action="store_true",
        help="Segment the input image into image_human.png / image_object.png instead of "
             "the rendered views (used to create the initial masks when not provided).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    fn = run_input_masks if args.input_masks else run
    fn(seq_dir=args.seq_dir, object_label=args.object, segmenter=args.segmenter,
       object_prompt=args.object_prompt, human_prompt=args.human_prompt,
       verbose=args.verbose)

"""
pipeline/steps/collate.py

Finalisation step — tidy a finished sequence folder so the few deliverables sit
at the top and every intermediate artefact is tucked into intermediate_results/.

Run it once, after the pipeline has finished, on the folder that was passed to
run_pipeline.py (`demo/example`, or any sequence folder under a --data_root).

After collation a sequence folder looks like:

    <seq>/
      image.jpg                   input image            ─┐
      image_human.png             human mask              │
      image_object.png            object mask             │ inputs — stay outside
      metadata.npz                camera intrinsics       │
      template.obj                object template         │
      gt_human.obj, gt_object.obj eval ground truth      ─┘
      full_img_textured.glb       combined LRM mesh       (also stays at the top)
      fitted_human.obj            final human mesh       ─┐
      segmented_object.obj        final object mesh       │ final outputs (collated)
      aligned_template.obj        final aligned template ─┘
      intermediate_results/       everything else (renders, fit/, correspondences/, …)

Standalone usage:
    python -m milo.pipeline.steps.collate --seq_dir /path/to/seq
    python -m milo.pipeline.steps.collate --data_root /path/to/data_root

Module usage:
    from milo.pipeline.steps.collate import run
    run(seq_dir="/path/to/seq")
"""

import argparse
import os
import shutil

INTERMEDIATE_DIR = "intermediate_results"

# Final deliverables that stay at the top of the folder, collated together.
# aligned_template.obj is written by template_align into correspondences/; it is
# promoted up to sit alongside the other final meshes (see run()).
FINAL_OUTPUTS = ("fitted_human.obj", "segmented_object.obj", "aligned_template.obj")

# The combined human+object LRM mesh stays at the top, next to the final outputs.
LRM_GLB = "full_img_textured.glb"

# Fixed-name inputs that stay outside: camera intrinsics, the object template, and
# the eval ground-truth meshes. The input image and its human/object masks are kept
# by extension (every top-level image is an input; the pipeline writes none there).
INPUT_FILES = ("metadata.npz", "template.obj", "gt_human.obj", "gt_object.obj")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# Pipeline-generated top-level images (render_final outputs). The image-extension
# rule in _is_input would otherwise misclassify them as inputs, so clear_outputs
# names them explicitly to delete them on an --overwrite clear.
RENDER_OUTPUTS = ("render_human_object.png", "render_human_template.png")


def _is_input(name: str) -> bool:
    """True if `name` is something the pipeline *consumes* but never produces — an
    input image / mask, the camera intrinsics, the template, the eval GT, or a
    hidden file. These survive both collation and an --overwrite clear (except
    RENDER_OUTPUTS, which clear_outputs deletes explicitly)."""
    if name.startswith("."):
        return True                                   # leave hidden files alone
    if name in INPUT_FILES:
        return True
    if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
        return True                                   # every top-level image is an input
    return False


def _is_kept(name: str) -> bool:
    """True if `name` is an input / final output / the LRM glb (stays at the top).
    Everything else a top-level entry is treated as an intermediate to be moved."""
    if name == INTERMEDIATE_DIR:
        return True                                   # never move the bucket into itself
    if name in FINAL_OUTPUTS or name == LRM_GLB:
        return True
    return _is_input(name)


def _move(src: str, dst_dir: str) -> None:
    """Move `src` into `dst_dir`, replacing any existing entry of the same name
    (so re-running after a fresh --overwrite pipeline refreshes stale copies)."""
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst) or os.path.islink(dst):
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    shutil.move(src, dst)


def clear_outputs(seq_dir: str) -> int:
    """Remove every generated artefact from a sequence folder, keeping only the
    inputs. Used for a clean --overwrite re-run, so a new run does not inherit a
    previous run's collated intermediate_results/ or any other stale output.
    Returns the number of top-level entries removed."""
    seq_dir = os.path.abspath(seq_dir)
    if not os.path.isdir(seq_dir):
        return 0
    removed = []
    for name in sorted(os.listdir(seq_dir)):
        if _is_input(name) and name not in RENDER_OUTPUTS:
            continue
        path = os.path.join(seq_dir, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        removed.append(name)
    if removed:
        print(f"[collate] {os.path.basename(seq_dir)}: cleared {len(removed)} previous "
              f"output(s) for --overwrite: {', '.join(removed)}")
    return len(removed)


def run(seq_dir: str) -> bool:
    """Collate one sequence folder. Returns True if it ran, False if skipped."""
    seq_dir = os.path.abspath(seq_dir)
    if not os.path.isdir(seq_dir):
        print(f"[collate] Skipping — not a directory: {seq_dir}")
        return False

    # Promote the aligned template (the optional template-alignment deliverable)
    # out of correspondences/ to the top, next to the other final meshes.
    corr_aligned = os.path.join(seq_dir, "correspondences", "aligned_template.obj")
    top_aligned = os.path.join(seq_dir, "aligned_template.obj")
    if os.path.isfile(corr_aligned) and not os.path.exists(top_aligned):
        shutil.move(corr_aligned, top_aligned)
        print("[collate] aligned_template.obj -> (top level)")

    inter_dir = os.path.join(seq_dir, INTERMEDIATE_DIR)
    moved = []
    for name in sorted(os.listdir(seq_dir)):
        if _is_kept(name):
            continue
        os.makedirs(inter_dir, exist_ok=True)
        _move(os.path.join(seq_dir, name), inter_dir)
        moved.append(name)

    if moved:
        print(f"[collate] {os.path.basename(seq_dir)}: moved {len(moved)} intermediate "
              f"item(s) into {INTERMEDIATE_DIR}/: {', '.join(moved)}")
    else:
        print(f"[collate] {os.path.basename(seq_dir)}: nothing to move — already collated.")

    kept = sorted(n for n in os.listdir(seq_dir) if n != INTERMEDIATE_DIR)
    print(f"[collate] {os.path.basename(seq_dir)}: top-level deliverables: {', '.join(kept)}")
    return True


def uncollate(seq_dir: str) -> bool:
    """Inverse of run(): restore a collated folder to its pre-collate layout so the
    per-step output paths are visible again (used by the pipeline's auto-continue to
    resume a sequence that was already collated). Moves every entry from
    intermediate_results/ back to the top level and demotes aligned_template.obj back
    into correspondences/. Returns True if it un-collated, False if nothing to do."""
    seq_dir = os.path.abspath(seq_dir)
    inter_dir = os.path.join(seq_dir, INTERMEDIATE_DIR)
    if not os.path.isdir(inter_dir):
        return False

    moved = []
    for name in sorted(os.listdir(inter_dir)):
        _move(os.path.join(inter_dir, name), seq_dir)     # replaces top-level collisions
        moved.append(name)
    os.rmdir(inter_dir)                                    # now empty

    # Demote the aligned template back into correspondences/ (inverse of run()'s
    # promote), so a re-collate re-promotes it from the same place.
    top_aligned = os.path.join(seq_dir, "aligned_template.obj")
    corr_dir = os.path.join(seq_dir, "correspondences")
    if os.path.isfile(top_aligned) and os.path.isdir(corr_dir):
        _move(top_aligned, corr_dir)
        print("[collate] aligned_template.obj -> correspondences/ (uncollate)")

    print(f"[collate] {os.path.basename(seq_dir)}: un-collated {len(moved)} item(s) "
          f"from {INTERMEDIATE_DIR}/ back to top level.")
    return True


def _sequences(seq_dir: str, data_root: str) -> list:
    seqs = []
    if data_root:
        root = os.path.abspath(data_root)
        seqs += [os.path.join(root, d) for d in sorted(os.listdir(root))
                 if os.path.isdir(os.path.join(root, d))]
    if seq_dir:
        seqs.append(os.path.abspath(seq_dir))
    return seqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collate a finished sequence folder: keep the inputs, final "
                    "outputs and the combined LRM .glb at the top, and move every "
                    "intermediate artefact into intermediate_results/."
    )
    parser.add_argument("--seq_dir", default=None,
                        help="A single sequence/demo folder to collate "
                             "(e.g. demo/example).")
    parser.add_argument("--data_root", default=None,
                        help="Collate every sequence sub-folder of this root.")
    args = parser.parse_args()
    if not args.seq_dir and not args.data_root:
        parser.error("Pass --seq_dir <folder> or --data_root <root>.")
    for s in _sequences(args.seq_dir, args.data_root):
        run(s)

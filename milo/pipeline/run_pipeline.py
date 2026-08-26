"""
pipeline/run_pipeline.py

Top-level orchestrator for the full preprocessing → MILO pipeline.

Steps (in order — see ALL_STEPS / TEMPLATE_STEPS / FINALIZE_STEPS below):
  0. auto_masks  — SAM 3 masks of the input image (image_{human,object}.png; Grounded-
                   SAM-2 with --segmenter gsam2), generated only when the sequence
                   does not already provide them
  1. run_lrm     — LRM recon of the combined human+object mesh (full_img_textured.glb):
                   Hunyuan3D-2 by default, SAM 3D Objects with --lrm sam3d
  2. render      — Multi-view render of LRM mesh (PNGs + visible_vertices .npy)
  3. img_segment — SAM 3 segmentation of rendered views (Grounded-SAM-2 with
                   --segmenter gsam2)
  4. mesh_segment — Vertex classification using the segmenter masks
  5. kp2d        — 2D keypoints per view: ViTPose body (COCO25) + HaMeR hands (MANO)
  6. triangulate — Multi-view triangulation → 3D keypoints
  7. init_smpl   — SMPL-H initialization (HMR2.0 body + HaMeR hands → milo_init.npz)
  8. fit         — MILO SMPL-H optimization (fit/run_opt.py)
  9. isolate     — Final meshes + filtered object point cloud from the LRM mesh
Optional template alignment (with --template): tmpl_render, correspond, template_align.
Finalisation (appended by default): collate, render_final.

Usage — all sequences in data_root (add --seq <name> for one; steps whose outputs
exist are skipped unless --overwrite):
    python pipeline/run_pipeline.py \\
        --data_root /path/to/data_root \\
        --object "chair"

Usage — distribute over multiple GPUs (one worker per GPU, sequences split evenly):
    python pipeline/run_pipeline.py \\
        --data_root /path/to/data_root \\
        --gpus 0,1,2,3

Usage — run this worker directly (called internally by multi-GPU dispatch):
    python pipeline/run_pipeline.py \\
        --data_root /path/to/data_root \\
        --gpu 0 --batch_index 0 --num_batches 4 ...
"""

import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
FIT_DIR = os.path.join(REPO_ROOT, "milo", "fit")

# Allow `from milo.pipeline.config import ...` when invoked as a script
# (python pipeline/run_pipeline.py), where sys.path[0] is the pipeline/ dir.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from milo.pipeline.config import (
    STEP_PYTHON as _STEP_PYTHON, MILO_PYTHON, DATASET_CFG, DEFAULT_FIT_CFG,
    resolve_object, template_path, fit_dir,
)
from milo.pipeline.steps._common import find_image
from milo.pipeline.steps.collate import (
    clear_outputs, uncollate, INTERMEDIATE_DIR, run as collate_run,
)

# fmt: off
ALL_STEPS = ["auto_masks", "run_lrm", "render", "img_segment", "mesh_segment", "kp2d", "triangulate", "init_smpl", "fit", "isolate"]
# Optional template-alignment steps (paper Sec 3.6); run only when --template is given.
TEMPLATE_STEPS = ["tmpl_render", "correspond", "template_align"]
# Finalisation steps, run by default as the very last steps in this order: collate
# tidies the seq folder (final outputs at the top, the rest into intermediate_results/);
# render_final then renders the collated deliverables (human+object, human+template)
# as white-bg colored PNGs from the top-level meshes collate places. Because collate
# moves the outputs the STEP_DONE checks look for, a re-run of an already-collated
# sequence first un-collates it, resumes the missing steps, then re-collates
# (see run_sequence).
FINALIZE_STEPS = ["collate", "render_final"]
# fmt: on
VALID_STEPS = ALL_STEPS + TEMPLATE_STEPS + FINALIZE_STEPS

# ---------------------------------------------------------------------------
# Per-step completion detection — the single skip authority. Step modules never
# self-skip; the driver decides run-vs-skip from STEP_DONE (see run_sequence).
#
# Each predicate is filesystem-only (it must NOT import a step module, which would
# pull in torch) and mirrors what its step writes. They check *completeness* (the
# full view count / the final artefact), not mere existence, so a half-finished
# step is detected as not-done and re-run.
# ---------------------------------------------------------------------------
# Multi-view render grid, mirrored from steps/render.py (_render_views_cam_rot):
# 12 rotation angles, and an elevation is kept iff -90 < (angle - 180) < 90.
_ROTATION_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
EXPECTED_VIEWS = (sum(1 for a in _ROTATION_ANGLES if -90 < a - 180 < 90)
                  * len(_ROTATION_ANGLES))   # 5 elevations x 12 azimuths = 60


def _exists(seq_dir, *rel) -> bool:
    return os.path.exists(os.path.join(seq_dir, *rel))


def _n_renders(seq_dir: str) -> int:
    """Number of rendered object views (rend_img_obj images, not .npy) in
    render_segment/renders/ — matches the img_segment / kp2d definitions."""
    renders = os.path.join(seq_dir, "render_segment", "renders")
    if not os.path.isdir(renders):
        return 0
    return len([f for f in os.listdir(renders)
                if "rend_img_obj_e" in f and ".npy" not in f
                and f.lower().endswith((".png", ".jpg", ".jpeg"))])


def _renders_done(seq_dir: str, subdir: str) -> bool:
    """A render/tmpl_render dir is done when it holds the full view grid of
    visible_vertices arrays (default resolution; the _highres variant is ignored)."""
    pat = os.path.join(seq_dir, "render_segment", subdir, "visible_vertices_e*_a*.npy")
    n = sum(1 for p in glob.glob(pat) if "_highres" not in os.path.basename(p))
    return n >= EXPECTED_VIEWS


def _img_segment_done(seq_dir: str) -> bool:
    n_renders = _n_renders(seq_dir)
    if n_renders == 0:
        return False
    segments = os.path.join(seq_dir, "render_segment", "segments")
    n_segments = len(os.listdir(segments)) if os.path.isdir(segments) else 0
    return n_segments >= 2 * n_renders          # person + object mask per render


def _kp2d_done(seq_dir: str) -> bool:
    # Body keypoints: one npy per render, so count == n_renders proves every view ran.
    # Hand npys are sparse (only views with a detected hand), so only the dir is required.
    n_renders = _n_renders(seq_dir)
    if n_renders == 0:
        return False
    n_kp = len(glob.glob(os.path.join(seq_dir, "keypoints", "*.npy")))
    return n_kp >= n_renders and os.path.isdir(os.path.join(seq_dir, "kp2d_hand"))


def _render_final_done(seq_dir: str, tmpl) -> bool:
    if not _exists(seq_dir, "render_human_object.png"):
        return False
    return _exists(seq_dir, "render_human_template.png") if tmpl else True


def _input_masks_done(seq_dir: str) -> bool:
    """True if both the human and object input masks exist under any accepted name.
    Mirrors run_lrm._find_mask's candidates (image_<kind>.png / <imgbase>_<kind>.png /
    <suffix>_<kind>.png)."""
    img = find_image(seq_dir)
    img_base = os.path.splitext(os.path.basename(img))[0] if img else "image"
    suffix = img_base.split("__")[-1]

    def _has(kind):
        return any(_exists(seq_dir, n) for n in
                   (f"image_{kind}.png", f"{img_base}_{kind}.png", f"{suffix}_{kind}.png"))

    return _has("human") and _has("object")


# step -> done(seq_dir, tmpl). collate/render_final note: collate has no marker (always
# re-collates after an un-collate). correspond's marker assumes the legacy
# "correspondences" dirname — the only one the pipeline path triggers.
STEP_DONE = {
    "auto_masks":     lambda s, t: _input_masks_done(s),
    "run_lrm":        lambda s, t: _exists(s, "full_img_textured.glb"),
    "render":         lambda s, t: _renders_done(s, "renders"),
    "img_segment":    lambda s, t: _img_segment_done(s),
    "mesh_segment":   lambda s, t: _exists(s, "render_segment",
                          "vertex_classification_data_multiaxis_boundary_elev_azim_adaptive.npz"),
    "kp2d":           lambda s, t: _kp2d_done(s),
    "triangulate":    lambda s, t: _exists(s, "keypoints_3d.npy"),
    "init_smpl":      lambda s, t: _exists(s, "milo_init.npz"),
    "fit":            lambda s, t: _exists(s, "fit", "meshes_smooth_obj.obj"),
    "isolate":        lambda s, t: _exists(s, "filtered_h3d_obj_pc.obj"),
    "tmpl_render":    lambda s, t: _renders_done(s, "renders_gt"),
    "correspond":     lambda s, t: _exists(s, "correspondences", "final_combined_correspondences.npz"),
    "template_align": lambda s, t: _exists(s, "correspondences", "alignment_transform.npz"),
    "render_final":   _render_final_done,
}


def _get_sequences(data_root: str, seq: str = None) -> list:
    if seq:
        return [seq]
    return sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )


# Final deliverables reported at the end of a sequence: (label, filename, template_only).
# template_only items (the aligned template and its render) are only expected when a
# template was provided, so they are omitted from the summary otherwise.
_FINAL_OUTPUTS = [
    ("Fitted human",             "fitted_human.obj",          False),
    ("Segmented object",         "segmented_object.obj",      False),
    ("Aligned template",         "aligned_template.obj",      True),
    ("Render (human+object)",    "render_human_object.png",   False),
    ("Render (human+template)",  "render_human_template.png", True),
]


def _resolve_output(seq_dir: str, name: str):
    """Resolve a final deliverable to an absolute path, checking the top level first
    then the collate/uncollate fallbacks (intermediate_results/, correspondences/).
    Returns the path if found, else None."""
    for rel in (name, os.path.join(INTERMEDIATE_DIR, name),
                os.path.join("correspondences", name)):
        p = os.path.join(seq_dir, rel)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def _print_final_outputs(seq_dir: str, tmpl) -> None:
    """Print the resolved on-disk location of every final deliverable (_FINAL_OUTPUTS)."""
    print(f"\n{'='*60}")
    print(f"Final outputs for {os.path.basename(seq_dir)}:")
    print(f"{'='*60}")
    for label, name, template_only in _FINAL_OUTPUTS:
        if template_only and not tmpl:
            continue                       # not expected without a template
        path = _resolve_output(seq_dir, name)
        print(f"  {label:26s} {path if path else '(not produced)'}")
    print(f"{'='*60}")


def _run_step(python: str, module: str, args: list, gpu: str = None,
              extra_env: dict = None) -> bool:
    """Run `python -m <module> <args>` in the given interpreter (a fresh process
    per step for a clean CUDA context). `python` is the unified-env interpreter
    by default; a step may be pinned to another env's interpreter via config.
    `extra_env` overrides/extends environment variables for this step only."""
    env_vars = os.environ.copy()
    if gpu:
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu
    if extra_env:
        env_vars.update(extra_env)
    cmd = [python, "-m", module] + args
    print(f"[{module.split('.')[-1]}] Running: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=env_vars)
    if rc != 0:
        print(f"[{module.split('.')[-1]}] Failed with exit code {rc}")
        return False
    return True


def run_sequence(
    seq: str,
    data_root: str,
    steps: list,
    overwrite_from,
    fit_data_root: str,
    hamer_checkpoint: str,
    gpu: str,
    data_cfg: str,
    object_label: str = "object",
    dataset: str = None,
    template: str = None,
    filter_object: bool = False,
    use_pytorch3d: bool = False,
    segmenter: str = "sam3",
    object_prompt: str = None,
    human_prompt: str = "a person",
    lrm: str = "hy3d",
) -> None:
    seq_dir = os.path.join(data_root, seq)
    # Object text label (short, single word) keying the mask filenames and the
    # correspondence prompt. Given directly via --object; in --dataset back-compat
    # mode it is derived from the folder. The SAM 3 segmentation text defaults to
    # it — pass an elaborate noun phrase via --object_prompt instead.
    obj = resolve_object(seq, dataset, object_label)
    # Args shared by the two segmentation dispatch sites (auto_masks, img_segment).
    seg_args = ["--object", obj, "--segmenter", segmenter, "--human_prompt", human_prompt]
    if object_prompt:
        seg_args += ["--object_prompt", object_prompt]

    # Resolve the object template up front — used both by the template steps and by
    # the render_final completion check. Generic inference accepts a template *file*;
    # the --dataset back-compat mode resolves a per-object template from a directory.
    if not template:
        tmpl = None
    elif dataset:
        tmpl = template_path(seq, dataset, template)
    else:
        tmpl = template if os.path.isfile(template) else None
    if template and (not tmpl or not os.path.isfile(tmpl)):
        if any(s in steps for s in TEMPLATE_STEPS):
            print(f"[template] Skipping template steps — mesh not found: {template}")
        tmpl = None

    # Run/skip decision per step. overwrite_from is the index in VALID_STEPS of the
    # earliest forced step (None = nothing forced). A forced step always runs; any
    # other step runs only when its outputs are not already complete per STEP_DONE
    # (the single skip authority).
    def _forced(step):
        return overwrite_from is not None and VALID_STEPS.index(step) >= overwrite_from

    def _run(step):
        if _forced(step):
            return True
        if STEP_DONE[step](seq_dir, tmpl):
            print(f"[{step}] Skipping {seq} — already complete.")
            return False
        return True

    print(f"\n{'='*60}")
    print(f"Processing sequence: {seq}  (object={obj!r})")
    print(f"Steps: {steps}")
    print(f"{'='*60}")

    # Clean-slate overwrite: forcing from the first step (run_lrm) drops every previous
    # output first — including a prior run's collated intermediate_results/ — so the
    # run starts fresh. Inputs (image, masks, metadata, template, GT) are kept.
    # Otherwise, for a plain resume or partial overwrite of an already-collated
    # sequence, un-collate so the per-step output paths are visible again to the
    # STEP_DONE checks; collate runs again at the end to restore the layout.
    was_collated = os.path.isdir(os.path.join(seq_dir, INTERMEDIATE_DIR))
    clean_slate = _forced("run_lrm") and "run_lrm" in steps
    if clean_slate:
        clear_outputs(seq_dir)
    elif was_collated:
        uncollate(seq_dir)

    if "auto_masks" in steps and not _input_masks_done(seq_dir):
        # Create the input human/object masks from the image (SAM 3 by default) when
        # they are absent (the run_lrm step requires them). Guarded on mask presence (not
        # _run) so it never clobbers user-provided masks, even under --overwrite (input
        # .png files survive clear_outputs — see collate._is_input).
        _run_step(_STEP_PYTHON["img_segment"], "milo.pipeline.steps.image_segment",
                  ["--seq_dir", seq_dir, "--input_masks"] + seg_args, gpu=gpu)
        # The segmenter skips writing empty masks and may fail outright; report that
        # here — otherwise the only signal is run_lrm's missing-mask error further down.
        if not _input_masks_done(seq_dir):
            hint = ("refine --object_prompt (a short descriptive noun phrase works best)"
                    if segmenter == "sam3" else "refine --object")
            print(f"[auto_masks] No input masks produced for {seq} — {segmenter} found no "
                  f"match for object {(object_prompt or obj)!r} (or failed). Provide "
                  f"image_{{human,object}}.png in the sequence folder, or {hint}; "
                  f"run_lrm will stop until they exist.")

    if "run_lrm" in steps and _run("run_lrm"):
        _run_step(_STEP_PYTHON["run_lrm"], "milo.pipeline.steps.run_lrm",
                  ["--seq_dir", seq_dir, "--lrm", lrm], gpu=gpu)

    if "render" in steps and _run("render"):
        _run_step(_STEP_PYTHON["render"], "milo.pipeline.steps.render",
                  ["--seq_dir", seq_dir] + (["--pytorch3d"] if use_pytorch3d else []),
                  gpu=gpu)

    if "img_segment" in steps and _run("img_segment"):
        _run_step(_STEP_PYTHON["img_segment"], "milo.pipeline.steps.image_segment",
                  ["--seq_dir", seq_dir] + seg_args, gpu=gpu)

    if "mesh_segment" in steps and _run("mesh_segment"):
        _run_step(_STEP_PYTHON["mesh_segment"], "milo.pipeline.steps.mesh_segment",
                  ["--seq_dir", seq_dir, "--object", obj], gpu=gpu)

    if "kp2d" in steps and _run("kp2d"):
        kp2d_args = ["--seq_dir", seq_dir]
        if hamer_checkpoint:
            kp2d_args += ["--checkpoint", hamer_checkpoint]
        _run_step(_STEP_PYTHON["kp2d"], "milo.pipeline.steps.kp2d", kp2d_args, gpu=gpu)

    if "triangulate" in steps and _run("triangulate"):
        _run_step(_STEP_PYTHON["triangulate"], "milo.pipeline.steps.triangulate",
                  ["--seq_dir", seq_dir], gpu=gpu)

    if "init_smpl" in steps and _run("init_smpl"):
        _run_step(_STEP_PYTHON["init_smpl"], "milo.pipeline.steps.init_smpl",
                  ["--seq_dir", seq_dir], gpu=gpu)

    if "fit" in steps and _run("fit"):
        log_dir = fit_dir(seq_dir)
        fit_args = [
            os.path.join(FIT_DIR, "run_opt.py"),
            f"data={data_cfg}",
            f"data.seq={seq}",
            f"data.root={fit_data_root}",
            "run_opt=True",
            "run_vis=True",
        ]
        env_vars = os.environ.copy()
        if gpu:
            env_vars["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [_STEP_PYTHON["fit"]] + fit_args
        print(f"[fit] Running: {' '.join(cmd)}")
        # The fit is a hydra app with its own chatty logging; quiet it by default
        # (capture stdout/stderr to fit/fit_log.txt). --verbose shows it.
        if os.environ.get("MILO_VERBOSE") == "1":
            rc = subprocess.call(cmd, cwd=FIT_DIR, env=env_vars)
        else:
            os.makedirs(log_dir, exist_ok=True)
            fit_log = os.path.join(log_dir, "fit_log.txt")
            with open(fit_log, "w") as _lf:
                rc = subprocess.call(cmd, cwd=FIT_DIR, env=env_vars,
                                     stdout=_lf, stderr=subprocess.STDOUT)
            if rc == 0:
                print(f"[fit] Done → {log_dir} (log: fit_log.txt)")
        if rc != 0:
            print(f"[fit] Failed with exit code {rc} (see {os.path.join(log_dir, 'fit_log.txt')})")

    if "isolate" in steps and _run("isolate"):
        isolate_args = ["--seq_dir", seq_dir]   # isolate reads <seq_dir>/fit by default
        if filter_object:
            isolate_args += ["--filter"]
        _run_step(_STEP_PYTHON["isolate"], "milo.pipeline.steps.isolate", isolate_args, gpu=gpu)

    # --- optional template-alignment steps (only when a template is provided) ---
    if tmpl and "tmpl_render" in steps and _run("tmpl_render"):
        _run_step(_STEP_PYTHON["tmpl_render"], "milo.pipeline.steps.tmpl_render",
                  ["--seq_dir", seq_dir, "--template", tmpl, "--object", obj]
                  + (["--pytorch3d"] if use_pytorch3d else []), gpu=gpu)

    if tmpl and "correspond" in steps and _run("correspond"):
        # correspond runs in the isolated geo-aware env (config.STEP_PYTHON). Its old
        # torch-1.13 cuFFT crashes on newer host drivers, so LD_PRELOAD a driver-
        # compatible cuFFT 10.9 given explicitly via MILO_CORRESPOND_CUFFT.
        correspond_env = {}
        cufft = os.environ.get("MILO_CORRESPOND_CUFFT")
        if cufft and _STEP_PYTHON["correspond"] != MILO_PYTHON:
            prev = os.environ.get("LD_PRELOAD", "")
            correspond_env["LD_PRELOAD"] = f"{cufft}:{prev}" if prev else cufft
        _run_step(_STEP_PYTHON["correspond"], "milo.pipeline.steps.correspond",
                  ["--seq_dir", seq_dir, "--template", tmpl, "--object", obj],
                  gpu=gpu, extra_env=correspond_env)

    if tmpl and "template_align" in steps and _run("template_align"):
        _run_step(_STEP_PYTHON["template_align"], "milo.pipeline.steps.template_align",
                  ["--seq_dir", seq_dir, "--template", tmpl], gpu=gpu)

    # --- finalisation: collate outputs (no done-marker — always runs when listed;
    #     kept last so it sees every step's outputs) ---
    if "collate" in steps:
        _run_step(MILO_PYTHON, "milo.pipeline.steps.collate",
                  ["--seq_dir", seq_dir], gpu=gpu)

    # --- finalisation: render the collated deliverables (after collate, so the
    #     top-level final meshes are in place) ---
    if "render_final" in steps and _run("render_final"):
        _run_step(_STEP_PYTHON["render_final"], "milo.pipeline.steps.render_final",
                  ["--seq_dir", seq_dir] + (["--pytorch3d"] if use_pytorch3d else []),
                  gpu=gpu)

    # If the sequence was collated when we started but the caller's explicit --steps
    # omitted collate, re-collate to restore the layout we tore down for the resume.
    if was_collated and "collate" not in steps and not clean_slate:
        collate_run(seq_dir)

    _print_final_outputs(seq_dir, tmpl)


def _spawn_worker(gpu: str, batch_index: int, num_batches: int, args) -> int:
    """Spawn a subprocess for one GPU worker handling its batch of sequences."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--data_root", args.data_root,
        "--gpu", gpu,
        "--batch_index", str(batch_index),
        "--num_batches", str(num_batches),
    ]
    if args.steps:
        cmd += ["--steps", args.steps]
    if args.seq:
        cmd += ["--seq", args.seq]
    if args.overwrite == "__ALL__":
        cmd += ["--overwrite"]                       # bare: overwrite everything
    elif args.overwrite is not None:
        cmd += ["--overwrite", args.overwrite]       # forward the step list verbatim
    if args.fit_data_root:
        cmd += ["--fit_data_root", args.fit_data_root]
    if args.hamer_checkpoint:
        cmd += ["--hamer_checkpoint", args.hamer_checkpoint]
    if args.template:
        cmd += ["--template", args.template]
    cmd += ["--object", args.object]
    cmd += ["--segmenter", args.segmenter]
    cmd += ["--lrm", args.lrm]
    if args.object_prompt:
        cmd += ["--object_prompt", args.object_prompt]
    cmd += ["--human_prompt", args.human_prompt]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.filter_object:
        cmd += ["--filter_object"]
    if args.pytorch3d:
        cmd += ["--pytorch3d"]
    if args.verbose:
        cmd += ["--verbose"]
    print(f"[dispatch] GPU {gpu} batch {batch_index}/{num_batches}: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Full preprocessing → MILO pipeline"
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Root directory containing sequence folders",
    )
    parser.add_argument(
        "--seq", default=None,
        help="Single sequence name to process (default: all sequences in data_root)",
    )
    parser.add_argument(
        "--steps", default=None,
        help=f"Comma-separated list of steps to run, executed exactly as given. "
             f"Default (omitted): all of {ALL_STEPS}, plus {TEMPLATE_STEPS} when "
             f"--template is provided, then the finalisation step {FINALIZE_STEPS}. "
             f"With an explicit list, include 'collate' to run it.",
    )
    parser.add_argument(
        "--overwrite", nargs="?", const="__ALL__", default=None,
        help="Force-rerun steps regardless of the auto-continue check. Bare "
             "--overwrite re-runs everything (clean slate from run_lrm). "
             "--overwrite <step1,step2,...> re-runs the EARLIEST named step (in "
             "pipeline order) and every step after it (downstream invalidation); "
             "earlier steps still auto-continue. Place a bare --overwrite last or "
             "before another flag so it does not swallow the next token.",
    )
    parser.add_argument(
        "--fit_data_root", default=None,
        help="data.root passed to run_opt.py (the fit reads <root>/<seq>/). "
             "Defaults to --data_root.",
    )
    parser.add_argument(
        "--hamer_checkpoint", default=None,
        help="Path to HaMeR checkpoint (default: DEFAULT_CHECKPOINT)",
    )
    parser.add_argument(
        "--object", default="object",
        help="Short single-word object label (e.g. 'chair') — keys the mask filenames "
             "and the correspondence prompt; the SAM 3 segmentation text defaults to "
             "it (use --object_prompt for a descriptive phrase). Default: 'object'.",
    )
    parser.add_argument(
        "--segmenter", default="sam3", choices=["sam3", "gsam2"],
        help="Segmentation backend for the auto_masks / img_segment steps: SAM 3 "
             "(default) or Grounded-SAM-2.",
    )
    parser.add_argument(
        "--lrm", default="hy3d", choices=["hy3d", "sam3d"],
        help="Reconstruction backend for the run_lrm step: Hunyuan3D-2 (default) "
             "or SAM 3D Objects.",
    )
    parser.add_argument(
        "--object_prompt", default=None,
        help="SAM 3 text prompt for the object (default: the --object label). An "
             "elaborate noun phrase grounds best (e.g. 'a grey trolley suitcase'). "
             "Ignored by --segmenter gsam2.",
    )
    parser.add_argument(
        "--human_prompt", default="a person",
        help="SAM 3 text prompt for the person (default: 'a person'). "
             "Ignored by --segmenter gsam2.",
    )
    parser.add_argument(
        "--dataset", default=None, choices=list(DATASET_CFG.keys()),
        help="(Back-compat, optional) dataset name to reproduce the eval paths — "
             "derives the object label from the folder name and selects a "
             f"dataset-specific fit config. Available: {list(DATASET_CFG.keys())}. "
             "Omit for plain single-image inference.",
    )
    parser.add_argument(
        "--template", default=None,
        help="Optional object template mesh (.obj/.glb). When given, the optional "
             f"template-alignment steps {TEMPLATE_STEPS} are run. With --dataset, a "
             "directory of per-object templates resolved by object id is also accepted.",
    )
    parser.add_argument(
        "--filter_object", action="store_true",
        help="In the isolate step, apply the point-cloud outlier/cluster filtering "
             "(default: off — keep all segmented object vertices).",
    )
    parser.add_argument(
        "--pytorch3d", action="store_true",
        help="Render (render / tmpl_render / render_final) with the pytorch3d backend "
             "instead of the default pyrender backend.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose per-view / per-iteration logging in every step (default: off).",
    )
    parser.add_argument(
        "--gpu", default=None,
        help="CUDA_VISIBLE_DEVICES for this worker (e.g. '0'). "
             "Use --gpus to dispatch across multiple GPUs instead.",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated list of GPU IDs to distribute sequences across "
             "(e.g. '0,1,2,3'). Spawns one worker subprocess per GPU.",
    )
    # Internal args used by worker subprocesses; not intended for direct use.
    parser.add_argument("--batch_index", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--num_batches", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Resolve to an absolute path: steps run with cwd=REPO_ROOT but the fit runs
    # with cwd=milo/fit, so a relative --data_root would break the fit.
    args.data_root = os.path.abspath(args.data_root)
    # Propagate verbosity to every step subprocess (they inherit this env; see _log.py).
    if args.verbose:
        os.environ["MILO_VERBOSE"] = "1"

    if args.steps is None:
        # default: the full pipeline, plus the optional template-alignment steps
        # (not in ALL_STEPS) when a template is provided, then the finalisation steps.
        steps = list(ALL_STEPS)
        if args.template:
            steps += [s for s in TEMPLATE_STEPS if s not in steps]
        steps += FINALIZE_STEPS
    else:
        # an explicit --steps list is honored exactly (no template steps auto-added)
        steps = [s.strip() for s in args.steps.split(",")]
    invalid = [s for s in steps if s not in VALID_STEPS]
    if invalid:
        parser.error(f"Unknown steps: {invalid}. Available: {VALID_STEPS}")

    # Parse --overwrite into overwrite_from: the index in VALID_STEPS of the earliest
    # step to force (None = nothing forced; 0 = everything). A named list forces the
    # earliest of the named steps and every step after it (downstream invalidation).
    if args.overwrite is None:
        overwrite_from = None
    elif args.overwrite == "__ALL__":
        overwrite_from = 0
    else:
        named = [s.strip() for s in args.overwrite.split(",")]
        bad = [s for s in named if s not in VALID_STEPS]
        if bad:
            parser.error(f"Unknown --overwrite steps: {bad}. Available: {VALID_STEPS}")
        overwrite_from = min(VALID_STEPS.index(s) for s in named)

    # --- Multi-GPU dispatch mode ---
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",")]
        num_batches = len(gpu_list)
        failed = []
        with ProcessPoolExecutor(max_workers=num_batches) as executor:
            futures = {executor.submit(_spawn_worker, gpu, i, num_batches, args): gpu
                       for i, gpu in enumerate(gpu_list)}
            for future in as_completed(futures):
                gpu = futures[future]
                rc = future.result()
                if rc != 0:
                    failed.append(gpu)
                    print(f"[dispatch] Worker on GPU {gpu} exited with code {rc}")
        if failed:
            print(f"[dispatch] {len(failed)} worker(s) failed: {failed}")
            sys.exit(1)
        print("\nAll GPU workers complete.")
        return

    # --- Single-worker mode (direct or spawned) ---
    sequences = _get_sequences(args.data_root, args.seq)
    if not sequences:
        print(f"No sequences found in {args.data_root}")
        sys.exit(1)

    # Slice sequences for this batch if running as a worker
    if args.batch_index is not None and args.num_batches is not None:
        sequences = sequences[args.batch_index::args.num_batches]
        print(f"[worker GPU={args.gpu}] Batch {args.batch_index}/{args.num_batches}: "
              f"{len(sequences)} sequence(s)")

    data_cfg = DATASET_CFG[args.dataset] if args.dataset else DEFAULT_FIT_CFG
    # The fit reads each seq from <data.root>/<seq>/, i.e. the same --data_root.
    fit_data_root = os.path.abspath(args.fit_data_root) if args.fit_data_root else args.data_root

    print(f"Processing {len(sequences)} sequence(s): {sequences[:5]}{'...' if len(sequences)>5 else ''}")
    print(f"Steps: {steps}")

    for seq in sequences:
        try:
            run_sequence(
                seq=seq,
                data_root=args.data_root,
                steps=steps,
                overwrite_from=overwrite_from,
                fit_data_root=fit_data_root,
                hamer_checkpoint=args.hamer_checkpoint,
                gpu=args.gpu,
                data_cfg=data_cfg,
                object_label=args.object,
                dataset=args.dataset,
                template=args.template,
                filter_object=args.filter_object,
                use_pytorch3d=args.pytorch3d,
                segmenter=args.segmenter,
                object_prompt=args.object_prompt,
                human_prompt=args.human_prompt,
                lrm=args.lrm,
            )
        except Exception as e:
            print(f"[pipeline] ERROR on {seq}: {e}")
            import traceback
            traceback.print_exc()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

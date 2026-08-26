"""
pipeline/steps/correspond.py   (env: geo-aware)

Geometry-aware semantic correspondence between the object TEMPLATE and the LRM
object (paper Sec 3.6), via GeoAware-SC (Zhang et al., "Telling Left from Right").
Two stages, run on one sequence:
  stage 1 — rank the LRM (H3D) renders by feature-similarity to the input image
            → correspondences/render_similarity_to_input.npz
  stage 2 — dense template↔LRM vertex correspondences over the top views
            → correspondences/final_combined_correspondences.npz

Upstream inputs (already produced): render_segment/renders/ + segments/ (LRM, from
`render` + `img_segment`), render_segment/renders_gt/ + segments_gt/ (template, from
`tmpl_render`), the input image + <img>_object.png mask, full_img_textured.glb,
metadata.npz. The template masks (segments_gt/) come straight from the
`tmpl_render` rasterizer depth (no extra segmentation model needed).

Heavy deps (Stable Diffusion + DINOv2 + ODISE) live in the `geo-aware` env and are
imported lazily so the rest of the pipeline never pays for them.

Standalone usage (env geo-aware):
    python -m milo.pipeline.steps.correspond --seq_dir /path/to/seq --object "chair" \
        --template /path/to/template.obj
"""

import argparse
import os

from milo.pipeline.steps._log import vprint, set_verbose


def run(seq_dir: str, template: str, object_label: str = "object", verbose: bool = False,
        skip_pair_vis: bool = False) -> bool:
    """Compute template↔LRM semantic correspondences for one sequence."""
    set_verbose(verbose)
    seq_dir = os.path.abspath(seq_dir)
    obj_name = object_label

    renders_gt = os.path.join(seq_dir, "render_segment", "renders_gt")
    if not os.path.isdir(renders_gt) or not os.listdir(renders_gt):
        print(f"[correspond] Skipping — renders_gt missing (run tmpl_render first): {renders_gt}")
        return False
    if not os.path.isfile(template):
        print(f"[correspond] Skipping — template not found: {template}")
        return False

    # Lazy import: triggers the GeoAware-SC SD/DINO/ODISE stack (geo-aware env only).
    from milo.pipeline.steps._geoaware import stage1, stage2

    vprint(f"[correspond] obj_name={obj_name!r} template={os.path.basename(template)}")
    # Load the SD + DINO + aggregation stack ONCE and share it across both stages
    # (each stage used to load it independently, ~1-2 min wasted per sequence).
    print("[correspond] loading SD + DINO models (shared across stages)...")
    models = stage1.load_models()

    # The driver decides whether correspond runs at all (its STEP_DONE check); when it
    # does run, recompute both sub-stages fully (overwrite=True) so a forced re-run
    # actually redoes the work rather than resuming stale stage outputs.
    ok1, h3d_feats = stage1.run(seq_dir, obj_name, overwrite=True,
                                models=models, return_top_features=True)

    # Hand stage1's top-K raw render features to stage2 so its H3D preload skips
    # re-extracting them (bit-identical; see preload_view_data in _geoaware/stage2.py).
    stage2.run(seq_dir, obj_name, template, overwrite=True, models=models,
               h3d_features=(h3d_feats if ok1 else None),
               save_pair_vis=not skip_pair_vis)
    print(f"[correspond] Done → {os.path.join(seq_dir, 'correspondences')}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeoAware-SC template↔LRM correspondences for one sequence (env: geo-aware)"
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory")
    parser.add_argument("--template", required=True, help="Path to the object template mesh (.obj/.glb)")
    parser.add_argument("--object", default="object",
                        help="Object text label (e.g. 'chair'). Default: 'object'.")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    parser.add_argument("--skip_pair_vis", action="store_true",
                        help="Skip the per-pair visualization PNGs (inspection only; npz outputs unaffected).")
    args = parser.parse_args()
    run(seq_dir=args.seq_dir, template=args.template, object_label=args.object, verbose=args.verbose,
        skip_pair_vis=args.skip_pair_vis)

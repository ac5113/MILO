# MILO pipeline

End-to-end single-image HOI pipeline. Given, per sequence, an input image (and
optionally its human/object masks), it reconstructs a combined human+object mesh,
segments it, lifts 2D body/hand keypoints to 3D, fits SMPL-H to the human, isolates
the object point cloud, and optionally aligns an object template.

## Steps

`run_pipeline.py` orchestrates the steps; each runs as a fresh subprocess in the
`milo` env via `python -m milo.pipeline.steps.<step>`.

| # | Step | Module | Output |
|---|------|--------|--------|
| — | `auto_masks` | `image_segment` (`--input_masks`) | `image_human.png` / `image_object.png` — masks of the input image, generated only when absent |
| 0 | `run_lrm` | `run_lrm` | `full_img_textured.glb` (combined human+object mesh; Hunyuan3D-2 default, SAM 3D Objects with `--lrm sam3d`) |
| 1 | `render` | `render` | `render_segment/renders/` (multi-view PNGs + visible-vertex `.npy`) |
| 2 | `img_segment` | `image_segment` | `render_segment/segments/` (per-view person + object masks; SAM 3 default, Grounded-SAM-2 with `--segmenter gsam2`) |
| 3 | `mesh_segment` | `mesh_segment` | `vertex_classification_*.npz` + `segmentation_colored_mesh.obj` |
| 4 | `kp2d` | `kp2d` | `keypoints/` (COCO25 body) + `kp2d_hand/` (MANO hands) |
| 5 | `triangulate` | `triangulate` | `keypoints_3d*.npy` (3D body + hand keypoints) |
| 6 | `init_smpl` | `init_smpl` | `milo_init.npz` (SMPL-H init: HMR2.0 body + HaMeR hands) |
| 7 | `fit` | `milo/fit/run_opt.py` | `<seq>/fit/` (2-stage SMPL-H fit) |
| 8 | `isolate` | `isolate` | **`fitted_human.obj`** + **`segmented_object.obj`** (final meshes) + `filtered_h3d_obj_pc.obj` |

Optional template-alignment steps (run only when `--template` is given; appended after `isolate`):

| Step | Module | Output |
|------|--------|--------|
| `tmpl_render` | `tmpl_render` | `render_segment/renders_gt/` (template renders) |
| `correspond` | `correspond` | `correspondences/final_combined_correspondences.npz` (GeoAware-SC) |
| `template_align` | `template_align` | `correspondences/alignment_transform.npz` + `aligned_template.obj` |

Finalisation steps (appended last by default; with an explicit `--steps` list, include them yourself):

| Step | Module | Output |
|------|--------|--------|
| `collate` | `collate` | Final meshes stay on top; intermediates move into `intermediate_results/` |
| `render_final` | `render_final` | `render_human_object.png` (+ `render_human_template.png` with a template) |

## Per-sequence inputs

Each sequence is a folder `<data_root>/<seq>/` containing:

- `image.<jpg|png>` — the input image (any basename; mask/viz files are skipped by
  name). **The only required input.**
- `image_human.png` / `image_object.png` *(optional, recommended)* — masks of the
  input image (or `<imgbase>_human.png` / `<imgbase>_object.png`).
- `metadata.npz` *(optional)* — intrinsics (`focal_length`, `principal_point`);
  default is focal `0.5·(H+W)` with the principal point at the image centre.

The masks are a prerequisite of `run_lrm`; when absent, `auto_masks` generates them
with SAM 3 (Grounded-SAM-2 with `--segmenter gsam2`). Auto masks union *all*
detected people/objects, so provide your own masks to isolate the interacting pair
when an image contains several. Provided masks are never overwritten, even under
`--overwrite`. (`img_segment`, step 2, segments the *rendered* views, not the input
image.)

## Configuration

Machine-specific settings live in [`config.py`](config.py): submodule/checkpoint
paths (`HUNYUAN3D_DIR`, `H3D_CHECKPOINTS_DIR`, `SAM3D_DIR`, `SAM3D_CHECKPOINTS_DIR`),
the per-step interpreter (`STEP_PYTHON`; `correspond` switches to
`$MILO_CORRESPOND_ENV/bin/python` when set), dataset object naming (`obj_name`),
template resolution (`template_path`), and the fit output dir (`fit_dir`).

## Environment

The whole pipeline runs in the single `milo` conda env, built per
[`docs/INSTALL.md`](../../docs/INSTALL.md); `run_pipeline.py` launches each step in
the current interpreter. The one exception is the optional `correspond` step, which
runs in a separate `geo-aware` env — set `MILO_CORRESPOND_ENV` and
`MILO_CORRESPOND_CUFFT` (see
[INSTALL.md → Optional: correspond](../../docs/INSTALL.md#optional-correspond-template-alignment)).

## Running

Single sequence, full pipeline:

```bash
python milo/pipeline/run_pipeline.py \
    --data_root /path/to/data_root \
    --seq <sequence_name> \
    --object "chair" \
    --object_prompt "a wooden chair" \
    --overwrite
```

All sequences in `data_root` (skips steps whose outputs already exist):

```bash
python milo/pipeline/run_pipeline.py --data_root /path/to/data_root --object "chair"
```

With object-template alignment: add `--template /path/to/template.obj`. Distribute
across GPUs: `--gpus 0,1,2,3`. Run a single step standalone:

```bash
python -m milo.pipeline.steps.run_lrm --seq_dir /path/to/data_root/<seq> [--lrm sam3d]
```

### Flags

- `--object <label>` — short **single-word** object label (default `"object"`). Keys
  the per-view mask filenames, the correspondence prompt and dataset naming; also the
  SAM 3 prompt when `--object_prompt` is omitted. Multi-word labels break the
  mask-filename lookup.
- `--object_prompt <phrase>` — SAM 3 text prompt for the object. A descriptive noun
  phrase grounds best (`"a grey trolley suitcase"`, not `"trolley"`). Ignored by
  `--segmenter gsam2`, which uses the terse `--object` label instead.
- `--human_prompt <phrase>` — SAM 3 text prompt for the person (default `"a person"`).
- `--segmenter <sam3|gsam2>` — segmentation backend for `auto_masks` / `img_segment`
  (default `sam3`).
- `--lrm <hy3d|sam3d>` — reconstruction backend for `run_lrm` (default `hy3d`).
  `sam3d` needs its gated weights downloaded first (see `docs/INSTALL.md`).
- `--steps` — comma-separated subset of steps, run exactly as given (default: all
  core steps, the template steps when `--template` is set, then `collate`/`render_final`).
- `--template <mesh>` — enable the template-alignment steps with a template mesh file.
- `--overwrite [step1,...]` — force re-runs (default: skip complete steps). Bare
  `--overwrite` re-runs everything from `run_lrm`; with a step list, the earliest
  named step *and every step after it* re-run (downstream invalidation).
- `--verbose` — per-view / per-iteration detail + the full fit log on the console
  (default: step boundaries + summaries; the fit log always goes to `<seq>/fit/fit_log.txt`).
- `--filter_object` — in `isolate`, apply point-cloud outlier/cluster filtering
  (default: off).
- `--pytorch3d` — render with the pytorch3d backend instead of pyrender.
- `--gpus <ids>` — distribute sequences across GPUs.
- `--fit_data_root` — override the fit's data root (default: `--data_root`).
- `--hamer_checkpoint` — path to a HaMeR checkpoint (default: bundled).

## Troubleshooting

See [`docs/INSTALL.md` → Troubleshooting](../../docs/INSTALL.md#troubleshooting).

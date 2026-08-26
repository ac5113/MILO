# Model weights & data

All weights and body models live under `_DATA/` at the repo root. Most are fetched
for you; a few are license-gated and need an account. Run the commands below with
the `milo` env active, after [installation](INSTALL.md).

## Repository layout

```
milo/
  pipeline/        # orchestration + per-step modules (milo.pipeline.steps.*)
  fit/             # SMPL-H fitting engine (hydra app; run with cwd=milo/fit)
  eval/            # dataset preparation + metrics
scripts/           # install_milo.sh, download_models.sh, eval_results.sh, setup_transfer_data.sh
docs/              # PIPELINE.md, INSTALL.md, DATA.md
third-party/       # git submodules
_DATA/             # model weights / body models / checkpoints (downloaded; gitignored)
```

## Quick start

```bash
bash scripts/download_models.sh
```

This fetches the Grounded-SAM-2 checkpoints (for the `--segmenter gsam2` fallback),
the HaMeR demo data, the Hunyuan3D-2 weights, and — each prompted, each optional —
the gated body models and the gated SAM 3D Objects weights (~13 GB, only for
`--lrm sam3d`). Everything else auto-downloads on the first pipeline run —
including the SAM 3 checkpoint, which is
gated on Hugging Face (one-time access request + login, see
[INSTALL.md → SAM 3 checkpoint](INSTALL.md#sam-3-checkpoint-default-segmenter)).

## Body models (registration required)

SMPL-H, VPoser v1.0, SMPL-X and MANO sit behind their own licenses on the MPI
download server — register on each page below and accept the license.
`scripts/download_models.sh` fetches all four when you answer **y** to its prompt
(it asks once for your `is.tue.mpg.de` login).

To place them by hand, only these files are needed:

- **SMPL-H** — [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de): *Extended SMPL+H
  model (used in AMASS)* (`smplh.tar.xz`), keep the neutral `model.npz` →
  `_DATA/body_models/smplh/neutral/model.npz`.
- **VPoser v1.0** (not v2.0) — [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de):
  `vposer_v1_0.zip` → `_DATA/body_models/vposer_v1_0/`.
- **SMPL-X** — [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de):
  `models_smplx_v1_1.zip`, keep `SMPLX_NEUTRAL.npz` (read for hand PCA) →
  `_DATA/body_models/smplx/SMPLX_NEUTRAL.npz`.
- **MANO** — [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de): `mano_v1_2.zip`,
  keep `MANO_RIGHT.pkl` (HaMeR mirrors left hands to it) →
  `_DATA/data/mano/MANO_RIGHT.pkl`.

If a scripted download 404s, the file name on your account's *Downloads* page may
differ — update the matching `sfile=` value in `scripts/download_models.sh`.

## What goes where

```
_DATA/
  body_models/
    smplh/neutral/model.npz        # SMPL-H        (gated)
    vposer_v1_0/                   # VPoser v1.0   (gated)
    smplx/SMPLX_NEUTRAL.npz        # SMPL-X        (gated)
  data/
    mano/MANO_RIGHT.pkl            # MANO          (gated)
    mano_mean_params.npz           # comes with the HaMeR demo data
  hamer_ckpts/checkpoints/hamer.ckpt   # HaMeR
  vitpose_ckpts/                       # ViTPose (comes with the HaMeR demo data)
  h3d_checkpoints/tencent/Hunyuan3D-2/ # Hunyuan3D-2
  sam3d_checkpoints/hf/                # SAM 3D Objects (gated; only for --lrm sam3d)
  transfer_data/                       # SMPL-X→SMPL-H setup (gated; only for the InterCap eval)
```

`hy3dgen` resolves the Hunyuan3D-2 weights via the `HY3DGEN_MODELS` env var; the
`run_lrm` step points it at `_DATA/h3d_checkpoints` (`milo/pipeline/config.py`) so
the pre-downloaded weights are used. SAM 3D Objects is instantiated from
`_DATA/sam3d_checkpoints/hf/pipeline.yaml`; the repo is gated — request access at
<https://huggingface.co/facebook/sam-3d-objects> and `huggingface-cli login` first.

Auto-downloaded at runtime (no action): HMR2.0 / 4D-Humans, HaMeR weights, and the
ViTDet person detector. `download_models.sh` pre-fetches the HaMeR demo data so the
~5.7 GB download doesn't stall the first run.

The optional `correspond` step additionally needs GeoAware-SC's
`results_spair/best_856.PTH` checkpoint — see
[INSTALL.md → Optional: correspond](INSTALL.md#optional-correspond-template-alignment).

## Per-sequence data layout

Each sequence is a directory containing the input image (+ optional masks and
intrinsics); the pipeline writes its artifacts back into it:

```
<seq>/
  image.jpg                          # input RGB (any basename) — the ONLY required input
  image_human.png                    # human mask (OPTIONAL, recommended; or <imgbase>_human.png)
  image_object.png                   # object mask (OPTIONAL, recommended; or <imgbase>_object.png)
  metadata.npz                       # OPTIONAL intrinsics; else focal=0.5*(H+W), centre
  full_img_textured.glb              # [run_lrm] combined LRM mesh
  render_segment/                    # [render/img_segment/mesh_segment] renders, masks,
                                     #   segmentation_colored_mesh.obj
  keypoints/                         # [kp2d]   COCO25 body keypoints (+ keypoints_viz/)
  kp2d_hand/                         # [kp2d]   HaMeR hand keypoints + overlays
  keypoints_3d*.{npy,obj}            # [triangulate] 3D body+hand keypoints
  milo_init.npz                      # [init_smpl] SMPL-H initialization
  fit/                               # [fit]    posed SMPL-H + object meshes, fit_log.txt
  fitted_human.obj                   # [isolate] FINAL SMPL-H human mesh
  segmented_object.obj               # [isolate] FINAL segmented object mesh
  filtered_h3d_obj_pc.obj            # [isolate] object point cloud (+ _indices.npy)
  correspondences/                   # [template] correspondences + alignment_transform.npz
  aligned_template.obj               # [template] aligned object template
  render_human_object.png            # [render_final] white-bg render of the final meshes
  render_human_template.png          # [render_final] same with the aligned template
  intermediate_results/              # [collate] everything above except inputs + finals
  gt_human.obj                       # [eval]   GT human (SMPL-H, vertex-corresponded)
  gt_object.obj                      # [eval]   GT object mesh (with faces)
  eval/                              # [eval]   --save_mesh: combined_aligned_{icp,template}.obj
```

For the InterCap / HODome / IMHD² eval sets, `milo/eval/prepare_dataset.py` produces
this layout (plus `gt_*.obj` and `object_template.obj`) for every frame in
`milo/eval/splits/<dataset>_test.txt` directly from the official releases — see
[Datasets and Evaluation](../README.md#datasets-and-evaluation).

The `gt_*.obj` files are only needed to score a sequence with
`scripts/eval_results.sh` (the eval resolves inputs from the top level or
`intermediate_results/`, so a collated run scores directly). The bundled
`demo/example` (an InterCap frame) ships its GT meshes, so the quick-start run can
be scored immediately.

The `fit` step's chatty logging goes to `<seq>/fit/fit_log.txt`; `--verbose` prints
it (and per-step detail) to the console instead.

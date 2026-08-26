# Method & pipeline

MILO reconstructs 3D human–object interactions from a single RGB image by *interpreting* a
combined human+object mesh produced by a Large Reconstruction Model (Hunyuan3D-2.0 by default,
SAM 3D Objects with `--lrm sam3d`). See the
[project page](https://ac5113.github.io/MILO) / paper for the method; this page lists the
implementing pipeline.

The pipeline is a sequence of single-purpose steps (`milo/pipeline/steps/`), each runnable
standalone or orchestrated by `milo/pipeline/run_pipeline.py`. All steps run in the single
`milo` environment, except the optional `correspond` step (see [INSTALL.md](INSTALL.md)).

| # | Stage (paper §) | Step(s) | What it does |
|---|---|---|---|
| 0 | Input Segmentation *(auto)* | `auto_masks` | SAM 3 masks of the input image (`image_{human,object}.png`; Grounded-SAM-2 via `--segmenter gsam2`), generated only when the sequence doesn't already provide them |
| 1 | Combined Mesh Generation (§3.2) | `run_lrm` | LRM → combined human+object mesh (`full_img_textured.glb`); Hunyuan3D-2.0 by default, SAM 3D Objects via `--lrm sam3d` |
| 2 | Multi-view Rendering | `render` | Render the LRM mesh from 60 viewpoints + visible-vertex maps (pyrender by default; `--pytorch3d` for the pytorch3d backend) |
| 3 | Point-cloud Segmentation (§3.5) | `img_segment`, `mesh_segment` | SAM 3 masks (default; Grounded-SAM-2 via `--segmenter gsam2`) → per-vertex human/object classification |
| 4 | 3D Keypoint Estimation (§3.3) | `kp2d`, `triangulate` | ViTPose (body) + HaMeR (hands) per view → robust 3D keypoints |
| 5 | Human Optimization (§3.4) | `init_smpl`, `fit` | HMR2.0/HaMeR init → 2-stage SMPL-H fit (root + pose) at T=1 |
| 6 | Object isolation | `isolate` | Filter the object point cloud from the segmented LRM mesh |
| 7 | **Template Alignment (§3.6, optional)** | `tmpl_render`, `correspond`, `template_align` | Render template → geometry-aware semantic correspondences ([GeoAware-SC](https://github.com/Junyi42/GeoAware-SC)) → weighted Sim(3) + ICP |
| 8 | Finalisation | `collate`, `render_final` | Tidy the sequence folder (finals on top, intermediates → `intermediate_results/`) + white-background renders of the final meshes |

For per-step inputs/outputs and run flags, see [`milo/pipeline/README.md`](../milo/pipeline/README.md).

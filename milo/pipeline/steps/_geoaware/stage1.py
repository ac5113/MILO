"""
pipeline/steps/_geoaware/stage1.py   (env: geo-aware)

Stage-1 view ranking: rank a sequence's LRM (H3D) renders by
mutual-nearest-neighbor (MNN) feature-similarity to the RGB input image.

For every render in ``{seq_dir}/render_segment/renders/`` extracts fused
SD+DINO features and computes the MNN ratio against the input-image features,
then writes the per-view ranking to
``{seq_dir}/correspondences/render_similarity_to_input.npz`` (parallel
arrays: ``types``, ``elevations``, ``azimuths``, ``mnn_ratios``, ``paths``).
Stage 2 (``stage2.py``) consumes this NPZ to pick the top-K H3D views.
"""

import os
import tempfile
import argparse
# Cache/scratch root for SD/DINO model downloads + temp chunks. Configurable via
# MILO_GEO_CACHE; defaults to a repo-local _cache/geo (repo root is 4 levels up).
storage_path = os.environ.get('MILO_GEO_CACHE') or os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '_cache', 'geo'))
os.makedirs(storage_path, exist_ok=True)

# Point tempfile at the cache root instead of the (possibly small) system /tmp
tempfile.tempdir = storage_path

os.environ['TORCH_HOME'] = storage_path
os.environ['XDG_CACHE_HOME'] = storage_path
os.environ['HF_HOME'] = storage_path  # For HuggingFace models
os.environ['TMPDIR'] = storage_path   # For temporary download chunks

# Specific to iopath / detectron2 / odise
os.environ['FVCORE_CACHE'] = storage_path

import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# MILO: resolve GeoAware-SC (model_utils/utils/preprocess_map) from the vendored submodule.
import os as _os, sys as _sys
_GEO = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "third-party", "GeoAware-SC"))
if _GEO not in _sys.path:
    _sys.path.insert(0, _GEO)
_CKPT = _os.path.join(_GEO, "results_spair", "best_856.PTH")

from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed

device = 'cuda'
set_seed(42)
num_patches = 60
img_size = 480

MAX_PIXELS_FOR_SIMILARITY = 10000
USE_CACHED_FEATURES = False
FALLBACK_TO_COMPUTE = True  # Need to compute for input image at minimum
TOP_K = 5

INTERCAP_MAPPING = ['trolley', 'skateboard', 'sports ball', 'umbrella', 'tennis_racquet', 'suitcase', 'chair', 'bottle', 'cup', 'stool']
_OBJ_NAME_OVERRIDE = None   # MILO: set by run() to bypass the InterCap folder-name mapping


def compute_bbox_from_mask(mask, tolerance=0.1):
    """Bounding box ``(x0, y0, x1, y1)`` of pixels > 128, padded by ``tolerance``
    (fraction of bbox size) and clamped to the image; ``None`` if mask is empty."""
    ys, xs = np.where(mask > 128)
    if len(xs) == 0:
        return None
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    
    pad_x = int(bbox_w * tolerance)
    pad_y = int(bbox_h * tolerance)
    
    h, w = mask.shape[:2]
    x_start = max(0, x_min - pad_x)
    y_start = max(0, y_min - pad_y)
    x_end = min(w, x_max + pad_x + 1)
    y_end = min(h, y_max + pad_y + 1)
    
    return (x_start, y_start, x_end, y_end)


def compute_mutual_nn_ratio(feat_a_norm, feat_b_norm, fg_indices_a, fg_indices_b,
                            max_pixels=10000, chunk_size=2000):
    """Fraction of foreground pixels in A whose cosine-NN in B maps back to them.

    Subsamples each side to ``max_pixels``; similarity is computed in
    ``chunk_size`` row blocks to bound GPU memory."""
    if len(fg_indices_a) > max_pixels:
        sampled_a = np.random.choice(len(fg_indices_a), max_pixels, replace=False)
        fg_indices_a = fg_indices_a[sampled_a]
    if len(fg_indices_b) > max_pixels:
        sampled_b = np.random.choice(len(fg_indices_b), max_pixels, replace=False)
        fg_indices_b = fg_indices_b[sampled_b]

    if len(fg_indices_a) == 0 or len(fg_indices_b) == 0:
        return 0.0

    feat_a_fg = feat_a_norm[fg_indices_a]
    feat_b_fg = feat_b_norm[fg_indices_b]

    forward_nn = []
    for i in range(0, len(feat_a_fg), chunk_size):
        end_i = min(i + chunk_size, len(feat_a_fg))
        sim = torch.mm(feat_a_fg[i:end_i], feat_b_fg.t())
        forward_nn.append(torch.argmax(sim, dim=1))
    forward_nn = torch.cat(forward_nn)

    reverse_nn = []
    for i in range(0, len(feat_b_fg), chunk_size):
        end_i = min(i + chunk_size, len(feat_b_fg))
        sim = torch.mm(feat_b_fg[i:end_i], feat_a_fg.t())
        reverse_nn.append(torch.argmax(sim, dim=1))
    reverse_nn = torch.cat(reverse_nn)

    # Vectorized mutual check (same integer count as the per-element loop, without
    # one synchronizing .item() call per pixel).
    mutual_count = int(
        (reverse_nn[forward_nn] == torch.arange(len(forward_nn), device=forward_nn.device)).sum().item()
    )

    mutual_nn_ratio = mutual_count / len(forward_nn)
    return mutual_nn_ratio


def get_feature_cache_path(render_path, obj_name, seq_dir):
    """Map a render PNG path to its ``.pt`` feature-cache path under
    ``{seq_dir}/cached_features/{relpath-from-render-segment}``."""
    features_base_dir = os.path.join(seq_dir, 'cached_features')
    render_segment_dir = os.path.dirname(os.path.dirname(render_path))
    rel_path = os.path.relpath(render_path, render_segment_dir)
    cache_dir = os.path.join(features_base_dir, os.path.dirname(rel_path))
    filename = os.path.basename(render_path).replace('.png', '.pt')
    return os.path.join(cache_dir, filename)


def load_cached_features(render_path, obj_name, seq_dir, img_size):
    """Load fused SD+DINO features from disk, moved to GPU.

    Returns ``(features_raw, features_upsampled)`` or ``(None, None)`` when
    the cache file is missing or unreadable. Upsamples on the fly when the
    cache only holds the raw feature map.
    """
    cache_path = get_feature_cache_path(render_path, obj_name, seq_dir)
    if not os.path.exists(cache_path):
        return None, None
    try:
        save_dict = torch.load(cache_path, map_location='cpu')
        features = save_dict.get('features_raw', None)
        features_upsampled = save_dict.get('features_upsampled', None)
        if features is not None:
            features = features.cuda()
        if features_upsampled is not None:
            features_upsampled = features_upsampled.cuda()
        if features is not None and features_upsampled is None:
            features_upsampled = F.interpolate(features, size=(img_size, img_size), mode='bilinear', align_corners=False)
        return features, features_upsampled
    except Exception as e:
        print(f"      Warning: Failed to load cached features from {cache_path}: {e}")
        return None, None


def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    """Extract L2-normalized fused SD + DINOv2 descriptors for one image.

    Takes one of ``img`` (PIL) or ``img_path``. Reads pre-cached SD/DINO
    ``.pt`` shards when present, else runs the extractors. SD blocks s3/s4/s5
    and DINO tokens are concatenated and passed through ``aggre_net`` to a
    unified 768-d feature map at ``num_patches x num_patches``.
    """
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    
    desc = aggre_net(desc_gathered)
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


def is_sequence_already_processed(seq_dir):
    """True if the sequence has a non-empty render_similarity_to_input.npz."""
    results_path = os.path.join(seq_dir, 'correspondences', 'render_similarity_to_input.npz')
    # results_path = os.path.join(seq_dir, 'correspondences_new', 'render_similarity_to_input.npz')
    results = dict(np.load(results_path, allow_pickle=True)) if os.path.exists(results_path) else None
    if results is not None and len(results['mnn_ratios']) > 0:
        return True
    return False


def process_single_image(img_path, sd_model, sd_aug, aggre_net, extractor_vit, feature_sink=None):
    """Rank all H3D renders of one sequence against the input image by MNN ratio.

    Writes render_similarity_to_input.npz and a top-K grid PNG under
    ``{seq_dir}/correspondences``. Returns True on success.

    When ``feature_sink`` is a dict, the raw (60x60) fused features of the
    top-``TOP_K`` renders are left in it keyed by ``(elevation, azimuth)`` (CPU
    fp32 tensors), so stage 2 can reuse them instead of re-extracting.
    """
    folder_name = os.path.basename(os.path.dirname(img_path))
    obj_name = _OBJ_NAME_OVERRIDE
    if obj_name is None:
        try:
            obj_name = INTERCAP_MAPPING[(int)(folder_name.split('__')[1]) - 1]
        except IndexError:
            print(f"  Could not determine obj_name for {folder_name}, skipping...")
            return False
    seq_dir = os.path.dirname(img_path)

    # =========================================================================
    # STEP 1: Load and crop the input image around its object segmentation mask
    # =========================================================================
    imgname = os.path.basename(img_path).replace('.jpg', '')
    input_mask_path = os.path.join(os.path.dirname(img_path), f'{imgname}_object.png')
    if not os.path.exists(input_mask_path):
        print(f"  Missing input segmentation mask at {input_mask_path}, skipping...")
        return False

    input_img_pil = Image.open(img_path).convert('RGB')
    input_mask_orig = np.array(Image.open(input_mask_path).convert('L'))

    input_bbox = compute_bbox_from_mask(input_mask_orig, tolerance=0.1)
    if input_bbox is None:
        print(f"  Empty input mask, skipping...")
        return False

    inp_x0, inp_y0, inp_x1, inp_y1 = input_bbox
    inp_crop_w, inp_crop_h = inp_x1 - inp_x0, inp_y1 - inp_y0

    input_img_cropped = input_img_pil.crop((inp_x0, inp_y0, inp_x1, inp_y1))
    input_mask_cropped = Image.fromarray(input_mask_orig[inp_y0:inp_y1, inp_x0:inp_x1])

    input_img_resized = resize(input_img_cropped, target_res=img_size, resize=True, to_pil=True)
    input_mask_resized = np.array(resize(input_mask_cropped, target_res=img_size, resize=True, to_pil=True))
    if input_mask_resized.ndim == 3:
        input_mask_resized = input_mask_resized[..., 0]

    print(f"  Input image cropped to bbox {input_bbox} ({inp_crop_w}x{inp_crop_h}), resized to {img_size}x{img_size}")

    # =========================================================================
    # STEP 2: Extract features for the input image
    # =========================================================================
    print(f"  Extracting features for input image...")
    feat_input = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=input_img_resized)
    feat_input_upsampled = F.interpolate(feat_input, size=(img_size, img_size), mode='bilinear', align_corners=False)

    with torch.no_grad():
        C, H, W = feat_input_upsampled.shape[1], feat_input_upsampled.shape[2], feat_input_upsampled.shape[3]
        feat_input_flat_norm = F.normalize(
            feat_input_upsampled.view(C, H * W).permute(1, 0), p=2, dim=1
        )  # [H*W, C]
        input_fg_flat = (input_mask_resized > 128).astype(np.float32).flatten()
        input_fg_indices = np.where(input_fg_flat > 0.5)[0]

    print(f"  Input foreground pixels: {len(input_fg_indices)}")

    # =========================================================================
    # STEP 3: Gather all renders (GT + H3D) and compare via MNN ratio
    # =========================================================================
    renders_h3d_list = sorted(glob.glob(os.path.join(seq_dir, 'render_segment', 'renders', '*.png')))

    all_render_meta = []

    for render_path in renders_h3d_list:
        fname = os.path.basename(render_path)
        elevation = int(fname.split('_e')[1].split('_a')[0])
        azimuth = int(fname.split('_a')[1].split('.png')[0])
        seg_path = render_path.replace('renders', 'segments').replace('obj', obj_name.replace(' ', '_'))
        all_render_meta.append({
            'type': 'h3d', 'elevation': elevation, 'azimuth': azimuth,
            'seg_path': seg_path, 'path': render_path
        })

    print(f"  Comparing input image against {len(all_render_meta)} H3D renders...")

    similarity_results = []

    for r_idx, meta in enumerate(all_render_meta):
        render_path = meta['path']
        seg_path = meta['seg_path']

        if not os.path.exists(seg_path):
            continue

        # Load render mask and crop
        render_mask_orig = np.array(Image.open(seg_path).convert('L'))
        render_bbox = compute_bbox_from_mask(render_mask_orig, tolerance=0.1)
        if render_bbox is None:
            continue

        rx0, ry0, rx1, ry1 = render_bbox

        render_pil = Image.open(render_path).convert('RGB')
        render_cropped = render_pil.crop((rx0, ry0, rx1, ry1))
        render_mask_cropped = Image.fromarray(render_mask_orig[ry0:ry1, rx0:rx1])

        render_resized = resize(render_cropped, target_res=img_size, resize=True, to_pil=True)
        render_mask_resized = np.array(resize(render_mask_cropped, target_res=img_size, resize=True, to_pil=True))
        if render_mask_resized.ndim == 3:
            render_mask_resized = render_mask_resized[..., 0]

        # Load or compute features for this render
        if USE_CACHED_FEATURES:
            feat_render, feat_render_up = load_cached_features(render_path, obj_name, seq_dir, img_size)
            if feat_render is None and FALLBACK_TO_COMPUTE:
                feat_render = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=render_resized)
                feat_render_up = F.interpolate(feat_render, size=(img_size, img_size), mode='bilinear', align_corners=False)
            elif feat_render is None:
                continue
        else:
            feat_render = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=render_resized)
            feat_render_up = F.interpolate(feat_render, size=(img_size, img_size), mode='bilinear', align_corners=False)

        # Compute MNN ratio
        with torch.no_grad():
            feat_render_flat_norm = F.normalize(
                feat_render_up.view(C, H * W).permute(1, 0), p=2, dim=1
            )
            render_fg_flat = (render_mask_resized > 128).astype(np.float32).flatten()
            render_fg_indices = np.where(render_fg_flat > 0.5)[0]

            mnn_ratio = compute_mutual_nn_ratio(
                feat_input_flat_norm, feat_render_flat_norm,
                input_fg_indices, render_fg_indices,
                max_pixels=MAX_PIXELS_FOR_SIMILARITY
            )

        similarity_results.append({
            'type': meta['type'],
            'elevation': meta['elevation'],
            'azimuth': meta['azimuth'],
            'mnn_ratio': mnn_ratio,
            'path': render_path,
        })

        if (r_idx + 1) % 50 == 0 or r_idx == len(all_render_meta) - 1:
            print(f"    Progress: {r_idx+1}/{len(all_render_meta)} renders processed")

        # Keep a CPU copy of the raw features for the stage-2 handoff (detached:
        # stage1 extraction runs with grad on the aggregation net).
        if feature_sink is not None:
            feature_sink[(meta['elevation'], meta['azimuth'])] = feat_render.detach().cpu()

        del feat_render, feat_render_up

    # =========================================================================
    # STEP 4: Rank and report top-K
    # =========================================================================
    similarity_results.sort(key=lambda x: x['mnn_ratio'], reverse=True)

    # Trim the handoff to the TOP_K renders stage 2 will actually use
    # (stage 2 takes the first TOP_K_INPUT_SIMILAR=TOP_K rows of the desc-sorted npz).
    if feature_sink is not None:
        top_keys = {(res['elevation'], res['azimuth']) for res in similarity_results[:TOP_K]}
        for key in list(feature_sink.keys()):
            if key not in top_keys:
                del feature_sink[key]

    print(f"\n  {'='*60}")
    print(f"  Top-{TOP_K} most similar renders to input image:")
    print(f"  {'='*60}")
    for rank, res in enumerate(similarity_results[:TOP_K]):
        print(f"    #{rank+1}  {res['type'].upper():4s}  elevation={res['elevation']:3d}  azimuth={res['azimuth']:3d}  "
              f"MNN={res['mnn_ratio']:.4f}  {os.path.basename(res['path'])}")

    # Save results
    output_dir = os.path.join(seq_dir, 'correspondences')
    # output_dir = os.path.join(seq_dir, 'correspondences_new')
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'render_similarity_to_input.npz')

    np.savez(results_path,
             types=np.array([r['type'] for r in similarity_results]),
             elevations=np.array([r['elevation'] for r in similarity_results]),
             azimuths=np.array([r['azimuth'] for r in similarity_results]),
             mnn_ratios=np.array([r['mnn_ratio'] for r in similarity_results]),
             paths=np.array([r['path'] for r in similarity_results]),
             top_k=TOP_K)
    print(f"  Saved similarity rankings to {results_path}")

    # Also save a quick visualization of top-K
    if len(similarity_results) > 0:
        vis_path = os.path.join(output_dir, 'top_similar_renders.png')
        n_show = min(TOP_K, len(similarity_results))
        fig, axes = plt.subplots(1, n_show + 1, figsize=(4 * (n_show + 1), 4))
        
        # Ensure axes is always iterable
        if n_show + 1 == 1:
            axes = [axes]
        
        axes[0].imshow(np.array(input_img_resized))
        axes[0].set_title('Input Image', fontsize=10, fontweight='bold')
        axes[0].axis('off')

        for rank in range(n_show):
            res = similarity_results[rank]
            # Reload and crop the render for display
            seg_p = res['path'].replace('renders', 'segments').replace('obj', obj_name.replace(' ', '_'))

            r_mask = np.array(Image.open(seg_p).convert('L'))
            r_bbox = compute_bbox_from_mask(r_mask, tolerance=0.1)
            r_pil = Image.open(res['path']).convert('RGB')
            if r_bbox is not None:
                r_pil = r_pil.crop((r_bbox[0], r_bbox[1], r_bbox[2], r_bbox[3]))
            r_pil = resize(r_pil, target_res=img_size, resize=True, to_pil=True)

            axes[rank + 1].imshow(np.array(r_pil))
            axes[rank + 1].set_title(
                f'#{rank+1} {res["type"].upper()} e={res["elevation"]} a={res["azimuth"]}\nMNN={res["mnn_ratio"]:.4f}',
                fontsize=9
            )
            axes[rank + 1].axis('off')

        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved top-{TOP_K} visualization to {vis_path}")
    else:
        print(f"  No valid renders found for comparison, skipping visualization")

    del feat_input, feat_input_upsampled, feat_input_flat_norm
    torch.cuda.empty_cache()
    
    return True


def divide_batches(root_dir, num_batches):
    """Split unprocessed images into ``num_batches`` contiguous batches.

    Skips images missing an ``*_object.png`` mask and sequences whose
    stage-1 output already exists.
    """
    img_list = list(glob.iglob(root_dir + '/**/*.jpg', recursive=False))
    all_img_list = sorted(img_list)
    
    print(f"Total images found: {len(all_img_list)}")

    img_paths_to_process = []
    skipped_count = 0
    
    print("Checking for already processed sequences...")
    no_mask_count = 0
    for img_path in tqdm(all_img_list, desc="Filtering images"):
        seq_dir = os.path.dirname(img_path)
        imgname = os.path.basename(img_path).replace('.jpg', '')
        input_mask_path = os.path.join(seq_dir, f'{imgname}_object.png')
        if not os.path.exists(input_mask_path):
            no_mask_count += 1
        elif is_sequence_already_processed(seq_dir):
            skipped_count += 1
        else:
            img_paths_to_process.append(img_path)

    print(f"Images skipped (no segmentation mask): {no_mask_count}")
    
    print(f"Images to process: {len(img_paths_to_process)}")
    print(f"Images skipped (already processed): {skipped_count}")
    
    total_images = len(img_paths_to_process)
    if total_images == 0:
        return []
        
    batch_size = (total_images + num_batches - 1) // num_batches  # Ceiling division
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batches.append(img_paths_to_process[start_idx:end_idx])
    
    return batches


def load_models():
    """Load the SD + DINO + aggregation models once (same calls/order as the
    legacy inline stage-1 load). Returns a dict with the models plus captured
    np/python RNG states.

    ``load_model`` reseeds the global RNGs via detectron2's ``seed_all_rng(42)``
    and the loads afterwards may consume np.random draws, so each stage's legacy
    inline load left the RNG in a load-dependent state that downstream
    subsampling depends on. To reproduce those exact streams when the load is
    shared, capture the state at the two points the legacy per-stage loads
    ended: after SD+DINO (== stage2's inline load) and after +aggregation net
    (== stage1's inline load).
    """
    import random as _pyrandom
    sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches * 16,
                                  num_timesteps=50, block_indices=[2, 5, 8, 11])
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
    rng_after_sd_dino = (np.random.get_state(), _pyrandom.getstate())
    aggre_net = AggregationNetwork(feature_dims=[640, 1280, 1280, 768], projection_dim=768, device='cuda')
    aggre_net.load_pretrained_weights(torch.load(_CKPT))
    rng_after_full = (np.random.get_state(), _pyrandom.getstate())
    return {
        'sd_model': sd_model, 'sd_aug': sd_aug,
        'extractor_vit': extractor_vit, 'aggre_net': aggre_net,
        'rng_stage1': rng_after_full,      # state a legacy stage-1 load ends in
        'rng_stage2': rng_after_sd_dino,   # state a legacy stage-2 load ends in
    }


def _restore_rng(rng_state):
    """Restore the (np.random, python-random) state captured by load_models()."""
    import random as _pyrandom
    np.random.set_state(rng_state[0])
    _pyrandom.setstate(rng_state[1])


def run(seq_dir, obj_name, overwrite=False, models=None, return_top_features=False):
    """MILO single-sequence entry (geo-aware env): rank H3D views vs the input image.

    Writes correspondences/render_similarity_to_input.npz.

    ``models``: optional dict from :func:`load_models` to skip the per-stage
    load. ``return_top_features=True`` returns
    ``(success, {(elev, azim): raw feature tensor})`` for the top-K renders so
    stage 2 can skip re-extracting them.
    """
    global _OBJ_NAME_OVERRIDE

    def _ret(ok, feats=None):
        return (ok, feats) if return_top_features else ok

    if not overwrite and is_sequence_already_processed(seq_dir):
        print(f"[correspond] stage1 output exists, skipping {os.path.basename(seq_dir)}")
        return _ret(False)
    jpgs = sorted(p for p in glob.glob(os.path.join(seq_dir, '*.jpg'))
                  if not (p.endswith('_human.jpg') or p.endswith('_object.jpg')))
    if not jpgs:
        print(f"[correspond] no input .jpg in {seq_dir}")
        return _ret(False)
    _OBJ_NAME_OVERRIDE = obj_name
    if models is None:
        print("[correspond] stage1: loading SD + DINO models...")
        models = load_models()
    else:
        # Restore the exact RNG state a legacy inline stage-1 load would have
        # left behind (see load_models), so downstream np.random subsampling
        # draws are bit-identical to the unshared-load flow.
        _restore_rng(models['rng_stage1'])
    sd_model, sd_aug = models['sd_model'], models['sd_aug']
    extractor_vit, aggre_net = models['extractor_vit'], models['aggre_net']
    feature_sink = {} if return_top_features else None
    ok = process_single_image(jpgs[0], sd_model, sd_aug, aggre_net, extractor_vit,
                              feature_sink=feature_sink)
    return _ret(ok, feature_sink)


def main(img_list=None):
    """Process each image in ``img_list`` sequentially; prints a run summary."""
    print("="*80)
    print("Render Similarity Comparison Script")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
    aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
    aggre_net.load_pretrained_weights(torch.load(_CKPT))
    print("Models loaded successfully!")
    
    print(f"\nFound {len(img_list)} images to process")
    print(f"Cache usage enabled: {USE_CACHED_FEATURES}")
    print(f"Fallback to compute: {FALLBACK_TO_COMPUTE}")
    print(f"Top-K results: {TOP_K}")
    print()
    
    total_processed = 0
    total_errors = 0
    
    for img_idx, img_path in enumerate(img_list):
        print(f"\n{'='*80}")
        print(f"Processing image {img_idx+1}/{len(img_list)}: {img_path}")
        print(f"{'='*80}")
        
        try:
            success = process_single_image(img_path, sd_model, sd_aug, aggre_net, extractor_vit)
            if success:
                total_processed += 1
            else:
                total_errors += 1
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            total_errors += 1
            continue

        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}")
    print(f"Total processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare input images to multi-view renders")
    parser.add_argument('--idx', type=int, default=None, help='Batch index (0-based)')
    parser.add_argument('--num_batches', type=int, default=4,
                       help='Number of batches to divide the dataset into')
    args = parser.parse_args()
    
    root_dir = os.environ.get("MILO_GEO_TEST_ROOT", "/path/to/data_seq")

    # ---- Batch processing ---------------------------------------------------
    if args.idx is not None:
        batches = divide_batches(
            root_dir=root_dir,
            num_batches=args.num_batches
        )
        
        if len(batches) == 0:
            print("No images to process")
            exit(0)
        
        if args.idx >= len(batches):
            print(f"Error: Batch index {args.idx} out of range (0-{len(batches)-1})")
            exit(1)
        
        img_list = batches[args.idx]
        print(f"\nProcessing batch {args.idx}/{len(batches)-1} with {len(img_list)} images")
        print(f"Images in this batch: {[os.path.basename(s) for s in img_list[:5]]}{'...' if len(img_list) > 5 else ''}\n")
        
        main(img_list=img_list)
    else:
        print("Processing all images (no batch index specified)")
        main(img_list=None)

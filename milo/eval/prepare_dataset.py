"""Prepare an eval dataset for the MILO pipeline.

Reads the dataset's split file (milo/eval/splits/<dataset>_test.txt — one
slash-separated address per line, relative to the dataset root), builds a
MILO per-sequence input dir for exactly those frames from the official dataset
release, and writes a .sh file with one run_pipeline.py call per sequence
(demo config: no --dataset flag; SAM 3 prompts below).

Per-sequence outputs:
  <frame>.jpg  image_human.png  image_object.png  metadata.npz
  gt_human.obj  gt_object.obj  object_template.obj
For InterCap, gt_human.obj is fitted from the dataset-shipped SMPL-X GT (SMPL-X ->
SMPL-H) by calling convert_intercap_gt_human.generate_gt_human — so `intercap` needs a
GPU and the deformation-transfer setup (scripts/setup_transfer_data.sh); hodome / imhd
are CPU-only.

Usage:
  python milo/eval/prepare_dataset.py intercap --raw_root /data/datasets/InterCap      --out_root <dir>
  python milo/eval/prepare_dataset.py hodome   --raw_root /data/datasets/HODome        --out_root <dir>
  python milo/eval/prepare_dataset.py imhd     --raw_root /data/datasets/IMHD-Dataset  --out_root <dir>

Run sequentially (NFS dislikes parallel writers).
"""

import argparse
import json
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings

import cv2
import numpy as np
import torch
import trimesh

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from milo.eval.eval_utils import HodomeSMPLH, ImhdBodyModel, rot6d_to_matrix
from milo.eval.convert_intercap_gt_human import (
    DEFAULT_TRANSFER_DATA,
    generate_gt_human,
    parse_split,
)
from milo.pipeline.config import _INTERCAP_MAPPING

DATASETS = ["intercap", "hodome", "imhd"]

# HODome / IMHD GT humans use the official SMPL+H v1.2 neutral model.npz (MANO
# website); InterCap GT humans come from the shipped SMPL-X fit via
# convert_intercap_gt_human.py, so no InterCap body model is needed here.
SMPLH_NEUTRAL_NPZ = os.path.join(REPO_ROOT, "_DATA", "body_models", "smplh", "neutral", "model.npz")

# InterCap 6-camera Azure Kinect intrinsics — fixed per camera across all
# sequences ((fx, fy), (cx, cy)); frames are used at native resolution.
INTERCAP_INTRINSICS = {
    "Frames_Cam1": ((918.457763671875, 918.4373779296875), (956.9661865234375, 555.944580078125)),
    "Frames_Cam2": ((915.2996215820312, 915.1966552734375), (956.664306640625, 551.6165771484375)),
    "Frames_Cam3": ((912.8626708984375, 912.6763305664062), (956.7200317382812, 554.2166748046875)),
    "Frames_Cam4": ((909.8202514648438, 909.6246948242188), (957.6181640625, 554.6029663085938)),
    "Frames_Cam5": ((920.533447265625, 920.0972290039062), (958.4615478515625, 550.4298706054688)),
    "Frames_Cam6": ((909.1763305664062, 909.2352905273438), (956.1480102539062, 555.0159301757812)),
}

# Working resolutions are pinned by the dataset masks (HODome mask_refine is
# 1280x720, IMHD mask videos are 1920x1080); frames are downscaled to match
# and the intrinsics are scaled accordingly. InterCap frames are used as-is.
HODOME_SIZE = (1280, 720)   # native video 3840x2160
IMHD_SIZE = (1920, 1080)    # native video 3840x2160

# ---------------------------------------------------------------------------
# SAM 3 object prompts
# ---------------------------------------------------------------------------

# InterCap — CARI4D scripts/preprocess_intercap.sh (05 from preprocess_intercap_s9_s10.sh);
# keyed by object id (2nd field of the sequence name).
INTERCAP_PROMPTS = {
    "01": "a green trolleycase",  "02": "a black skateboard",
    "03": "an orange and white soccer ball", "04": "a white dotted umbrella",
    "05": "a red tennis racket",  "06": "a black case",
    "07": "a blue chair",         "08": "a green glass bottle",
    "09": "a white paper cup",    "10": "a blue stool",
}
# HODome / IMHD — authored from the actual frames (see objects/{hodome,imhd}_<noun>.jpg);
# keyed by the object noun parsed from the sequence name.
HODOME_PROMPTS = {
    "baseball": "a wooden baseball bat", "bigsofa": "a white sofa",
    "book": "a white book",              "box": "a brown cardboard box",
    "case": "a black case",              "chair": "a wooden chair",
    "desk": "a white desk",              "flower": "a vase of flowers",
    "keyboard": "a white keyboard",      "monitor": "a black computer monitor",
    "pan": "a white pan",                "pillow": "a white pillow",
    "pingpong": "a red pingpong paddle", "pink": "a pink and white bucket",
    "smallsofa": "a yellow and grey armchair", "table": "a white round table",
    "talltable": "a tall grey wooden table",  "tennis": "a black tennis racket",
    "trashcan": "a white trashcan",     "trolleycase": "a white trolleycase",
}
IMHD_PROMPTS = {
    "bat": "a wooden baseball bat", "broom": "a black and red broom",
    "chair": "a wooden chair",      "dumbbell": "a purple dumbbell",
    "kettlebell": "a pink kettlebell", "pan": "a black pan",
    "suitcase": "a grey trolleycase", "tennis": "a black tennis racket",
}

SEQ_OUTPUTS = ["image_human.png", "image_object.png", "metadata.npz",
               "gt_human.obj", "gt_object.obj", "object_template.obj"]
# InterCap gt_human.obj is produced separately by convert_intercap_gt_human.py.
INTERCAP_OUTPUTS = [f for f in SEQ_OUTPUTS if f != "gt_human.obj"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seq_name(parts):
    return "__".join(parts)


def require_raw_root(dataset, raw_root, marker, hint):
    """Fail early with a clear message when --raw_root doesn't look like the release."""
    if not os.path.exists(os.path.join(raw_root, marker)):
        raise SystemExit(f"[{dataset}] {raw_root} is missing '{marker}' — "
                         f"--raw_root must be {hint}")


def load_template(path):
    """Load a template .obj for its geometry only. trimesh's OBJ parser emits
    All-NaN-slice / invalid-cast RuntimeWarnings while unmerging texture faces on
    some scanned templates; the vertices/faces are unaffected (verified), so mute
    that noise here rather than per-object across a full run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return trimesh.load(path, process=False, maintain_order=True)


def seq_complete(seq_dir, frame, outputs=SEQ_OUTPUTS):
    files = [f"{frame}.jpg"] + outputs
    return all(os.path.exists(os.path.join(seq_dir, f)) for f in files)


def mask_bbox(mask_hum, mask_obj):
    combined = np.clip(mask_hum.astype(np.int64) + mask_obj.astype(np.int64), 0, 255)
    ys, xs = np.where(combined > 0)
    if len(xs) == 0:
        return None
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])


def ffmpeg_extract(video, frames, out_dir, fmt):
    """Extract only the given native frame indices, named %06d.<fmt> by index."""
    frames = sorted(set(int(n) for n in frames))
    sel = "+".join(f"eq(n\\,{n})" for n in frames)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
           "-vf", f"select='{sel}'", "-vsync", "0"]
    if fmt in ("jpg", "jpeg"):
        cmd += ["-qscale:v", "1"]
    cmd += [os.path.join(out_dir, f"tmp_%06d.{fmt}")]
    subprocess.run(cmd, check=True)
    # select emits the surviving frames in temporal order numbered 1..k (indices
    # past the end of the video are simply absent), independent of the video's
    # timebase — rename rank -> frame index. (-frame_pts names by pts, which only
    # equals the frame index for CFR video in a 1/fps timebase.)
    for rank, n in enumerate(frames, 1):
        src = os.path.join(out_dir, f"tmp_{rank:06d}.{fmt}")
        if os.path.exists(src):
            os.rename(src, os.path.join(out_dir, f"{n:06d}.{fmt}"))


# ---------------------------------------------------------------------------
# InterCap
# ---------------------------------------------------------------------------

def process_intercap(entries, raw_root, out_root, overwrite,
                     transfer_data=DEFAULT_TRANSFER_DATA, gpu=0, maxiters=200, batch=32):
    require_raw_root("intercap", raw_root, "sequences",
                     "the InterCap release root (holding sequences/ and objects/)")

    done = []
    for parts in entries:
        p, o, seg, cam, _, frame = parts
        seq = seq_name(parts)
        seq_dir = os.path.join(out_root, seq)
        if not overwrite and seq_complete(seq_dir, frame, INTERCAP_OUTPUTS):
            done.append(seq)
            continue

        base = os.path.join(raw_root, "sequences", p, o, seg)
        img_src = os.path.join(base, cam, "color", f"{frame}.jpg")
        hum_mask_src = os.path.join(base, cam, "mask", f"{frame}_human.png")
        obj_mask_src = os.path.join(base, cam, "mask", f"{frame}_object.png")
        # GT object: InterCap's shipped posed object mesh (world frame), one per
        # segment-frame. GT human is fitted below via generate_gt_human.
        obj_mesh_src = os.path.join(base, "Mesh", f"{frame}_second_obj.ply")
        template_src = os.path.join(raw_root, "objects", f"{o}.obj")
        srcs = [img_src, hum_mask_src, obj_mask_src, obj_mesh_src, template_src]
        if not all(os.path.exists(f) for f in srcs):
            missing = [os.path.basename(f) for f in srcs if not os.path.exists(f)]
            print(f"[skip] {seq}: missing {', '.join(missing)}")
            continue

        os.makedirs(seq_dir, exist_ok=True)
        shutil.copy(img_src, os.path.join(seq_dir, f"{frame}.jpg"))
        shutil.copy(hum_mask_src, os.path.join(seq_dir, "image_human.png"))
        shutil.copy(obj_mask_src, os.path.join(seq_dir, "image_object.png"))
        shutil.copy(template_src, os.path.join(seq_dir, "object_template.obj"))

        focal, princpt = INTERCAP_INTRINSICS[cam]
        metadata = {"focal_length": np.array(focal, dtype=np.float32),
                    "principal_point": np.array(princpt, dtype=np.float32)}
        bbox = mask_bbox(cv2.imread(hum_mask_src, cv2.IMREAD_GRAYSCALE),
                         cv2.imread(obj_mask_src, cv2.IMREAD_GRAYSCALE))
        if bbox is not None:
            metadata["bbox"] = bbox
        np.savez(os.path.join(seq_dir, "metadata.npz"), **metadata)

        obj_mesh = load_template(obj_mesh_src)
        trimesh.Trimesh(vertices=obj_mesh.vertices, faces=obj_mesh.faces, process=False) \
            .export(os.path.join(seq_dir, "gt_object.obj"))
        done.append(seq)

    # gt_human: fit the shipped SMPL-X GT to SMPL-H (GPU). Reuses the logic in
    # convert_intercap_gt_human.py; only the meshes for `done` sequences are needed.
    done_set = set(done)
    generate_gt_human([p for p in entries if seq_name(p) in done_set], raw_root, out_root,
                      transfer_data=transfer_data, gpu=gpu, maxiters=maxiters,
                      batch=batch, overwrite=overwrite)
    return done


# ---------------------------------------------------------------------------
# HODome
# ---------------------------------------------------------------------------

def process_hodome(entries, raw_root, out_root, overwrite):
    require_raw_root("hodome", raw_root, "dataset_information.json",
                     "the HODome (NeuralDome) release root")
    with open(os.path.join(raw_root, "dataset_information.json")) as f:
        dataset_info = json.load(f)
    calibration_dates = [d for d in dataset_info if re.fullmatch(r"\d{8}", d)]
    body_model = HodomeSMPLH(SMPLH_NEUTRAL_NPZ)

    groups = {}
    for parts in entries:
        groups.setdefault((parts[0], parts[1]), []).append(parts[2])

    done = []
    for (sname, cam), frames in sorted(groups.items()):
        pending = [f for f in frames
                   if overwrite or not seq_complete(os.path.join(out_root, seq_name([sname, cam, f])), f)]
        done += [seq_name([sname, cam, f]) for f in frames if f not in pending]
        if not pending:
            continue

        date = next((d for d in calibration_dates if sname in dataset_info.get(d, [])), None)
        if date is None:
            print(f"[skip] {sname}: no calibration date in dataset_information.json")
            continue
        with open(os.path.join(raw_root, "calibration", date, "calibration.json")) as f:
            camera_params = json.load(f)
        cam_idx = str(int(cam) - 1)  # video folder is 1-indexed, calibration 0-indexed
        K = np.array(camera_params[cam_idx]["K"]).reshape(3, 3)
        K /= 3840 / HODOME_SIZE[0]
        K[2, 2] = 1
        RT = np.array(camera_params[cam_idx]["RT"]).reshape(4, 4)
        R, T = RT[:3, :3], RT[:3, 3]

        obj = sname.split("_")[-1]
        template_path = os.path.join(raw_root, "scaned_object", obj, f"{obj}_face1000.obj")
        template = load_template(template_path)
        template_verts = np.asarray(template.vertices).copy()

        video = os.path.join(raw_root, "videos", sname, f"data{cam}.mp4")
        if not os.path.exists(video):
            print(f"[skip] {sname}__{cam}: missing video {video}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg_extract(video, [int(f) for f in pending], tmp, "jpg")
            for frame in pending:
                seq = seq_name([sname, cam, frame])
                seq_dir = os.path.join(out_root, seq)
                frame_src = os.path.join(tmp, f"{int(frame):06d}.jpg")
                hum_mask_src = os.path.join(raw_root, "mask_refine", sname, "human", cam_idx, f"{frame}.png")
                obj_mask_src = os.path.join(raw_root, "mask_refine", sname, "object", cam_idx, f"{frame}.png")
                smpl_src = os.path.join(raw_root, "mocap", sname, "smplh", f"{frame}.json")
                obj_rt_src = os.path.join(raw_root, "mocap", sname, "object", "refine", "json", f"{frame}.json")
                missing = [f for f in (frame_src, hum_mask_src, obj_mask_src, smpl_src, obj_rt_src)
                           if not os.path.exists(f)]
                if missing:
                    print(f"[skip] {seq}: missing {', '.join(os.path.basename(m) for m in missing)}")
                    continue

                os.makedirs(seq_dir, exist_ok=True)
                img = cv2.imread(frame_src)
                if img.shape[1::-1] == HODOME_SIZE:  # already at mask resolution — keep as-is
                    shutil.copy(frame_src, os.path.join(seq_dir, f"{frame}.jpg"))
                else:
                    cv2.imwrite(os.path.join(seq_dir, f"{frame}.jpg"), cv2.resize(img, HODOME_SIZE))
                shutil.copy(hum_mask_src, os.path.join(seq_dir, "image_human.png"))
                shutil.copy(obj_mask_src, os.path.join(seq_dir, "image_object.png"))

                metadata = {"focal_length": K[:2, :2].diagonal(),
                            "principal_point": K[:2, 2], "extrinsics": RT}
                bbox = mask_bbox(cv2.imread(hum_mask_src, cv2.IMREAD_GRAYSCALE),
                                 cv2.imread(obj_mask_src, cv2.IMREAD_GRAYSCALE))
                if bbox is not None:
                    metadata["bbox"] = bbox
                np.savez(os.path.join(seq_dir, "metadata.npz"), **metadata)

                # GT human: easymocap SMPL-H forward, then camera extrinsics.
                with open(smpl_src) as f:
                    smpl_params = json.load(f)["annots"][0]
                verts = body_model(**smpl_params).dot(R.T) + T
                trimesh.Trimesh(vertices=verts, faces=body_model.faces) \
                    .export(os.path.join(seq_dir, "gt_human.obj"))

                # GT object: refined 6D rotation + translation, then camera extrinsics.
                with open(obj_rt_src) as f:
                    refine = json.load(f)
                obj_R = rot6d_to_matrix(torch.from_numpy(np.array(refine["object_R"]))) \
                    .numpy().reshape(3, 3).T
                obj_T = np.array(refine["object_T"]).reshape(1, 3)
                obj_verts = (template_verts.dot(obj_R.T) + obj_T).dot(R.T) + T
                trimesh.Trimesh(vertices=obj_verts, faces=template.faces, process=False) \
                    .export(os.path.join(seq_dir, "gt_object.obj"))

                shutil.copy(template_path, os.path.join(seq_dir, "object_template.obj"))
                done.append(seq)
    return done


# ---------------------------------------------------------------------------
# IMHD
# ---------------------------------------------------------------------------

def _imhd_find_gt_pkl(gt_dir, frame_num):
    if not os.path.isdir(gt_dir):
        return None, None
    for pkl_file in sorted(os.listdir(gt_dir)):
        m = re.match(r"gt_(\d+)_(\d+)_(-?\d+)\.pkl", pkl_file)
        if not m:
            continue
        start, end = int(m.group(2)), int(m.group(3))
        covers = (frame_num >= start) if end == -1 else (start <= frame_num <= end)
        if covers:
            return pkl_file, frame_num - start
    return None, None


def process_imhd(entries, raw_root, out_root, overwrite):
    require_raw_root("imhd", raw_root, "calibrations",
                     "the IMHD² release root (e.g. /data/datasets/IMHD-Dataset, not /data/datasets/IMHD)")
    body_model = ImhdBodyModel(SMPLH_NEUTRAL_NPZ)

    groups = {}
    for parts in entries:
        date, segment, sequence, cam, frame = parts
        groups.setdefault((date, segment, sequence, cam), []).append(frame)

    done = []
    for (date, segment, sequence, cam), frames in sorted(groups.items()):
        pending = [f for f in frames
                   if overwrite or not seq_complete(
                       os.path.join(out_root, seq_name([date, segment, sequence, cam, f])), f)]
        done += [seq_name([date, segment, sequence, cam, f]) for f in frames if f not in pending]
        if not pending:
            continue

        with open(os.path.join(raw_root, "calibrations", date, "extrin.json")) as f:
            extrin = json.load(f)
        with open(os.path.join(raw_root, "calibrations", date, "intrin.json")) as f:
            intrin = json.load(f)["color"]
        K = np.array([intrin["fx"], 0, intrin["cx"], 0, intrin["fy"], intrin["cy"], 0, 0, 1]).reshape(3, 3)
        K /= 3840 / IMHD_SIZE[0]
        K[2, 2] = 1
        RT = np.eye(4)
        RT[:3, :3] = np.array(extrin["rotation"]).reshape(3, 3)
        RT[:3, 3] = np.array(extrin["translation"]).reshape(3)

        obj = segment.split("_")[2]
        obj = "baseball" if obj == "bat" else obj
        template_path = os.path.join(raw_root, "object_templates", obj, f"{obj}_simplified_transformed.obj")
        template = load_template(template_path)
        template_verts = np.asarray(template.vertices, dtype=np.float32)
        template_verts = template_verts - template_verts.mean(axis=0)

        seq_root = os.path.join(raw_root, "video_release", date, segment, sequence)
        mask_root = os.path.join(raw_root, "mask_release", date, segment, sequence)
        gt_dir = os.path.join(raw_root, "ground_truth", date, segment, sequence)
        gt_cache = {}

        # data videos are 1-indexed (data1..data32), mask videos 0-indexed (0..31).
        mask_vid = f"{int(cam) - 1}.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            idxs = [int(f) for f in pending]
            for sub, video, fmt in (("img", os.path.join(seq_root, f"data{cam}.mp4"), "jpg"),
                                    ("hum", os.path.join(mask_root, "human_mask", mask_vid), "png"),
                                    ("obj", os.path.join(mask_root, "object_mask", mask_vid), "png")):
                os.makedirs(os.path.join(tmp, sub), exist_ok=True)
                if not os.path.exists(video):
                    print(f"[warn] missing video: {video}")
                    continue
                ffmpeg_extract(video, idxs, os.path.join(tmp, sub), fmt)

            for frame in pending:
                seq = seq_name([date, segment, sequence, cam, frame])
                seq_dir = os.path.join(out_root, seq)
                frame_num = int(frame)

                pkl_file, rel_frame = _imhd_find_gt_pkl(gt_dir, frame_num)
                if pkl_file is None:
                    print(f"[skip] {seq}: no ground-truth pkl covers frame {frame_num}")
                    continue
                if pkl_file not in gt_cache:
                    with open(os.path.join(gt_dir, pkl_file), "rb") as f:
                        gt_cache[pkl_file] = pickle.load(f)
                gt = gt_cache[pkl_file]

                srcs = {sub: os.path.join(tmp, sub, f"{frame_num:06d}.{fmt}")
                        for sub, fmt in (("img", "jpg"), ("hum", "png"), ("obj", "png"))}
                missing = [s for s in srcs.values() if not os.path.exists(s)]
                if missing:
                    print(f"[skip] {seq}: frame {frame_num} not extracted from video/masks")
                    continue

                os.makedirs(seq_dir, exist_ok=True)
                img = cv2.imread(srcs["img"])
                if img.shape[1::-1] == IMHD_SIZE:  # already at mask resolution — keep as-is
                    shutil.copy(srcs["img"], os.path.join(seq_dir, f"{frame}.jpg"))
                else:
                    cv2.imwrite(os.path.join(seq_dir, f"{frame}.jpg"), cv2.resize(img, IMHD_SIZE))
                masks = {}
                for sub, name in (("hum", "image_human.png"), ("obj", "image_object.png")):
                    mask = cv2.imread(srcs[sub], cv2.IMREAD_GRAYSCALE)
                    # H.264 decode leaves a gray halo around the silhouette — binarize.
                    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
                    if mask.shape[::-1] != IMHD_SIZE:
                        mask = cv2.resize(mask, IMHD_SIZE, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(seq_dir, name), mask)
                    masks[sub] = mask

                metadata = {"focal_length": K[:2, :2].diagonal(),
                            "principal_point": K[:2, 2], "extrinsics": RT}
                bbox = mask_bbox(masks["hum"], masks["obj"])
                if bbox is not None:
                    metadata["bbox"] = bbox
                np.savez(os.path.join(seq_dir, "metadata.npz"), **metadata)

                # GT human: egoego SMPL-H forward (already in camera/world frame).
                verts, faces = body_model(
                    betas=torch.from_numpy(gt["smplShape"][rel_frame]).unsqueeze(0).float(),
                    root_orient=torch.from_numpy(gt["smplPose"][rel_frame][:3]).unsqueeze(0).float(),
                    pose_body=torch.from_numpy(gt["smplPose"][rel_frame][3:66]).unsqueeze(0).float(),
                    pose_hand=torch.from_numpy(gt["smplHandPose"][rel_frame]).unsqueeze(0).float(),
                    trans=torch.from_numpy(gt["smplTrans"][rel_frame]).unsqueeze(0).float())
                trimesh.Trimesh(vertices=verts, faces=faces, process=False) \
                    .export(os.path.join(seq_dir, "gt_human.obj"))

                # GT object: mean-centered template posed with the GT rotation/translation.
                rot, _ = cv2.Rodrigues(np.asarray(gt["objectRot"][rel_frame], dtype=np.float64))
                obj_verts = template_verts.dot(rot.T) + gt["objectTrans"][rel_frame].reshape(1, 3)
                trimesh.Trimesh(vertices=obj_verts, faces=template.faces,
                                process=False, maintain_order=True) \
                    .export(os.path.join(seq_dir, "gt_object.obj"))

                shutil.copy(template_path, os.path.join(seq_dir, "object_template.obj"))
                done.append(seq)
    return done


# ---------------------------------------------------------------------------
# Job script
# ---------------------------------------------------------------------------

def object_label(dataset, seq):
    if dataset == "intercap":
        return _INTERCAP_MAPPING[int(seq.split("__")[1]) - 1]
    if dataset == "hodome":
        return seq.split("__")[0].split("_")[-1]
    return seq.split("__")[1].split("_")[2]  # imhd


def object_prompt(dataset, seq):
    if dataset == "intercap":
        return INTERCAP_PROMPTS[seq.split("__")[1]]
    if dataset == "hodome":
        return HODOME_PROMPTS[seq.split("__")[0].split("_")[-1]]
    return IMHD_PROMPTS[seq.split("__")[1].split("_")[2]]  # imhd


def write_job_sh(dataset, seqs, out_root, out_sh):
    out_root = os.path.abspath(out_root)
    lines = ["#!/usr/bin/env bash",
             f"# {dataset} eval jobs — generated by milo/eval/prepare_dataset.py",
             f"# {shlex.join(sys.argv)}"]
    for seq in sorted(seqs):
        lines.append(shlex.join([
            "python", "milo/pipeline/run_pipeline.py",
            "--data_root", out_root, "--seq", seq,
            "--object", object_label(dataset, seq),
            "--object_prompt", object_prompt(dataset, seq),
            "--template", os.path.join(out_root, seq, "object_template.obj"),
        ]))
    with open(out_sh, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(out_sh, 0o755)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--raw_root", required=True,
                        help="Official dataset root (InterCap / HODome / IMHD-Dataset)")
    parser.add_argument("--out_root", required=True,
                        help="Output data_root; one sequence dir is created per split entry")
    parser.add_argument("--splits", default=None,
                        help="Split file (default: milo/eval/splits/<dataset>_test.txt)")
    parser.add_argument("--out_sh", default=None,
                        help="Output job script (default: run_<dataset>_eval.sh)")
    parser.add_argument("--seqs", default=None,
                        help="Comma-separated subset of sequence names (with __) to process")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild sequence dirs even when their outputs already exist")
    # InterCap gt_human fit (SMPL-X -> SMPL-H); ignored for hodome / imhd.
    parser.add_argument("--transfer_data", default=DEFAULT_TRANSFER_DATA,
                        help="[intercap] dir with smplx2smplh_deftrafo_setup.pkl "
                             "(default _DATA/transfer_data; see scripts/setup_transfer_data.sh)")
    parser.add_argument("--gpu", type=int, default=0, help="[intercap] GPU for the gt_human fit")
    parser.add_argument("--maxiters", type=int, default=200, help="[intercap] LBFGS iters")
    parser.add_argument("--batch", type=int, default=32, help="[intercap] meshes per gendered fit")
    args = parser.parse_args()

    splits = args.splits or os.path.join(REPO_ROOT, "milo", "eval", "splits", f"{args.dataset}_test.txt")
    out_sh = args.out_sh or f"run_{args.dataset}_eval.sh"
    entries = parse_split(splits)
    if args.seqs:
        keep = set(args.seqs.split(","))
        entries = [p for p in entries if seq_name(p) in keep]
        print(f"[warn] --seqs: {out_sh} will be rewritten with only the "
              f"{len(entries)} selected sequences")
    print(f"{args.dataset}: {len(entries)} sequences from {splits}")
    os.makedirs(args.out_root, exist_ok=True)

    if args.dataset == "intercap":
        done = process_intercap(entries, args.raw_root, args.out_root, args.overwrite,
                                transfer_data=args.transfer_data, gpu=args.gpu,
                                maxiters=args.maxiters, batch=args.batch)
    else:
        process = {"hodome": process_hodome, "imhd": process_imhd}
        done = process[args.dataset](entries, args.raw_root, args.out_root, args.overwrite)

    write_job_sh(args.dataset, done, args.out_root, out_sh)
    skipped = len(entries) - len(done)
    print(f"prepared {len(done)} sequences ({skipped} skipped) -> {args.out_root}")
    print(f"wrote {out_sh} ({len(done)} lines)")


if __name__ == "__main__":
    main()

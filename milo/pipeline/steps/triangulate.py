"""
pipeline/steps/triangulate.py

Multi-view triangulation of 2D body and hand keypoints → 3D keypoints.
Outputs keypoints_3d[_highres][_ransac].npy plus _lhand/_rhand variants,
triangulation_stats*.npz, and a colored skeleton OBJ.

The triangulation core (process_all_sequences and helpers) lives in this module;
run() / __main__ below wrap it as a single-sequence pipeline step.

Standalone usage:
    python -m milo.pipeline.steps.triangulate --seq_dir /path/to/seq

Module usage:
    from milo.pipeline.steps.triangulate import run
    run(seq_dir="/path/to/seq")
"""

import os
import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from tqdm import tqdm
import trimesh
from itertools import combinations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from pytorch3d.renderer import look_at_view_transform

from milo.pipeline.steps._log import vprint, set_verbose


# ============================================================
# Triangulation core
# ============================================================


def triangulation_residuals(x, imgpoints, cams):
    """
    Reprojection-residual function for least_squares triangulation.
    Projects through all cameras in one batched numpy call.

    Args:
        x: 3D point to optimize [x, y, z]
        imgpoints: {view_id: [u, v]} 2D keypoint coordinates
        cams: {view_id: camera_params} camera parameters

    Returns:
        Flattened (N*2,) array of per-view [dx, dy] reprojection errors (pixels)
    """
    view_ids = list(imgpoints.keys())
    if not view_ids:
        return np.array([])

    X = x.reshape(3, 1)

    # Prefer the cached 'R' matrix; fall back to cv2.Rodrigues on 'r'.
    Rs = []
    for v in view_ids:
        if 'R' in cams[v]:
            Rs.append(cams[v]['R'])
        else:
            R, _ = cv2.Rodrigues(cams[v]['r'])
            Rs.append(R)

    Rs  = np.stack(Rs)                                              # (N, 3, 3)
    ts  = np.stack([cams[v]['t'].ravel() for v in view_ids])        # (N, 3)
    fxs = np.array([cams[v]['mtx'][0, 0] for v in view_ids])
    fys = np.array([cams[v]['mtx'][1, 1] for v in view_ids])
    cxs = np.array([cams[v]['mtx'][0, 2] for v in view_ids])
    cys = np.array([cams[v]['mtx'][1, 2] for v in view_ids])
    obs = np.stack([imgpoints[v] for v in view_ids])                # (N, 2)

    p_cam = (Rs @ X).squeeze(-1) + ts                              # (N, 3)
    x_proj = fxs * p_cam[:, 0] / p_cam[:, 2] + cxs
    y_proj = fys * p_cam[:, 1] / p_cam[:, 2] + cys

    residuals = np.column_stack([x_proj - obs[:, 0],
                                  y_proj - obs[:, 1]])              # (N, 2)
    return residuals.ravel()


def triangulate_from_two_views(view_id1, view_id2, imgpoints, cams):
    """
    Triangulate a 3D point from exactly two views (LM least squares from origin).

    Returns:
        3D point as numpy array [x, y, z]
    """
    imgpoints_subset = {view_id1: imgpoints[view_id1], view_id2: imgpoints[view_id2]}
    cams_subset      = {view_id1: cams[view_id1],      view_id2: cams[view_id2]}

    result = least_squares(
        triangulation_residuals,
        np.array([0.0, 0.0, 0.0]),
        args=(imgpoints_subset, cams_subset),
        method='lm'
    )
    return result.x


def _project_point_batch(point_3d, view_ids, cams):
    """
    Project a single 3D point through multiple cameras in one batched
    numpy call.

    Returns (x_proj, y_proj) each of shape (N,).
    """
    X = point_3d.reshape(3, 1)

    Rs = []
    for v in view_ids:
        if 'R' in cams[v]:
            Rs.append(cams[v]['R'])
        else:
            R, _ = cv2.Rodrigues(cams[v]['r'])
            Rs.append(R)

    Rs  = np.stack(Rs)                                           # (N, 3, 3)
    ts  = np.stack([cams[v]['t'].ravel() for v in view_ids])     # (N, 3)
    fxs = np.array([cams[v]['mtx'][0, 0] for v in view_ids])
    fys = np.array([cams[v]['mtx'][1, 1] for v in view_ids])
    cxs = np.array([cams[v]['mtx'][0, 2] for v in view_ids])
    cys = np.array([cams[v]['mtx'][1, 2] for v in view_ids])

    p_cam  = (Rs @ X).squeeze(-1) + ts                          # (N, 3)
    x_proj = fxs * p_cam[:, 0] / p_cam[:, 2] + cxs
    y_proj = fys * p_cam[:, 1] / p_cam[:, 2] + cys
    return x_proj, y_proj


def count_inliers(point_3d, imgpoints, cams, reproj_threshold, exclude_views=None):
    """
    Count how many views are inliers for a given 3D point
    (single batched projection over all views).

    Args:
        point_3d: 3D point [x, y, z]
        imgpoints: Dictionary of {view_id: [u, v]} 2D keypoint coordinates
        cams: Dictionary of {view_id: camera_params} camera parameters
        reproj_threshold: Maximum reprojection error to be considered an inlier
        exclude_views: Set of view IDs to exclude from counting

    Returns:
        inlier_views: List of view IDs that are inliers
        inlier_errors: List of corresponding reprojection errors
    """
    if exclude_views is None:
        exclude_views = set()

    view_ids = [v for v in imgpoints if v not in exclude_views]
    if not view_ids:
        return [], []

    x_proj, y_proj = _project_point_batch(point_3d, view_ids, cams)
    obs    = np.stack([imgpoints[v] for v in view_ids])             # (N, 2)
    errors = np.sqrt((x_proj - obs[:, 0])**2 + (y_proj - obs[:, 1])**2)

    inlier_views  = [v for v, e in zip(view_ids, errors) if e <= reproj_threshold]
    inlier_errors = [float(e) for e in errors if e <= reproj_threshold]
    return inlier_views, inlier_errors


def triangulate_with_ransac(imgpoints, cams, reproj_threshold=5.0, min_views=2):
    """
    Triangulate a 3D point using RANSAC on view pairs.

    Args:
        imgpoints: Dictionary of {view_id: [u, v]} 2D keypoint coordinates
        cams: Dictionary of {view_id: camera_params} camera parameters
        reproj_threshold: Reprojection error threshold for inliers (in pixels)
        min_views: Minimum number of inlier views required

    Returns:
        point_3d: Triangulated 3D point [x, y, z]
        inlier_views: List of view IDs used in final triangulation
        cost: Final optimization cost
    """
    view_ids = list(imgpoints.keys())

    if len(view_ids) < 2:
        return None, [], float('inf')

    best_inliers      = []
    best_point        = None
    best_num_inliers  = 0

    for view_id1, view_id2 in combinations(view_ids, 2):
        point_3d = triangulate_from_two_views(view_id1, view_id2, imgpoints, cams)

        inlier_views, _ = count_inliers(
            point_3d, imgpoints, cams, reproj_threshold, exclude_views=None
        )

        if len(inlier_views) > best_num_inliers:
            best_num_inliers = len(inlier_views)
            best_inliers     = inlier_views
            best_point       = point_3d

    if best_num_inliers >= min_views and best_point is not None:
        imgpoints_inliers = {v: imgpoints[v] for v in best_inliers}
        cams_inliers      = {v: cams[v]      for v in best_inliers}

        result = least_squares(
            triangulation_residuals,
            best_point,
            args=(imgpoints_inliers, cams_inliers),
            method='lm'
        )
        return result.x, best_inliers, result.cost

    return None, [], float('inf')


# ============================================================
# Mesh generation helpers
# ============================================================


def create_sphere(center, radius, color, subdivisions=2):
    """
    Create a colored UV sphere mesh

    Args:
        center: (x, y, z) center position
        radius: sphere radius
        color: (r, g, b) color values in 0-1 range
        subdivisions: number of subdivisions (higher = smoother)

    Returns:
        vertices (with colors), faces as numpy arrays
    """
    u = np.linspace(0, 2 * np.pi, 8 * (2**subdivisions))
    v = np.linspace(0, np.pi, 4 * (2**subdivisions))

    vertices = []
    for i in range(len(v)):
        for j in range(len(u)):
            x = radius * np.sin(v[i]) * np.cos(u[j]) + center[0]
            y = radius * np.sin(v[i]) * np.sin(u[j]) + center[1]
            z = radius * np.cos(v[i]) + center[2]
            vertices.append([x, y, z, color[0], color[1], color[2]])

    vertices = np.array(vertices)

    faces = []
    cols  = len(u)
    rows  = len(v)

    for i in range(rows - 1):
        for j in range(cols - 1):
            v1 = i * cols + j
            v2 = v1 + 1
            v3 = v1 + cols
            v4 = v3 + 1

            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    return vertices, np.array(faces)


def create_cylinder(start, end, radius, color, subdivisions=8):
    """
    Create a colored cylinder between two points

    Args:
        start: (x, y, z) start position
        end: (x, y, z) end position
        radius: cylinder radius
        color: (r, g, b) color values in 0-1 range
        subdivisions: number of radial subdivisions

    Returns:
        vertices (with colors), faces as numpy arrays
    """
    start = np.array(start)
    end   = np.array(end)

    direction = end - start
    length    = np.linalg.norm(direction)

    if length < 1e-6:
        return np.array([]), np.array([])

    direction = direction / length

    if abs(direction[0]) < 0.9:
        perpendicular = np.cross(direction, [1, 0, 0])
    else:
        perpendicular = np.cross(direction, [0, 1, 0])
    perpendicular  = perpendicular / np.linalg.norm(perpendicular)
    perpendicular2 = np.cross(direction, perpendicular)

    vertices = []

    for i in range(subdivisions):
        angle  = 2 * np.pi * i / subdivisions
        offset = radius * (np.cos(angle) * perpendicular + np.sin(angle) * perpendicular2)

        v_start = start + offset
        vertices.append([v_start[0], v_start[1], v_start[2], color[0], color[1], color[2]])

        v_end = end + offset
        vertices.append([v_end[0], v_end[1], v_end[2], color[0], color[1], color[2]])

    vertices = np.array(vertices)

    faces = []
    for i in range(subdivisions):
        next_i = (i + 1) % subdivisions

        v1 = i * 2
        v2 = i * 2 + 1
        v3 = next_i * 2
        v4 = next_i * 2 + 1

        faces.append([v1, v3, v2])
        faces.append([v2, v3, v4])

    return vertices, np.array(faces)


# ============================================================
# Skeleton definitions — body (COCO25 OpenPose) + hands (MANO)
# ============================================================


def get_skeleton_definition():
    """
    Define skeleton connectivity for COCO25 OpenPose format (25 keypoints).
    Returns dictionary of {part_name: [(start_idx, end_idx), ...]}

    COCO25 OpenPose 25-keypoint format:
    0-18: Body keypoints (19 keypoints including Neck and MidHip)
    19-24: Feet keypoints (6 keypoints: left big toe, small toe, heel, right big toe, small toe, heel)

    Keypoint indices:
    0: Nose, 1: Neck, 2: R-Shoulder, 3: R-Elbow, 4: R-Wrist
    5: L-Shoulder, 6: L-Elbow, 7: L-Wrist, 8: MidHip, 9: R-Hip
    10: R-Knee, 11: R-Ankle, 12: L-Hip, 13: L-Knee, 14: L-Ankle
    15: R-Eye, 16: L-Eye, 17: R-Ear, 18: L-Ear
    19: L-BigToe, 20: L-SmallToe, 21: L-Heel
    22: R-BigToe, 23: R-SmallToe, 24: R-Heel
    """
    skeleton = {
        'head':       [(1, 0), (0, 15), (0, 16), (15, 17), (16, 18)],
        'torso':      [(1, 2), (1, 5), (1, 8), (8, 9), (8, 12)],
        'right_arm':  [(2, 3), (3, 4), (2, 17)],
        'left_arm':   [(5, 6), (6, 7), (5, 18)],
        'right_leg':  [(9, 10), (10, 11)],
        'left_leg':   [(12, 13), (13, 14)],
        'left_foot':  [(14, 19), (14, 21), (19, 20)],
        'right_foot': [(11, 22), (11, 24), (22, 23)],
    }
    return skeleton


def get_color_scheme():
    """
    Define color scheme for body parts (COCO25 format).
    Returns dictionary mapping body part to RGB color (0-1 range)
    """
    colors = {
        'head':       [1.0, 0.2, 0.2],
        'torso':      [0.2, 0.8, 0.2],
        'right_arm':  [0.6, 0.2, 1.0],
        'left_arm':   [0.2, 0.4, 1.0],
        'right_leg':  [1.0, 0.5, 0.0],
        'left_leg':   [0.0, 0.8, 0.8],
        'left_foot':  [0.8, 0.8, 0.2],
        'right_foot': [0.8, 0.4, 0.6],
        'joint':      [0.9, 0.9, 0.9],
    }
    return colors


def get_mano_skeleton_definition():
    """
    Define MANO hand skeleton connectivity (21 keypoints per hand).

    MANO 21 keypoint ordering:
        0  : wrist
        1-4  : index  finger  (MCP → PIP → DIP → tip)
        5-8  : middle finger  (MCP → PIP → DIP → tip)
        9-12 : ring   finger  (MCP → PIP → DIP → tip)
        13-16: pinky  finger  (MCP → PIP → DIP → tip)
        17-20: thumb           (CMC → MCP → IP  → tip)

    Returns:
        Dictionary of {part_name: [(start_idx, end_idx), ...]}
    """
    skeleton = {
        'wrist':  [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)],
        'index':  [(1, 2), (2, 3), (3, 4)],
        'middle': [(5, 6), (6, 7), (7, 8)],
        'ring':   [(9, 10), (10, 11), (11, 12)],
        'pinky':  [(13, 14), (14, 15), (15, 16)],
        'thumb':  [(17, 18), (18, 19), (19, 20)],
    }
    return skeleton


def get_left_hand_color_scheme():
    """
    Cool blue-toned palette for left hand.
    Each finger gets a distinct hue so bones are individually readable.
    'joint' is the sphere color shared by all left-hand keypoints.
    """
    return {
        'wrist':  [0.4, 0.7, 0.9],
        'index':  [0.1, 0.5, 1.0],
        'middle': [0.1, 0.8, 0.8],
        'ring':   [0.2, 0.9, 0.5],
        'pinky':  [0.3, 0.6, 1.0],
        'thumb':  [0.5, 0.3, 1.0],
        'joint':  [0.7, 0.85, 0.95],
    }


def get_right_hand_color_scheme():
    """
    Warm red/orange-toned palette for right hand.
    Each finger gets a distinct hue so bones are individually readable.
    'joint' is the sphere color shared by all right-hand keypoints.
    """
    return {
        'wrist':  [0.9, 0.5, 0.3],
        'index':  [1.0, 0.2, 0.1],
        'middle': [1.0, 0.6, 0.1],
        'ring':   [1.0, 0.8, 0.2],
        'pinky':  [0.9, 0.3, 0.5],
        'thumb':  [0.8, 0.1, 0.4],
        'joint':  [0.95, 0.85, 0.7],
    }


# ============================================================
# OBJ export
# ============================================================


def _write_skeleton_to_obj(f, keypoints_3d, skeleton, colors,
                           sphere_radius, bone_radius, vertex_offset, label=""):
    """
    Write one complete skeleton group (spheres + colored bones) into an
    already-open OBJ file.  Designed to be called once per skeleton
    (body, left hand, right hand) so that all groups share a single
    vertex-offset counter and produce a valid combined OBJ.

    Args:
        f: Open file handle positioned for writing
        keypoints_3d: (N, 3) or (N, 4) array of 3D keypoint positions for this group
        skeleton: {part_name: [(start_idx, end_idx), ...]} connectivity
        colors: {part_name: [r, g, b]} color map (must include 'joint')
        sphere_radius: Radius of joint spheres
        bone_radius: Radius of bone cylinders
        vertex_offset: Current cumulative vertex count (for 1-indexed OBJ faces)
        label: Human-readable label written as OBJ comments

    Returns:
        Updated vertex_offset after all geometry has been written.
    """
    if keypoints_3d.shape[1] == 4:
        keypoints_xyz = keypoints_3d[:, :3]
    else:
        keypoints_xyz = keypoints_3d

    joint_color = colors['joint']

    f.write("# {} keypoint spheres\n".format(label))
    for kp_idx, kp in enumerate(keypoints_xyz):
        if np.all(kp == 0.0):
            continue

        sphere_verts, sphere_faces = create_sphere(
            kp, sphere_radius, joint_color, subdivisions=1)

        if len(sphere_verts) > 0:
            for v in sphere_verts:
                f.write("v {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f}\n".format(
                    v[0], v[1], v[2], v[3], v[4], v[5]))
            for face in sphere_faces:
                f.write("f {} {} {}\n".format(
                    face[0] + 1 + vertex_offset,
                    face[1] + 1 + vertex_offset,
                    face[2] + 1 + vertex_offset))
            vertex_offset += len(sphere_verts)
    f.write("\n")

    f.write("# {} skeleton bones\n".format(label))
    for part_name, connections in skeleton.items():
        f.write("# {} - {}\n".format(label, part_name))
        part_color = colors[part_name]

        for start_idx, end_idx in connections:
            if start_idx < keypoints_xyz.shape[0] and end_idx < keypoints_xyz.shape[0]:
                start_pos = keypoints_xyz[start_idx]
                end_pos   = keypoints_xyz[end_idx]

                if np.all(start_pos == 0.0) or np.all(end_pos == 0.0):
                    continue

                cyl_verts, cyl_faces = create_cylinder(
                    start_pos, end_pos, bone_radius, part_color, subdivisions=6)

                if len(cyl_verts) > 0:
                    for v in cyl_verts:
                        f.write("v {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f}\n".format(
                            v[0], v[1], v[2], v[3], v[4], v[5]))
                    for face in cyl_faces:
                        f.write("f {} {} {}\n".format(
                            face[0] + 1 + vertex_offset,
                            face[1] + 1 + vertex_offset,
                            face[2] + 1 + vertex_offset))
                    vertex_offset += len(cyl_verts)
        f.write("\n")

    return vertex_offset


def save_keypoints_as_obj(keypoints_3d, output_path, skeleton=None,
                          sphere_radius=0.02, bone_radius=0.01,
                          left_hand_keypoints=None, right_hand_keypoints=None,
                          hand_sphere_radius=0.008, hand_bone_radius=0.004):
    """
    Save 3D keypoints as a single OBJ file containing the body skeleton
    and, optionally, left and right MANO hand skeletons.  All three
    skeleton groups share one vertex namespace so the file can be loaded
    as a single mesh in any viewer.

    Args:
        keypoints_3d: (N_body, 3) or (N_body, 4) array of body 3D keypoint positions
        output_path: Path to save OBJ file
        skeleton: Body skeleton definition dict.  If None, uses default COCO25 OpenPose skeleton.
        sphere_radius: Radius of body joint spheres
        bone_radius: Radius of body bone cylinders
        left_hand_keypoints: (21, 3) or (21, 4) array of left-hand 3D keypoints, or None
        right_hand_keypoints: (21, 3) or (21, 4) array of right-hand 3D keypoints, or None
        hand_sphere_radius: Radius of hand joint spheres
        hand_bone_radius: Radius of hand bone cylinders
    """
    if skeleton is None:
        skeleton = get_skeleton_definition()

    body_colors = get_color_scheme()

    with open(output_path, 'w') as f:
        f.write("# 3D Skeleton Keypoints with Vertex Colors (COCO25 OpenPose 25-keypoint format)\n")
        f.write("# Number of body keypoints: {}\n".format(keypoints_3d.shape[0]))
        if left_hand_keypoints is not None:
            f.write("# Number of left hand keypoints (MANO): {}\n".format(
                left_hand_keypoints.shape[0]))
        if right_hand_keypoints is not None:
            f.write("# Number of right hand keypoints (MANO): {}\n".format(
                right_hand_keypoints.shape[0]))
        f.write("# Format: v x y z r g b\n\n")

        vertex_offset = 0

        vertex_offset = _write_skeleton_to_obj(
            f, keypoints_3d, skeleton, body_colors,
            sphere_radius, bone_radius, vertex_offset, label="Body")

        if left_hand_keypoints is not None:
            mano_skeleton = get_mano_skeleton_definition()
            lhand_colors  = get_left_hand_color_scheme()
            vertex_offset = _write_skeleton_to_obj(
                f, left_hand_keypoints, mano_skeleton, lhand_colors,
                hand_sphere_radius, hand_bone_radius, vertex_offset,
                label="Left Hand")

        if right_hand_keypoints is not None:
            mano_skeleton = get_mano_skeleton_definition()
            rhand_colors  = get_right_hand_color_scheme()
            vertex_offset = _write_skeleton_to_obj(
                f, right_hand_keypoints, mano_skeleton, rhand_colors,
                hand_sphere_radius, hand_bone_radius, vertex_offset,
                label="Right Hand")

    vprint("Saved colored skeleton OBJ to: {}".format(output_path))


# ============================================================
# Camera helpers
# ============================================================


def get_camera_params_from_render(elevation, azimuth, distance, focal_length, principal_point,
                                   image_height, image_width, device='cpu'):
    """
    Reconstruct camera parameters from rendering parameters.

    The returned dict includes the precomputed rotation matrix 'R' alongside
    the Rodrigues vector 'r', so the hot projection path avoids repeated
    cv2.Rodrigues calls.

    Args:
        elevation: Elevation angle in degrees (zero-centered: -90 to 90)
        azimuth: Azimuth angle in degrees (0 to 360)
        distance: Camera distance from origin
        focal_length: [fx, fy] focal length in pixels
        principal_point: [cx, cy] principal point in pixels
        image_height, image_width: Image dimensions
        device: torch device

    Returns:
        Dictionary with camera parameters in OpenCV format (plus 'R' matrix)
    """
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

    R_np = R[0].cpu().numpy()
    T_np = T[0].cpu().numpy()

    R_flip = np.array([[-1.0, 0.0, 0.0],
                       [ 0.0, -1.0, 0.0],
                       [ 0.0,  0.0, 1.0]])
    R_np = R_np @ R_flip
    R_np = np.linalg.inv(R_np)

    r_vec, _ = cv2.Rodrigues(R_np)

    mtx = np.array([
        [focal_length[0], 0, principal_point[0]],
        [0, focal_length[1], principal_point[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    return {
        'r':    r_vec,
        'R':    R_np,   # Cached full rotation matrix — avoids repeated Rodrigues calls
        't':    T_np.reshape(3, 1),
        'mtx':  mtx,
        'dist': np.zeros(5, dtype=np.float32)
    }


def _build_cam_cache(view_keypoints, distance, focal_length, principal_point,
                     img_height, img_width, device):
    """
    Pre-compute one camera dict per unique (elevation, azimuth) view, shared
    across all keypoints (avoids per-keypoint look_at_view_transform +
    cv2.Rodrigues calls).

    Args:
        view_keypoints: {view_id: {'keypoints': ..., 'elevation': float, 'azimuth': float}}
        (remaining args match get_camera_params_from_render)

    Returns:
        {view_id: camera_params_dict}
    """
    cam_cache = {}
    for view_id, data in view_keypoints.items():
        cam_cache[view_id] = get_camera_params_from_render(
            elevation=data['elevation'],
            azimuth=data['azimuth'],
            distance=distance,
            focal_length=focal_length,
            principal_point=principal_point,
            image_height=img_height,
            image_width=img_width,
            device=device
        )
    return cam_cache


# ============================================================
# Per-keypoint worker (module-level for pickling)
# ============================================================


def _triangulate_one_keypoint_task(args):
    """
    Picklable module-level worker for ProcessPoolExecutor: triangulates a
    single keypoint from multi-view observations.

    Args:
        args: tuple of (kp_idx, kp_data_per_view, cam_cache,
                        confidence_threshold, use_ransac, ransac_reproj_threshold,
                        min_views)
              where kp_data_per_view is {view_id: kp_array_for_this_keypoint}

    Returns:
        dict with keys: kp_idx, point (shape (4,)), cost (float or None), stat (dict)
    """
    (kp_idx, kp_data_per_view, cam_cache,
     confidence_threshold, use_ransac, ransac_reproj_threshold,
     min_views) = args

    imgpoints   = {}
    cams        = {}
    confidences = {}

    for view_id, kp in kp_data_per_view.items():
        if kp[2] > confidence_threshold:
            imgpoints[view_id]   = kp[:2]
            confidences[view_id] = float(kp[2])
            cams[view_id]        = cam_cache[view_id]

    null_stat = {
        'keypoint_idx':   kp_idx,
        'n_observations': len(imgpoints),
        'n_inliers':      0,
        'inlier_views':   [],
        'cost':           float('inf'),
        'mean_confidence': 0.0,
    }

    if len(imgpoints) < min_views:
        return {'kp_idx': kp_idx, 'point': np.zeros(4), 'cost': None, 'stat': null_stat}

    if use_ransac:
        point_3d, inlier_views, cost = triangulate_with_ransac(
            imgpoints, cams,
            reproj_threshold=ransac_reproj_threshold,
            min_views=min_views
        )

        if point_3d is None:
            return {'kp_idx': kp_idx, 'point': np.zeros(4), 'cost': None, 'stat': null_stat}

        inlier_confs   = [confidences[v] for v in inlier_views if v in confidences]
        mean_confidence = np.mean(inlier_confs) if inlier_confs else 0.0

        stat = {
            'keypoint_idx':   kp_idx,
            'n_observations': len(imgpoints),
            'n_inliers':      len(inlier_views),
            'inlier_views':   inlier_views,
            'cost':           cost,
            'mean_confidence': mean_confidence,
        }
        point = np.array([point_3d[0], point_3d[1], point_3d[2], mean_confidence])
        return {'kp_idx': kp_idx, 'point': point, 'cost': cost, 'stat': stat}

    else:
        result = least_squares(
            triangulation_residuals,
            np.zeros(3),
            args=(imgpoints, cams),
            method='lm'
        )

        all_confs      = [confidences[v] for v in imgpoints if v in confidences]
        mean_confidence = np.mean(all_confs) if all_confs else 0.0

        stat = {
            'keypoint_idx':   kp_idx,
            'n_observations': len(imgpoints),
            'n_inliers':      len(imgpoints),
            'inlier_views':   list(imgpoints.keys()),
            'cost':           result.cost,
            'mean_confidence': mean_confidence,
        }
        point = np.array([result.x[0], result.x[1], result.x[2], mean_confidence])
        return {'kp_idx': kp_idx, 'point': point, 'cost': result.cost, 'stat': stat}


# ============================================================
# Shared triangulation loop
# ============================================================

# Number of worker processes for keypoint-level parallelism.
# Defaults to all logical CPUs; reduce if memory is tight.
_KP_WORKERS = max(1, multiprocessing.cpu_count())


def _triangulate_keypoint_set(view_keypoints, distance, focal_length, principal_point,
                               img_height, img_width, device,
                               min_views=2, confidence_threshold=0.3,
                               use_ransac=True, ransac_reproj_threshold=5.0,
                               label="keypoints"):
    """
    Triangulate one set of 2D keypoints (body *or* one hand) observed across
    multiple rendered views.  This is the shared inner loop called separately
    for body, left hand, and right hand.

    Args:
        view_keypoints: {view_id: {'keypoints': (N,3), 'elevation': float, 'azimuth': float}}
                        The (N,3) array columns are [x, y, confidence].
        distance: Camera distance from origin used during rendering
        focal_length: [fx, fy] in pixels
        principal_point: [cx, cy] in pixels
        img_height, img_width: Rendered image dimensions
        device: torch device
        min_views: Minimum number of confident observations required
        confidence_threshold: Minimum detection confidence to include a view
        use_ransac: Whether to use RANSAC for robust triangulation
        ransac_reproj_threshold: Reprojection error threshold for RANSAC (pixels)
        label: Human-readable label used in progress bars and log messages

    Returns:
        keypoints_3d: (N, 4) array of triangulated 3D points with confidence
        triangulation_costs: list of optimisation costs (one per successful keypoint)
        keypoint_stats: list of per-keypoint stat dicts
    """
    if not view_keypoints:
        return None, [], []

    n_keypoints = None
    for view_id, data in view_keypoints.items():
        n_kp = data['keypoints'].shape[0]
        if n_keypoints is None:
            n_keypoints = n_kp
        elif n_kp != n_keypoints:
            pass  # view keypoint-count mismatch (tolerated)

    if n_keypoints is None or n_keypoints == 0:
        return None, [], []

    # ------------------------------------------------------------------
    # Build camera cache once, reused by all keypoints
    # ------------------------------------------------------------------
    cam_cache = _build_cam_cache(
        view_keypoints, distance, focal_length, principal_point,
        img_height, img_width, device
    )

    # Build per-keypoint task args — each task only carries its own slice
    # of observations so the full keypoints array doesn't need to be pickled
    # repeatedly for every worker.
    tasks = []
    for kp_idx in range(n_keypoints):
        kp_data_per_view = {
            view_id: data['keypoints'][kp_idx]
            for view_id, data in view_keypoints.items()
        }
        tasks.append((
            kp_idx, kp_data_per_view, cam_cache,
            confidence_threshold, use_ransac, ransac_reproj_threshold,
            min_views
        ))

    keypoints_3d        = np.zeros((n_keypoints, 4))
    triangulation_costs = []
    keypoint_stats      = [None] * n_keypoints
    successful          = 0

    # ------------------------------------------------------------------
    # Parallel keypoint triangulation
    # ------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=_KP_WORKERS) as executor:
        future_to_idx = {executor.submit(_triangulate_one_keypoint_task, t): t[0]
                         for t in tasks}

        for future in tqdm(as_completed(future_to_idx),
                           total=n_keypoints,
                           desc=f"Triangulating {label}"):
            res    = future.result()
            kp_idx = res['kp_idx']
            keypoints_3d[kp_idx]   = res['point']
            keypoint_stats[kp_idx] = res['stat']
            if res['cost'] is not None:
                triangulation_costs.append(res['cost'])
                successful += 1

    _cost = f", mean cost {np.mean(triangulation_costs):.3f}" if triangulation_costs else ""
    vprint(f"[{label}] triangulated {successful}/{n_keypoints}{_cost}")

    return keypoints_3d, triangulation_costs, keypoint_stats


# ============================================================
# Main multi-view triangulation entry point
# ============================================================


# COCO25 OpenPose body-keypoint indices for the wrists
COCO_LEFT_WRIST  = 7   # L-Wrist in COCO25 format
COCO_RIGHT_WRIST = 4   # R-Wrist in COCO25 format


def _select_best_hand_candidate(raw, reference_point=None):
    """
    Pick the single best hand detection out of N candidates, reducing
    (N, 21, 2) to (21, 2):

    1. Proximity (primary): if reference_point is given, choose the candidate
       whose centroid is closest to it.
    2. Largest 2D bounding-box area (fallback) when the body wrist is missing
       or low-confidence.

    Args:
        raw: (N, 21, 2) array of hand candidates, or (21, 2) if only one.
        reference_point: (2,) [x, y] expected hand location (typically the
            body wrist), or None.

    Returns:
        (21, 2) array — the selected candidate.
    """
    if raw.ndim == 2:
        return raw
    if raw.shape[0] == 1:
        return raw[0]

    n_candidates = raw.shape[0]

    if reference_point is not None:
        centroids = raw.mean(axis=1)
        distances = np.linalg.norm(centroids - reference_point, axis=1)
        best_idx  = int(np.argmin(distances))
        vprint(f"  [_select_best_hand_candidate] {n_candidates} candidates → "
              f"selected idx {best_idx} (dist {distances[best_idx]:.1f} px to body wrist)")
    else:
        mins     = raw.min(axis=1)
        maxs     = raw.max(axis=1)
        extents  = maxs - mins
        areas    = extents[:, 0] * extents[:, 1]
        best_idx = int(np.argmax(areas))
        vprint(f"  [_select_best_hand_candidate] {n_candidates} candidates → "
              f"selected idx {best_idx} (area {areas[best_idx]:.0f} px² — no body wrist available)")

    return raw[best_idx]


def _normalize_hand_keypoints(raw):
    """
    Normalise a hand keypoint array so it matches the (N_kp, 3) layout
    expected by the triangulation loop, where columns are [x, y, confidence].

    Handles any number of leading batch / detection dimensions by repeatedly
    taking index [0] until the array is 2-D.

    Supported on-disk formats:
        • (1, 21, 2)  - single detection, xy only   → (21, 3)  conf=1
        • (N, 21, 2)  - N detections, xy only       → (21, 3)  takes first, conf=1
        • (21, 2)     - no batch dim, xy only        → (21, 3)  conf=1
        • (1, 21, 3)  - single detection, with conf  → (21, 3)
        • (21, 3)     - already correct              → pass through

    Args:
        raw: numpy array as loaded directly from the .npy file

    Returns:
        Array of shape (N_kp, 3) with columns [x, y, confidence].
    """
    raw_shape = raw.shape
    kp = np.array(raw)

    while kp.ndim > 2:
        kp = kp[0]

    if kp.ndim == 1:
        kp = kp.reshape(1, -1)

    if len(raw_shape) > 2:
        vprint(f"  [_normalize_hand_keypoints] collapsed {raw_shape} → {kp.shape}")

    if kp.shape[-1] == 2:
        ones = np.ones((kp.shape[0], 1), dtype=kp.dtype)
        kp   = np.concatenate([kp, ones], axis=-1)

    return kp


def triangulate_keypoints_multiview(data_root, folder_name, rotation_angles,
                                     min_views=2, confidence_threshold=0.3,
                                     resolution_multiplier=1.0,
                                     use_ransac=True, ransac_reproj_threshold=5.0):
    """
    Triangulate 2D body and hand keypoints from multiple rendered views
    to recover 3D poses.

    Args:
        data_root: Root directory for data
        folder_name: Name of the specific sequence folder
        rotation_angles: List of rotation angles used in rendering
        min_views: Minimum number of views required for triangulation
        confidence_threshold: Minimum keypoint confidence score
        resolution_multiplier: Resolution scaling factor used in rendering
        use_ransac: Whether to use RANSAC for robust triangulation
        ransac_reproj_threshold: Reprojection error threshold for RANSAC inliers (pixels)

    Returns:
        Dictionary with keys 'body', 'left_hand', 'right_hand', each holding
        (keypoints_3d, costs, stats) or None.
    """
    device = torch.device('cpu')

    seq_path      = os.path.join(data_root, folder_name)
    metadata_path = os.path.join(seq_path, 'metadata.npz')
    metadata = dict(np.load(metadata_path)) if os.path.exists(metadata_path) else None

    # Discover the input image (skip mask / viz files).
    _skip = ('human', 'object', 'mask', 'viz', 'seg', 'depth', 'normal')
    _imgs = sorted(
        f for f in os.listdir(seq_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        and not any(s in f.lower() for s in _skip)
    )
    if not _imgs:
        print(f"No input image found in {folder_name}")
        return None
    img_path = os.path.join(seq_path, _imgs[0])

    full_img = cv2.imread(img_path)

    pad_h              = abs((full_img.shape[0] // 2) * 2 - full_img.shape[0])
    pad_w              = abs((full_img.shape[1] // 2) * 2 - full_img.shape[1])
    img_height_padded  = full_img.shape[0] + pad_h
    img_width_padded   = full_img.shape[1] + pad_w

    img_height = int(img_height_padded * resolution_multiplier)
    img_width  = int(img_width_padded  * resolution_multiplier)

    if metadata is not None:
        focal_length    = metadata["focal_length"]    * resolution_multiplier
        principal_point = metadata["principal_point"] * resolution_multiplier
    else:
        # In-the-wild fallback: standard focal heuristic, principal at image center.
        _h, _w = full_img.shape[0], full_img.shape[1]
        _f = 0.5 * (_h + _w)
        focal_length    = np.array([_f, _f], dtype=np.float32) * resolution_multiplier
        principal_point = np.array([_w / 2.0, _h / 2.0], dtype=np.float32) * resolution_multiplier
        print(f"[triangulate] No metadata.npz — default focal={_f:.1f}, principal=center")

    h3d_obj_path = None
    for _ext in ('.obj', '.glb'):
        _cand = os.path.join(seq_path, f'full_img_textured{_ext}')
        if os.path.exists(_cand):
            h3d_obj_path = _cand
            break
    if h3d_obj_path is None:
        print(f"No full_img_textured.obj/.glb found in {folder_name}")
        return None
    h3d_obj_mesh = trimesh.load(h3d_obj_path, process=False, force='mesh')
    h3d_verts    = h3d_obj_mesh.vertices

    centered_bounds_min = h3d_verts.min(axis=0)
    centered_bounds_max = h3d_verts.max(axis=0)
    max_extent = max(
        np.linalg.norm(centered_bounds_min),
        np.linalg.norm(centered_bounds_max)
    )

    safety_margin   = 1.0
    distance_width  = (max_extent * focal_length[0] * safety_margin) / (img_width  / 2.0)
    distance_height = (max_extent * focal_length[1] * safety_margin) / (img_height / 2.0)
    distance        = max(distance_width, distance_height)

    view_keypoints_body  = {}
    view_keypoints_lhand = {}
    view_keypoints_rhand = {}

    resolution_suffix  = "_highres" if resolution_multiplier > 1.0 else ""
    keypoints_dir      = os.path.join(data_root, folder_name, 'keypoints')
    keypoints_hand_dir = os.path.join(data_root, folder_name, 'kp2d_hand')

    vprint(f"Loading body keypoints (COCO25 OpenPose) from:  {keypoints_dir}")
    vprint(f"Loading hand keypoints from:  {keypoints_hand_dir}")

    body_view_count  = 0
    lhand_view_count = 0
    rhand_view_count = 0

    for elevation in rotation_angles:
        elevation_centered = elevation - 180
        if elevation_centered <= -90 or elevation_centered >= 90:
            continue

        for azimuth in rotation_angles:
            view_id = f"e{elevation_centered}_a{azimuth}"

            # ---- Body keypoints (COCO25 OpenPose) ---------------------------
            kp_file = os.path.join(
                keypoints_dir,
                f'rend_img_obj_{view_id}{resolution_suffix}.npy')
            if not os.path.exists(kp_file):
                kp_file = os.path.join(keypoints_dir, f'rend_img_obj_{view_id}.npy')

            if os.path.exists(kp_file):
                kp2d = np.load(kp_file)
                if kp2d.shape[0] > 0:
                    view_keypoints_body[view_id] = {
                        'keypoints': kp2d,
                        'elevation': elevation_centered,
                        'azimuth':   azimuth
                    }
                    body_view_count += 1

            # ---- Left hand keypoints ----------------------------------------
            kp_lhand_file = os.path.join(
                keypoints_hand_dir,
                f'rend_img_obj_{view_id}{resolution_suffix}_left.npy')
            if not os.path.exists(kp_lhand_file):
                kp_lhand_file = os.path.join(
                    keypoints_hand_dir, f'rend_img_obj_{view_id}_left.npy')

            if os.path.exists(kp_lhand_file):
                raw_lhand = np.load(kp_lhand_file)
                if raw_lhand.ndim == 3 and raw_lhand.shape[0] > 1:
                    ref_pt = None
                    if view_id in view_keypoints_body:
                        body_kps = view_keypoints_body[view_id]['keypoints']
                        if (body_kps.shape[0] > COCO_LEFT_WRIST
                                and body_kps[COCO_LEFT_WRIST, 2] > confidence_threshold):
                            ref_pt = body_kps[COCO_LEFT_WRIST, :2]
                    raw_lhand = _select_best_hand_candidate(raw_lhand, ref_pt)
                kp2d_lhand = _normalize_hand_keypoints(raw_lhand)
                if kp2d_lhand.shape[0] > 0:
                    view_keypoints_lhand[view_id] = {
                        'keypoints': kp2d_lhand,
                        'elevation': elevation_centered,
                        'azimuth':   azimuth
                    }
                    lhand_view_count += 1

            # ---- Right hand keypoints ---------------------------------------
            kp_rhand_file = os.path.join(
                keypoints_hand_dir,
                f'rend_img_obj_{view_id}{resolution_suffix}_right.npy')
            if not os.path.exists(kp_rhand_file):
                kp_rhand_file = os.path.join(
                    keypoints_hand_dir, f'rend_img_obj_{view_id}_right.npy')

            if os.path.exists(kp_rhand_file):
                raw_rhand = np.load(kp_rhand_file)
                if raw_rhand.ndim == 3 and raw_rhand.shape[0] > 1:
                    ref_pt = None
                    if view_id in view_keypoints_body:
                        body_kps = view_keypoints_body[view_id]['keypoints']
                        if (body_kps.shape[0] > COCO_RIGHT_WRIST
                                and body_kps[COCO_RIGHT_WRIST, 2] > confidence_threshold):
                            ref_pt = body_kps[COCO_RIGHT_WRIST, :2]
                    raw_rhand = _select_best_hand_candidate(raw_rhand, ref_pt)
                kp2d_rhand = _normalize_hand_keypoints(raw_rhand)
                if kp2d_rhand.shape[0] > 0:
                    view_keypoints_rhand[view_id] = {
                        'keypoints': kp2d_rhand,
                        'elevation': elevation_centered,
                        'azimuth':   azimuth
                    }
                    rhand_view_count += 1

    vprint(f"Loaded body keypoints (COCO25 OpenPose) from {body_view_count} views")
    vprint(f"Loaded left  hand keypoints from {lhand_view_count} views")
    vprint(f"Loaded right hand keypoints from {rhand_view_count} views")

    if body_view_count < min_views:
        print(f"Insufficient body views ({body_view_count} < {min_views}), cannot triangulate")
        return None

    body_result = _triangulate_keypoint_set(
        view_keypoints_body, distance, focal_length, principal_point,
        img_height, img_width, device,
        min_views=min_views,
        confidence_threshold=confidence_threshold,
        use_ransac=use_ransac,
        ransac_reproj_threshold=ransac_reproj_threshold,
        label="Body (COCO25 OpenPose)"
    )

    lhand_result = None
    if lhand_view_count >= min_views:
        lhand_result = _triangulate_keypoint_set(
            view_keypoints_lhand, distance, focal_length, principal_point,
            img_height, img_width, device,
            min_views=min_views,
            confidence_threshold=confidence_threshold,
            use_ransac=use_ransac,
            ransac_reproj_threshold=ransac_reproj_threshold,
            label="Left Hand"
        )
    else:
        vprint(f"Skipping left hand: only {lhand_view_count} views available (need {min_views})")

    rhand_result = None
    if rhand_view_count >= min_views:
        rhand_result = _triangulate_keypoint_set(
            view_keypoints_rhand, distance, focal_length, principal_point,
            img_height, img_width, device,
            min_views=min_views,
            confidence_threshold=confidence_threshold,
            use_ransac=use_ransac,
            ransac_reproj_threshold=ransac_reproj_threshold,
            label="Right Hand"
        )
    else:
        vprint(f"Skipping right hand: only {rhand_view_count} views available (need {min_views})")

    return {
        'body':       body_result,
        'left_hand':  lhand_result,
        'right_hand': rhand_result
    }


# ============================================================
# Persistence helpers
# ============================================================


def _has_valid_keypoints(kp3d):
    """Return True if at least one keypoint is non-zero (i.e. was triangulated)."""
    if kp3d is None:
        return False
    return not np.all(kp3d[:, :3] == 0.0)


def _save_stats(output_dir, filename, kp3d, costs, stats, use_ransac, ransac_reproj_threshold):
    """Persist triangulation statistics for one keypoint set to a .npz file."""
    stats_file = os.path.join(output_dir, filename)

    if stats:
        stats_arrays = {
            'keypoint_idx':       np.array([s['keypoint_idx']    for s in stats]),
            'n_observations':     np.array([s['n_observations']  for s in stats]),
            'n_inliers':          np.array([s['n_inliers']       for s in stats]),
            'per_keypoint_costs': np.array([s['cost']            for s in stats]),
            'mean_confidences':   np.array([s['mean_confidence'] for s in stats]),
        }
    else:
        stats_arrays = {}

    np.savez(
        stats_file,
        keypoints_3d=kp3d,
        costs=np.array(costs),
        mean_cost=np.mean(costs)    if costs else 0,
        median_cost=np.median(costs) if costs else 0,
        n_successful=len(costs),
        n_total=kp3d.shape[0],
        use_ransac=use_ransac,
        ransac_reproj_threshold=ransac_reproj_threshold if use_ransac else 0,
        **stats_arrays
    )
    vprint(f"Saved triangulation statistics to: {stats_file}")


# ============================================================
# Sequence worker (module-level for pickling)
# ============================================================


def _process_one_sequence(args):
    """
    Module-level worker so sequences can be processed in
    parallel via ProcessPoolExecutor in process_all_sequences().

    Each sequence is fully independent (separate files, no shared state), so
    process-level parallelism is safe and efficient.

    Note: keypoint-level parallelism (_KP_WORKERS) is already active inside
    each sequence.  When running many sequences in parallel, consider setting
    _KP_WORKERS = 1 (or a small value) to avoid spawning too many processes:
        total_processes ≈ _SEQ_WORKERS x _KP_WORKERS
    """
    (folder_name, data_root, rotation_angles, output_suffix,
     min_views, confidence_threshold, resolution_multiplier,
     use_ransac, ransac_reproj_threshold,
     save_obj, skeleton,
     sphere_radius, bone_radius,
     hand_sphere_radius, hand_bone_radius) = args

    vprint(f"\n{'='*60}\nProcessing: {folder_name}\n{'='*60}")

    results = triangulate_keypoints_multiview(
        data_root=data_root,
        folder_name=folder_name,
        rotation_angles=rotation_angles,
        min_views=min_views,
        confidence_threshold=confidence_threshold,
        resolution_multiplier=resolution_multiplier,
        use_ransac=use_ransac,
        ransac_reproj_threshold=ransac_reproj_threshold
    )

    if results is None:
        print(f"Failed to triangulate keypoints for {folder_name}")
        return

    output_dir = os.path.join(data_root, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    resolution_suffix_str = "_highres" if resolution_multiplier > 1.0 else ""
    ransac_suffix         = "_ransac"  if use_ransac else ""
    base_suffix           = f"{output_suffix}{resolution_suffix_str}{ransac_suffix}"

    body_kp3d = None
    if results['body'] is not None:
        body_kp3d, body_costs, body_stats = results['body']

        np.save(os.path.join(output_dir, f'keypoints{base_suffix}.npy'), body_kp3d)
        vprint(f"Saved body 3D keypoints (COCO25 OpenPose) with confidence to: "
              f"keypoints{base_suffix}.npy")
        vprint(f"  Shape: {body_kp3d.shape} (columns: x, y, z, mean_confidence)")

        _save_stats(output_dir, f'triangulation_stats{base_suffix}.npz',
                    body_kp3d, body_costs, body_stats, use_ransac, ransac_reproj_threshold)
    else:
        print("Body triangulation returned None - skipping body outputs.")

    lhand_kp3d = None
    if results['left_hand'] is not None:
        lhand_kp3d, lhand_costs, lhand_stats = results['left_hand']

        np.save(os.path.join(output_dir, f'keypoints{base_suffix}_lhand.npy'), lhand_kp3d)
        vprint(f"Saved left hand 3D keypoints with confidence to: "
              f"keypoints{base_suffix}_lhand.npy")
        vprint(f"  Shape: {lhand_kp3d.shape} (columns: x, y, z, mean_confidence)")

        _save_stats(output_dir, f'triangulation_stats{base_suffix}_lhand.npz',
                    lhand_kp3d, lhand_costs, lhand_stats, use_ransac, ransac_reproj_threshold)

    rhand_kp3d = None
    if results['right_hand'] is not None:
        rhand_kp3d, rhand_costs, rhand_stats = results['right_hand']

        np.save(os.path.join(output_dir, f'keypoints{base_suffix}_rhand.npy'), rhand_kp3d)
        vprint(f"Saved right hand 3D keypoints with confidence to: "
              f"keypoints{base_suffix}_rhand.npy")
        vprint(f"  Shape: {rhand_kp3d.shape} (columns: x, y, z, mean_confidence)")

        _save_stats(output_dir, f'triangulation_stats{base_suffix}_rhand.npz',
                    rhand_kp3d, rhand_costs, rhand_stats, use_ransac, ransac_reproj_threshold)

    if save_obj and body_kp3d is not None:
        lhand_for_obj = lhand_kp3d if _has_valid_keypoints(lhand_kp3d) else None
        rhand_for_obj = rhand_kp3d if _has_valid_keypoints(rhand_kp3d) else None

        obj_file = os.path.join(output_dir, f'keypoints{base_suffix}.obj')
        save_keypoints_as_obj(
            body_kp3d, obj_file,
            skeleton=skeleton,
            sphere_radius=sphere_radius,
            bone_radius=bone_radius,
            left_hand_keypoints=lhand_for_obj,
            right_hand_keypoints=rhand_for_obj,
            hand_sphere_radius=hand_sphere_radius,
            hand_bone_radius=hand_bone_radius
        )


# ============================================================
# Batch processing
# ============================================================


# Number of sequences to process in parallel.
# Total spawned processes ≈ _SEQ_WORKERS x _KP_WORKERS — tune accordingly.
_SEQ_WORKERS = max(1, multiprocessing.cpu_count() // max(1, _KP_WORKERS))


def process_all_sequences(data_root, rotation_angles, output_suffix='_3d',
                          min_views=2, confidence_threshold=0.3,
                          resolution_multiplier=1.0,
                          use_ransac=True, ransac_reproj_threshold=5.0,
                          save_obj=True, skeleton=None,
                          sphere_radius=0.02, bone_radius=0.01,
                          hand_sphere_radius=0.008, hand_bone_radius=0.004,
                          sequence_folders=None):
    """
    Process all sequences in data_root: triangulate body (COCO25 OpenPose) and hand
    keypoints, persist the results, and (optionally) export a combined OBJ.

    Sequences are processed in parallel via ProcessPoolExecutor.
    Worker counts are printed at startup; tune via the _SEQ_WORKERS /
    _KP_WORKERS module globals (run() / --kp_workers sets _KP_WORKERS).

    Output files per sequence (inside ``<seq>/``):
        * ``keypoints_3d[_highres][_ransac].npy``
        * ``keypoints_3d[…]_lhand.npy``
        * ``keypoints_3d[…]_rhand.npy``
        * ``triangulation_stats_3d[…].npz``  (+ _lhand / _rhand variants)
        * ``keypoints_3d[…].obj``

    Args:
        data_root: Root directory containing sequence folders
        rotation_angles: List of rotation angles used in rendering
        output_suffix: Suffix for output filename
        min_views: Minimum number of views required for triangulation
        confidence_threshold: Minimum keypoint confidence score
        resolution_multiplier: Resolution scaling factor
        use_ransac: Whether to use RANSAC for robust triangulation
        ransac_reproj_threshold: Reprojection error threshold for RANSAC (pixels)
        save_obj: Whether to save combined OBJ files
        skeleton: Body skeleton connectivity definition (dict). None → default COCO25 OpenPose.
        sphere_radius: Radius of body joint spheres in the OBJ
        bone_radius: Radius of body bone cylinders in the OBJ
        hand_sphere_radius: Radius of hand joint spheres in the OBJ
        hand_bone_radius: Radius of hand bone cylinders in the OBJ
        sequence_folders: List of specific folders to process (None = discover all)
    """
    if sequence_folders is None:
        sequence_folders = []
        for item in os.listdir(data_root):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.npz')
                keypoints_dir = os.path.join(item_path, 'keypoints')
                if os.path.exists(metadata_path) and os.path.exists(keypoints_dir):
                    sequence_folders.append(item)

    sequence_folders = sorted(sequence_folders)
    vprint(f"Found {len(sequence_folders)} sequences to process")
    vprint(f"Sequence-level parallelism: {_SEQ_WORKERS} workers  |  "
          f"Keypoint-level parallelism: {_KP_WORKERS} workers/sequence")

    task_args = [
        (folder_name, data_root, rotation_angles, output_suffix,
         min_views, confidence_threshold, resolution_multiplier,
         use_ransac, ransac_reproj_threshold,
         save_obj, skeleton,
         sphere_radius, bone_radius,
         hand_sphere_radius, hand_bone_radius)
        for folder_name in sequence_folders
    ]

    # ------------------------------------------------------------------
    # Parallel sequence processing
    # ------------------------------------------------------------------
    if _SEQ_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=_SEQ_WORKERS) as executor:
            futures = {executor.submit(_process_one_sequence, a): a[0]
                       for a in task_args}
            for future in tqdm(as_completed(futures),
                               total=len(sequence_folders),
                               desc="Processing sequences"):
                folder_name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[ERROR] {folder_name}: {exc}")
    else:
        for args in tqdm(task_args, desc="Processing sequences"):
            try:
                _process_one_sequence(args)
            except Exception as exc:
                print(f"[ERROR] {args[0]}: {exc}")


# ============================================================
# Pipeline step interface
# ============================================================


def _output_path(seq_dir: str) -> str:
    return os.path.join(seq_dir, "keypoints_3d.npy")


def run(
    seq_dir: str,
    rotation_angles: list = None,
    min_views: int = 3,
    confidence_threshold: float = 0.6,
    resolution_multiplier: float = 1.0,
    use_ransac: bool = True,
    ransac_reproj_threshold: float = 5.0,
    kp_workers: int = None,
    verbose: bool = False,
) -> bool:
    """
    Triangulate 3D keypoints for one sequence.

    Args:
        seq_dir: Path to sequence directory.
        rotation_angles: Angles used during rendering (default: 0..330 step 30).
        min_views: Minimum views per keypoint for triangulation.
        confidence_threshold: Minimum 2D keypoint confidence.
        resolution_multiplier: Must match rendering resolution.
        use_ransac: Use RANSAC for robust triangulation.
        ransac_reproj_threshold: RANSAC inlier threshold in pixels.
        kp_workers: Number of keypoint-level parallel workers (None = cpu_count).

    Returns:
        True if the step ran, False if skipped.
    """
    set_verbose(verbose)
    global _KP_WORKERS, _SEQ_WORKERS

    seq_dir = os.path.abspath(seq_dir)
    out_path = _output_path(seq_dir)

    kp_dir = os.path.join(seq_dir, "keypoints")
    if not os.path.isdir(kp_dir):
        print(f"[triangulate] Skipping — keypoints dir not found: {kp_dir}")
        return False

    if rotation_angles is None:
        rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    if kp_workers is None:
        kp_workers = max(1, multiprocessing.cpu_count())

    # Single-sequence step: parallelize at keypoint level only.
    _KP_WORKERS = kp_workers
    _SEQ_WORKERS = 1

    folder_name = os.path.basename(seq_dir)
    data_root = os.path.dirname(seq_dir)

    print(f"[triangulate] Running on {seq_dir}")
    multiprocessing.freeze_support()
    process_all_sequences(
        data_root=data_root,
        rotation_angles=rotation_angles,
        output_suffix="_3d",
        min_views=min_views,
        confidence_threshold=confidence_threshold,
        resolution_multiplier=resolution_multiplier,
        use_ransac=use_ransac,
        ransac_reproj_threshold=ransac_reproj_threshold,
        save_obj=True,
        skeleton=None,
        sequence_folders=[folder_name],
    )

    print(f"[triangulate] Done → {out_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triangulate 3D keypoints for one sequence"
    )
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--resolution_multiplier", type=float, default=1.0)
    parser.add_argument("--no_ransac", action="store_true")
    parser.add_argument("--kp_workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose per-view / per-iteration logging.")
    args = parser.parse_args()
    run(
        seq_dir=args.seq_dir,
        min_views=args.min_views,
        confidence_threshold=args.confidence_threshold,
        resolution_multiplier=args.resolution_multiplier,
        use_ransac=not args.no_ransac,
        kp_workers=args.kp_workers,
        verbose=args.verbose,
    )

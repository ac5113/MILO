import os
import imageio
import numpy as np
import torch
import trimesh

from body_model import run_smpl, run_obj
from geometry import camera as cam_util
from geometry.mesh import make_batch_mesh
from geometry.plane import parse_floor_plane, get_plane_transform

from util.tensor import detach_all, to_torch, move_to

from .fig_specs import get_seq_figure_skip, get_seq_static_lookat_points
from .tools import smpl_to_geometry


def prep_result_vis(res, vis_mask, track_ids, body_model):
    """
    :param res (dict) with (B, T, *) tensor elements, B tracks and T frames
    :param vis_mask (B, T) with visibility of each track in each frame
    :param track_ids (B,) index of each track
    """
    print("RESULT FIELDS", res.keys())
    res = detach_all(res)
    with torch.no_grad():
        world_smpl = run_smpl(
            body_model,
            res["trans"],
            res["root_orient"],
            res["pose_body"],
            res.get("betas", None),
            res.get("hum_scale", None),
            res.get("hand_pose", None),
        )
        if "obj_trans" in res:
            world_obj = run_obj(
                res["obj_trans"],
                res["obj_scale"],
                res["obj_verts"],
                res["obj_faces"],
                obj_rot6d=res.get("obj_rot6d", None),
            )
    T_w2c = None
    floor_plane = None
    if "cam_R" in res and "cam_t" in res:
        T_w2c = cam_util.make_4x4_pose(res["cam_R"][0], res["cam_t"][0])
    if "floor_plane" in res:
        floor_plane = res["floor_plane"][0]
    if "obj_trans" in res:
        return build_scene_dict(
            [world_smpl, world_obj],
            vis_mask,
            track_ids,
            T_w2c=T_w2c,
            floor_plane=floor_plane,
        )
    return build_scene_dict(
            [world_smpl],
            vis_mask,
            track_ids,
            T_w2c=T_w2c,
            floor_plane=floor_plane,
        )


def build_scene_dict(
    meshes, vis_mask, track_ids, T_w2c=None, floor_plane=None, **kwargs
):
    scene_dict = {}

    # first get the geometry of the people
    # lists of length T with (B, V, 3), (F, 3), (B, 3)
    if len(meshes) == 1:
        world_smpl = meshes[0]
        scene_dict["geometry"] = [smpl_to_geometry(
            world_smpl["vertices"], world_smpl["faces"], vis_mask, track_ids
        )]
    else:
        world_smpl, world_obj = meshes
        scene_dict["geometry"] = []
        scene_dict["geometry"].append(
            smpl_to_geometry(
                world_smpl["vertices"], world_smpl["faces"], vis_mask, track_ids
            )
        )
        scene_dict["geometry"].append(
            smpl_to_geometry(
                world_obj["vertices"], world_obj["faces"], vis_mask, track_ids
            )
        )

    if T_w2c is None:
        T_w2c = torch.eye(4)[None]

    T_c2w = torch.linalg.inv(T_w2c)
    # rotate the camera slightly down and translate back and up
    T = cam_util.make_4x4_pose(
        cam_util.rotx(-np.pi / 10), torch.tensor([0, -1, -2])
    ).to(T_c2w.device)

    scene_dict["cameras"] = {
        "src_cam": T_c2w,
        "front": torch.einsum("ij,...jk->...ik", T, T_c2w),
    }

    if floor_plane is not None:
        # compute the ground transform
        # use the first appearance of a track as the reference point
        tid, sid = torch.where(vis_mask > 0)
        idx = tid[torch.argmin(sid)]
        root = world_smpl["joints"][idx, 0, 0].detach().cpu()
        floor = parse_floor_plane(floor_plane.detach().cpu())
        R, t = get_plane_transform(torch.tensor([0.0, 1.0, 0.0]), floor, root)
        scene_dict["ground"] = cam_util.make_4x4_pose(R, t)

    return scene_dict


def render_scene_dict(renderer, scene_dict, out_name, fps=30, **kwargs):
    # lists of T (B, V, 3), (B, 3), (F, 3)
    verts, colors, faces, bounds = scene_dict["geometry"]
    print("NUM VERTS", len(verts))

    # add a top view
    scene_dict["cameras"]["above"] = cam_util.make_4x4_pose(
        torch.eye(3), torch.tensor([0, 0, -10])
    )[None]

    for cam_name, cam_poses in scene_dict["cameras"].items():
        print("rendering scene for", cam_name)
        # cam_poses are (T, 4, 4)
        render_bg = cam_name == "src_cam"
        ground_pose = scene_dict.get("ground", None)
        frames = renderer.render_video(
            cam_poses[None], verts, faces, colors, render_bg, ground_pose=ground_pose
        )
        imageio.mimwrite(f"{out_name}_{cam_name}.mp4", frames, fps=fps)


def animate_scene(
    vis,
    scene,
    out_name,
    seq_name=None,
    accumulate=False,
    render_views=["src_cam", "front", "above", "side"],
    render_bg=True,
    render_cam=True,
    render_ground=True,
    debug=False,
    **kwargs,
):
    if len(render_views) < 1:
        return

    if 'input' in os.path.basename(out_name):
        mesh_flag = False
    else:
        mesh_flag = True
    scene, meshes, mesh_hum, mesh_obj = build_pyrender_scene(
        vis,
        scene,
        seq_name,
        render_views=render_views,
        render_cam=render_cam,
        accumulate=accumulate,
        debug=debug,
        mesh_flag=mesh_flag,
    )

    print("RENDERING VIEWS", scene["cameras"].keys())
    render_ground = render_ground and "ground" in scene
    save_paths = []
    for cam_name, cam_poses in scene["cameras"].items():
        is_src = cam_name == "src_cam"
        show_bg = is_src and render_bg
        show_ground = render_ground and not is_src
        show_cam = render_cam and not is_src
        vis_name = f"{out_name}_{cam_name}"
        print(f"{cam_name} has {len(cam_poses)} poses")
        skip = 10 if debug else 1
        vis.set_camera_seq(cam_poses[::skip])
        save_path = vis.animate(
            vis_name,
            render_bg=show_bg,
            render_ground=show_ground,
            render_cam=show_cam,
            **kwargs,
        )
        save_paths.append(save_path)
    if 'root_fit' in os.path.basename(out_name):
        mesh_save_path = os.path.dirname(save_paths[0]) + "/meshes_root.obj"
    else:
        mesh_save_path = os.path.dirname(save_paths[0]) + "/meshes_smooth.obj"
    if meshes:
        # Transform meshes to save
        meshes[0].vertices[:, 1] *= -1  # flip y-axis
        meshes[0].vertices[:, 2] *= -1  # flip z-axis
        if mesh_hum is not None:
            mesh_hum[0].vertices[:, 1] *= -1
            mesh_hum[0].vertices[:, 2] *= -1
        if mesh_obj is not None:
            mesh_obj[0].vertices[:, 1] *= -1
            mesh_obj[0].vertices[:, 2] *= -1

        meshes[0].export(mesh_save_path)
        mesh_hum[0].export(mesh_save_path.replace(".obj", "_hum.obj"))
        mesh_obj[0].export(mesh_save_path.replace(".obj", "_obj.obj"))

    return save_paths


def build_pyrender_scene(
    vis,
    scene,
    seq_name,
    render_views=["src_cam", "front", "above", "side"],
    render_cam=True,
    accumulate=False,
    debug=False,
    mesh_flag=False,
):
    """
    :param vis (viewer object)
    :param scene (scene dict with geometry, cameras, etc)
    :param accumulate (optional bool, default False) whether to render entire trajectory together
    :param render_views (list str) camera views to render
    """
    if len(render_views) < 1:
        return

    assert all(view in ["src_cam", "front", "above", "side"] for view in render_views)

    scene = move_to(detach_all(scene), "cpu")
    src_cams = scene["cameras"]["src_cam"]
    verts = []
    colors = []
    faces = []
    bounds = []
    for geometry in scene["geometry"]:
        v, c, f, b = geometry
        verts.append(v)
        colors.append(c)
        faces.append(f)
        bounds.append(b)
    T = len(verts[0])
    print(f"{T} mesh frames")

    # set camera views
    if not "cameras" in scene:
        scene["cameras"] = {}

    # remove default views from source camera perspective if desired
    if "src_cam" not in render_views:
        scene["cameras"].pop("src_cam", None)
    if "front" not in render_views:
        scene["cameras"].pop("front", None)

    # add static viewpoints if desired
    top_pose = []
    side_pose = []
    skip = []
    for b in bounds:
        tp, sp, s = get_static_views(seq_name, b)
        top_pose.append(tp)
        side_pose.append(sp)
        skip.append(s)
    if "above" in render_views:
        scene["cameras"]["above"] = top_pose[0][None]
    if "side" in render_views:
        scene["cameras"]["side"] = side_pose[0][None]

    # accumulate meshes if possible (can only accumulate for static camera)
    moving_cam = "src_cam" in render_views or "front" in render_views
    accumulate = accumulate and not moving_cam
    # skip = _skip if accumulate else 1
    skip = 1

    vis.clear_meshes()

    if "ground" in scene:
        vis.set_ground(scene["ground"])

    if debug:
        skip = 10
    times = list(range(0, T, skip))
    for t in times:
        meshes_hum = make_batch_mesh(verts[0][t], faces[0][t], colors[0][t])
        if len(verts) > 1:
            meshes_obj = make_batch_mesh(verts[1][t], faces[1][t][0][0], colors[1][t])
            # for idx in range(len(meshes_hum)):
            #     meshes_hum[idx].vertices[:, 1] *= -1
            #     meshes_hum[idx].vertices[:, 2] *= -1
            meshes_hum_copy = meshes_hum.copy()
            comb_meshes = [trimesh.util.concatenate(meshes_hum[i], meshes_obj[i]) for i in range(len(meshes_hum))]
            meshes_hum = comb_meshes
        else:
            meshes_obj = None
        if accumulate:
            vis.add_static_meshes(meshes_hum)
        else:
            vis.add_mesh_frame(meshes_hum, debug=debug)

    # add camera markers
    if render_cam:
        if accumulate:
            vis.add_camera_markers_static(src_cams[::skip])
        else:
            vis.add_camera_markers(src_cams[::skip])

    if mesh_flag:
        return scene, meshes_hum, meshes_hum_copy, meshes_obj
    return scene, None, None, None


def get_static_views(seq_name=None, bounds=None):
    print("STATIC VIEWS FOR SEQ NAME", seq_name)
    up = torch.tensor([0.0, 1.0, 0.0])

    skip = get_seq_figure_skip(seq_name)
    top_vp, side_vp = get_seq_static_lookat_points(seq_name, bounds)
    top_source, top_target = top_vp
    side_source, side_target = side_vp
    top_pose = cam_util.lookat_matrix(top_source, top_target, up)
    side_pose = cam_util.lookat_matrix(side_source, side_target, up)
    return top_pose, side_pose, skip


def make_image_grid_2x2(out_path, img_paths, overwrite=False):
    """Tile 4 PNG stills (input + 3 views) into a 2x2 grid PNG (single-image output)."""
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} already exists, skipping.")
        return
    if any(not os.path.isfile(p) for p in img_paths):
        print("not all grid inputs exist!", img_paths)
        return
    import cv2
    imgs = [imageio.imread(p)[..., :3] for p in img_paths]
    h = min(im.shape[0] for im in imgs)
    w = min(im.shape[1] for im in imgs)
    imgs = [cv2.resize(im, (w, h)) for im in imgs]
    top = np.concatenate([imgs[0], imgs[1]], axis=1)
    bot = np.concatenate([imgs[2], imgs[3]], axis=1)
    imageio.imwrite(out_path, np.concatenate([top, bot], axis=0))
    print("wrote image grid to", out_path)

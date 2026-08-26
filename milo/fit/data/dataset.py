import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import glob

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import trimesh

from body_model import OP_NUM_JOINTS, SMPL_JOINTS
from util.logger import Logger

"""
Single-image, single-person dataset for HOI fitting.

The SMPL-H init (milo_init.npz) is produced by the init_smpl pipeline step and
lives, together with the LRM object mesh, the triangulated 3D keypoints, the
camera metadata and the input image, in the per-sequence directory
<base_path>/<seq>/. There is no video / shots / tracks / temporal interpolation:
the fit runs at T=1, B=1.
"""

DEFAULT_GROUND = np.array([0.0, -1.0, 0.0, -0.5])
MIN_KEYP_CONF = 0.4


def get_dataset_from_cfg(cfg):
    args = cfg.data
    if not args.get("use_cams", False):
        args.sources.cameras = ""
    args.sources = expand_source_paths(args.sources)
    print("DATA SOURCES", args.sources)
    check_data_sources(args)
    return SingleImageDataset(args.sources, args.seq)


def expand_source_paths(data_sources):
    return {k: get_data_source(v) for k, v in data_sources.items()}


def get_data_source(source):
    matches = glob.glob(source)
    if len(matches) < 1:
        print(f"{source} does not exist")
        return source  # return anyway for default values
    if len(matches) > 1:
        raise ValueError(f"{source} is not unique")
    return matches[0]


def check_data_sources(args):
    # Single-image: the SMPL-H init (milo_init.npz) is produced by the init_smpl
    # pipeline step next to the input image — nothing to preprocess here.
    return


# Optional eval-only root (set $MILO_EVAL_DATA_ROOT) holding the dataset folders
# below. Only consulted by the --dataset back-compat path when data.images is a
# dataset-name token rather than a real directory; plain single-image inference
# passes a real <data.root>/<seq> dir and never reaches this.
_EVAL_DATA_ROOT = os.environ.get("MILO_EVAL_DATA_ROOT")
_EVAL_SUBDIR = {
    'intercap':  'intercap_slahmr/data_seq',
    'hodome':    'hodome_slahmr/data',
    'imhd':      'imhd_slahmr/data',
}


def base_path_for(images_src):
    """Per-dataset eval root holding <seq>/{image, milo_init.npz, metadata.npz,
    full_img_textured.glb, keypoints_3d_ransac*.npy}. Resolved
    under $MILO_EVAL_DATA_ROOT; returns images_src unchanged if it is unset."""
    if _EVAL_DATA_ROOT is None:
        return images_src
    for key, sub in _EVAL_SUBDIR.items():
        if key in images_src:
            return os.path.join(_EVAL_DATA_ROOT, sub)
    return _EVAL_DATA_ROOT


def input_image_path(seq_dir):
    """First *.jpg in seq_dir that is not a mask/visualization."""
    jpgs = sorted(
        f for f in glob.glob(os.path.join(seq_dir, "*.jpg"))
        if not (f.endswith("_human.jpg") or f.endswith("_object.jpg") or "viz" in os.path.basename(f))
    )
    return jpgs[0] if jpgs else None


def op_keypoints_from_wholebody(all_keypoints):
    """Remap (133,3) ViTPose wholebody keypoints to the (25 body + 21 + 21 hand, 3)
    OpenPose layout the fit consumes (same mapping as the old read_keypoints)."""
    body = np.zeros([OP_NUM_JOINTS, 3], dtype=np.float32)
    body[[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = all_keypoints[:17]
    body[-6:] = all_keypoints[17:23]
    left_hand = all_keypoints[-42:-21]
    right_hand = all_keypoints[-21:]
    return np.concatenate([body, left_hand, right_hand], axis=0).astype(np.float32)


class SingleImageDataset(Dataset):
    def __init__(self, data_sources, seq_name):
        self.seq_name = seq_name
        self.data_sources = data_sources

        # Generic inference: data_sources["images"] is already <data.root>/<seq>.
        # Back-compat: fall back to the dataset-name → physical-root map.
        src = data_sources["images"]
        if os.path.isdir(src):
            self.seq_dir = src
        elif os.path.isdir(os.path.join(base_path_for(src), seq_name)):
            self.seq_dir = os.path.join(base_path_for(src), seq_name)
        else:
            self.seq_dir = src  # reported by the assert below
        self.base_path = os.path.dirname(self.seq_dir)
        self.img_path = input_image_path(self.seq_dir)
        assert self.img_path is not None, f"no input image in {self.seq_dir}"

        img_h, img_w = cv2.imread(self.img_path).shape[:2]
        self.img_size = (img_w, img_h)

        # one image, one person, single frame
        self.seq_len = 1
        self.n_tracks = 1
        self.track_ids = [1]
        self.sel_img_paths = [self.img_path]
        self.sel_img_names = [get_name(self.img_path)]
        print(f"USING SINGLE IMAGE {img_w}x{img_h}: {self.img_path}")

        # stub attrs for legacy output helpers (optim.output.save_track_info)
        self.track_vis_masks = [np.ones(1, dtype=np.float32)]
        self.data_start, self.data_end = 0, 1
        self.start_idx, self.end_idx = 0, 1

        self.data_dict = {}
        self.cam_data = None

    def __len__(self):
        return self.n_tracks  # 1

    def load_data(self, interp_input=False):  # interp_input kept for call-site compat
        if len(self.data_dict) > 0:
            return

        self.load_camera_data()
        seq_dir = self.seq_dir

        # --- metadata + intrinsics (metadata.npz optional for in-the-wild) ---
        meta_path = os.path.join(seq_dir, "metadata.npz")
        if os.path.exists(meta_path):
            metadata = dict(np.load(meta_path))
            fx, fy = float(metadata["focal_length"][0]), float(metadata["focal_length"][1])
            cx, cy = float(metadata["principal_point"][0]), float(metadata["principal_point"][1])
        else:
            _w, _h = self.img_size
            f = 0.5 * (_h + _w)
            fx = fy = f
            cx, cy = _w / 2.0, _h / 2.0
            metadata = {"focal_length": np.array([fx, fy], np.float32),
                        "principal_point": np.array([cx, cy], np.float32)}
            print(f"[dataset] No metadata.npz — default focal={f:.1f}, principal=center")
        self.intrins = torch.tensor([fx, fy, cx, cy]).float()

        # --- SMPL-H init from the init_smpl step ---
        init = np.load(os.path.join(seq_dir, "milo_init.npz"))
        n_joints = len(SMPL_JOINTS) - 1  # 21
        init_body_pose = init["body_pose"][:n_joints][None].astype(np.float32)   # (1,21,3)
        init_root_orient = init["global_orient"][None].astype(np.float32)         # (1,3)
        init_trans = init["cam_trans"][None].astype(np.float32)                   # (1,3) PHALP convention
        init_hand_pose = np.concatenate(
            [init["left_hand_pose"], init["right_hand_pose"]], axis=0
        )[None].astype(np.float32)                                                # (1,30,3)

        # 2D keypoints: 133 wholebody -> 67 OpenPose layout, zero low-confidence
        joints2d = op_keypoints_from_wholebody(init["vitpose_keypoints"])[None]   # (1,67,3)
        joints2d[np.repeat(joints2d[:, :, [2]] < MIN_KEYP_CONF, 3, axis=2)] = 0

        # The HMR2 init translation is in the PHALP camera (focal 0.5*(H+W),
        # principal at image center). Re-express it in the metadata intrinsics,
        # preserving the 2D projection (depth scales with the focal ratio; the x/y
        # offset accounts for the principal-point shift).
        img_w, img_h = self.img_size
        f_phalp = 0.5 * (img_h + img_w)
        cx_phalp, cy_phalp = img_w / 2.0, img_h / 2.0
        fx_m, fy_m, cx_m, cy_m = [float(v) for v in self.intrins]
        f_m = 0.5 * (fx_m + fy_m)
        tz = init_trans[..., 2]
        t_meta = init_trans.copy()
        t_meta[..., 0] = init_trans[..., 0] + tz * (cx_phalp - cx_m) / f_phalp
        t_meta[..., 1] = init_trans[..., 1] + tz * (cy_phalp - cy_m) / f_phalp
        t_meta[..., 2] = tz * (f_m / f_phalp)
        init_trans = t_meta

        init_obj_path = os.path.join(seq_dir, "full_img_textured.glb")
        if not os.path.exists(init_obj_path):
            init_obj_path = os.path.join(seq_dir, "full_img_textured.obj")

        # stored as per-track lists of length 1 (one person) so the legacy
        # output helpers (save_input_poses / get_input_dict / render_keypoints_2d)
        # that np.stack / index by track keep working.
        self.data_dict = {
            "joints2d": [joints2d],
            "init_body_pose": [init_body_pose],
            "init_root_orient": [init_root_orient],
            "init_trans": [init_trans],
            "init_hand_pose": [init_hand_pose],
            "floor_plane": [(DEFAULT_GROUND[:3] * DEFAULT_GROUND[3:]).astype(np.float32)],
            "object_path": [init_obj_path],
            "metadata": metadata,
            "kp3d_body_path": os.path.join(seq_dir, "keypoints_3d_ransac.npy"),
            "kp3d_lhand_path": os.path.join(seq_dir, "keypoints_3d_ransac_lhand.npy"),
            "kp3d_rhand_path": os.path.join(seq_dir, "keypoints_3d_ransac_rhand.npy"),
        }

    def __getitem__(self, idx):
        if len(self.data_dict) < 1:
            self.load_data()
        d = self.data_dict

        obs = {}
        obs["joints2d"] = torch.from_numpy(d["joints2d"][idx])            # (1,67,3)
        obs["init_body_pose"] = torch.from_numpy(d["init_body_pose"][idx])  # (1,21,3)
        obs["init_hand_pose"] = torch.from_numpy(d["init_hand_pose"][idx])  # (1,30,3)
        obs["init_root_orient"] = torch.from_numpy(d["init_root_orient"][idx])  # (1,3)
        obs["init_trans"] = torch.from_numpy(d["init_trans"][idx])        # (1,3)
        T = obs["init_trans"].shape[0]  # 1 (also the per-sample T axis builder)

        # input image
        obs["img"] = torch.Tensor(
            cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1) / 255.0  # (C,H,W)

        # object mesh — sam3d/gt are already in MILO convention; others need a 180° X flip
        obj_path = d["object_path"][0]
        if obj_path.endswith(".glb"):
            init_obj_mesh = trimesh.load(obj_path, process=False, force='mesh')
        else:
            init_obj_mesh = trimesh.load(obj_path, process=False)
        if "sam3d" in obj_path or "gt" in obj_path:
            R = torch.eye(3, dtype=torch.float32)
        else:
            R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
        init_verts = torch.Tensor(init_obj_mesh.vertices) @ R.T
        obs["init_obj_verts"] = init_verts.unsqueeze(0).repeat(T, 1, 1)
        obs["obj_faces"] = torch.Tensor(init_obj_mesh.faces).unsqueeze(0).repeat(T, 1, 1)

        # 3D keypoints (same rotation as the object); shape (K, 3/4)
        kp3d_body_raw = torch.from_numpy(np.load(d["kp3d_body_path"])).float()
        kp3d_body, kp3d_body_conf = kp3d_body_raw[..., :3], kp3d_body_raw[..., 3]
        try:
            r = torch.from_numpy(np.load(d["kp3d_lhand_path"])).float()
            kp3d_lhand, kp3d_lhand_conf = r[..., :3], r[..., 3]
        except FileNotFoundError:
            kp3d_lhand, kp3d_lhand_conf = torch.zeros(21, 3), torch.zeros(21)
        try:
            r = torch.from_numpy(np.load(d["kp3d_rhand_path"])).float()
            kp3d_rhand, kp3d_rhand_conf = r[..., :3], r[..., 3]
        except FileNotFoundError:
            kp3d_rhand, kp3d_rhand_conf = torch.zeros(21, 3), torch.zeros(21)

        kp3d_body = kp3d_body @ R.T
        kp3d_lhand = kp3d_lhand @ R.T
        kp3d_rhand = kp3d_rhand @ R.T

        obs["kp3d_body"] = kp3d_body.unsqueeze(0).repeat(T, 1, 1)
        obs["kp3d_lhand"] = kp3d_lhand.unsqueeze(0).repeat(T, 1, 1)
        obs["kp3d_rhand"] = kp3d_rhand.unsqueeze(0).repeat(T, 1, 1)
        obs["kp3d_body_conf"] = kp3d_body_conf.unsqueeze(0).repeat(T, 1)
        obs["kp3d_lhand_conf"] = kp3d_lhand_conf.unsqueeze(0).repeat(T, 1)
        obs["kp3d_rhand_conf"] = kp3d_rhand_conf.unsqueeze(0).repeat(T, 1)

        obs["floor_plane"] = torch.from_numpy(d["floor_plane"][idx])
        obs["vis_mask"] = torch.ones(T)
        obs["seq_interval"] = torch.tensor([0, 1], dtype=torch.int)
        obs["track_interval"] = torch.tensor([0, 1], dtype=torch.int)
        obs["track_id"] = int(self.track_ids[idx])
        obs["seq_name"] = self.seq_name
        return obs

    def load_camera_data(self):
        self.cam_data = CameraData(
            self.data_sources.get("cameras", ""), self.seq_len, self.img_size
        )

    def get_camera_data(self):
        if self.cam_data is None:
            raise ValueError
        cam_dict = self.cam_data.as_dict()
        cam_dict["intrins"] = cam_dict["scale"] * self.intrins.unsqueeze(0).repeat(
            cam_dict["cam_R"].shape[0], 1
        )
        return cam_dict


class CameraData(object):
    def __init__(self, cam_dir, seq_len, img_size):
        self.img_size = img_size
        self.cam_dir = cam_dir
        self.seq_len = seq_len
        self.load_data()

    def load_data(self):
        img_w, img_h = self.img_size
        fpath = os.path.join(self.cam_dir, "cameras.npz") if self.cam_dir else ""
        if fpath and os.path.isfile(fpath):
            Logger.log(f"Loading cameras from {fpath}...")
            cam_R, cam_t, intrins, width, height = load_cameras_npz(fpath)
            self.scale = img_w / width
            self.intrins = self.scale * intrins[: self.seq_len]
            self.cam_R = cam_R[: self.seq_len]
            self.cam_t = cam_t[: self.seq_len]
            self.is_static = False
        else:
            Logger.log("Using static camera (single image).")
            self.scale = 1.0
            default_focal = 0.5 * (img_h + img_w)
            self.intrins = torch.tensor(
                [default_focal, default_focal, img_w / 2, img_h / 2]
            )[None].repeat(self.seq_len, 1)
            self.cam_R = torch.eye(3)[None].repeat(self.seq_len, 1, 1)
            self.cam_t = torch.zeros(self.seq_len, 3)
            self.is_static = True

        Logger.log(f"Images have {img_w}x{img_h}, intrins {self.intrins[0]}")

    def world2cam(self):
        return self.cam_R, self.cam_t

    def cam2world(self):
        R = self.cam_R.transpose(-1, -2)
        t = -torch.einsum("bij,bj->bi", R, self.cam_t)
        return R, t

    def as_dict(self):
        return {
            "cam_R": self.cam_R,      # (T, 3, 3)
            "cam_t": self.cam_t,      # (T, 3)
            "scale": self.scale,      # float
            "intrins": self.intrins,  # (T, 4)
            "static": self.is_static,  # bool
        }


def load_cameras_npz(camera_path):
    assert os.path.splitext(camera_path)[-1] == ".npz"
    cam_data = np.load(camera_path)
    height, width, focal = (
        int(cam_data["height"]),
        int(cam_data["width"]),
        float(cam_data["focal"]),
    )
    w2c = torch.from_numpy(cam_data["w2c"])  # (N, 4, 4)
    cam_R = w2c[:, :3, :3]
    cam_t = w2c[:, :3, 3]
    N = len(w2c)
    if "intrins" in cam_data:
        intrins = torch.from_numpy(cam_data["intrins"].astype(np.float32))
    else:
        intrins = torch.tensor([focal, focal, width / 2, height / 2])[None].repeat(N, 1)
    print(f"Loaded {N} cameras")
    return cam_R, cam_t, intrins, width, height


def is_image(x):
    return (x.endswith(".png") or x.endswith(".jpg")) and not x.startswith(".")


def get_name(x):
    return os.path.splitext(os.path.basename(x))[0]

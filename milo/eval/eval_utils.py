"""GT-mesh helpers for prepare_dataset.py.

Ports of the SMPL-H forward passes used by the original per-dataset collect
scripts, so the eval GT meshes are reproduced bit-for-bit:
  - HodomeSMPLH: easymocap-style layer (NeuralDome mocap jsons carry the full
    156-dim pose; Rh/Th are applied to the vertices about the origin, NOT as
    the SMPL global orient).
  - ImhdBodyModel: egoego's SMPL-H wrapper (16 betas, flat hands, no PCA).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLH
from smplx.lbs import lbs, batch_rodrigues
from smplx.utils import Struct


def rot6d_to_matrix(rot_6d):
    """6D rotation representation -> 3x3 matrices (Zhou et al.)."""
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _dense(a):
    if isinstance(a, np.ndarray) and a.dtype == object:
        a = a.item()
    if hasattr(a, "todense"):
        a = a.todense()
    return np.asarray(a, dtype=np.float32)


class HodomeSMPLH(nn.Module):
    """SMPL-H layer matching NeuralDome's easymocap SMPLHModel forward.

    The mocap jsons store `poses` at the full 156 dims (root + body + both
    hands), so the PCA hand machinery is bypassed; `Rh`/`Th` rotate/translate
    the LBS output about the origin (easymocap convention).
    """

    def __init__(self, model_npz, num_shapes=16):
        super().__init__()
        data = dict(np.load(model_npz, allow_pickle=True))
        self.faces = np.asarray(data["f"], dtype=np.int64)
        self.register_buffer("v_template", torch.from_numpy(_dense(data["v_template"])))
        self.register_buffer("shapedirs", torch.from_numpy(_dense(data["shapedirs"])[..., :num_shapes]))
        posedirs = _dense(data["posedirs"])
        posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T  # (P, V*3)
        self.register_buffer("posedirs", torch.from_numpy(posedirs))
        self.register_buffer("J_regressor", torch.from_numpy(_dense(data["J_regressor"])))
        self.register_buffer("weights", torch.from_numpy(_dense(data["weights"])))
        kintree = data["kintree_table"]
        if kintree.ndim == 2:
            kintree = kintree[0]
        parents = torch.from_numpy(np.asarray(kintree, dtype=np.int64))
        parents[0] = -1
        self.register_buffer("parents", parents)

    @torch.no_grad()
    def forward(self, poses, shapes, Rh, Th, **_ignored):
        """All args are (1, N) arrays/lists from a mocap json annot; returns (V, 3) numpy verts."""
        poses = torch.as_tensor(np.asarray(poses, dtype=np.float32))
        shapes = torch.as_tensor(np.asarray(shapes, dtype=np.float32))
        Rh = torch.as_tensor(np.asarray(Rh, dtype=np.float32))
        Th = torch.as_tensor(np.asarray(Th, dtype=np.float32))
        verts, _ = lbs(shapes, poses, self.v_template, self.shapedirs, self.posedirs,
                       self.J_regressor, self.parents, self.weights)
        rot = batch_rodrigues(Rh)
        verts = torch.matmul(verts, rot.transpose(1, 2)) + Th.unsqueeze(1)
        return verts[0].cpu().numpy()


class ImhdBodyModel(nn.Module):
    """egoego's SMPL-H wrapper (body_model.py), trimmed to what IMHD GT needs:
    .npz model, 16 betas (shapedirs zero-padded), zeroed hand PCA, flat hand mean."""

    def __init__(self, bm_path, num_betas=16):
        super().__init__()
        smpl_dict = np.load(bm_path, encoding="latin1", allow_pickle=True)
        data_struct = Struct(**smpl_dict)
        data_struct.hands_componentsl = np.zeros((0))
        data_struct.hands_componentsr = np.zeros((0))
        data_struct.hands_meanl = np.zeros((15 * 3))
        data_struct.hands_meanr = np.zeros((15 * 3))
        V, D, B = data_struct.shapedirs.shape
        data_struct.shapedirs = np.concatenate(
            [data_struct.shapedirs, np.zeros((V, D, 300 - B))], axis=-1)
        self.bm = SMPLH(bm_path, model_type="smplh", data_struct=data_struct,
                        num_betas=num_betas, batch_size=1,
                        use_pca=False, flat_hand_mean=True)

    @torch.no_grad()
    def forward(self, betas, root_orient, pose_body, pose_hand, trans):
        n_hand = SMPLH.NUM_HAND_JOINTS * 3
        out = self.bm(betas=betas, global_orient=root_orient, body_pose=pose_body,
                      left_hand_pose=pose_hand[:, :n_hand],
                      right_hand_pose=pose_hand[:, n_hand:],
                      transl=trans)
        return out.vertices[0].cpu().numpy(), np.asarray(self.bm.faces, dtype=np.int64)

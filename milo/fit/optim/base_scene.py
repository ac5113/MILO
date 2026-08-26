import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose, run_smpl
from geometry.rotation import (
    rotation_matrix_to_angle_axis,
    batch_rodrigues,
    angle_axis_to_rotation_matrix,
    matrix_to_rotation_6d,
    rot6d_to_rotmat,
)
from util.logger import Logger
from util.tensor import move_to, detach_all

from .helpers import estimate_initial_trans
from .params import CameraParams

# from data.object_data import obj_dict


J_BODY = len(SMPL_JOINTS) - 1  # no root


class BaseSceneModel(nn.Module):
    """
    Scene model of sequences of human poses.
    All poses are in their own INDEPENDENT camera reference frames.
    A basic class mostly for testing purposes.

    Parameters:
        batch_size:  number of sequences to optimize
        seq_len:     length of the sequences
        body_model:  SMPL body model
        pose_prior:  VPoser model
        fit_gender:  gender of model (optional)
    """

    def __init__(
        self,
        batch_size,
        seq_len,
        body_model,
        pose_prior,
        fit_gender="neutral",
        use_init=False,
        opt_cams=False,
        opt_scale=True,
        opt_obj=True,
        **kwargs,
    ):
        super().__init__()
        B, T = batch_size, seq_len
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.body_model = body_model
        self.fit_gender = fit_gender

        self.pose_prior = pose_prior
        self.latent_pose_dim = self.pose_prior.latentD

        self.num_betas = body_model.bm.num_betas
        self.num_pca_comps = body_model.bm.num_pca_comps

        self.smpl2op_map = smpl_to_openpose(
            self.body_model.model_type,
            use_hands=True,
            use_face=False,
            use_face_contour=False,
            openpose_format="coco25",
        )

        self.use_init = use_init
        print("USE INIT", use_init)
        self.opt_scale = opt_scale
        self.opt_cams = opt_cams
        self.opt_obj = opt_obj
        print("OPT SCALE", self.opt_scale)
        print("OPT CAMERAS", self.opt_cams)
        print("OPT OBJECT", self.opt_obj)
        self.params = CameraParams(batch_size)

    def initialize(self, obs_data, cam_data):
        Logger.log("Initializing scene model with observed data")

        # initialize cameras
        self.params.set_cameras(
            cam_data,
            opt_scale=self.opt_scale,
            opt_cams=self.opt_cams,
            opt_focal=self.opt_cams,
        )

        # initialize body params
        B, T = self.batch_size, self.seq_len
        device = next(iter(cam_data.values())).device
        init_betas = torch.zeros(B, self.num_betas, device=device)
        init_scale = torch.ones(B, device=device)

        if self.use_init and "init_body_pose" in obs_data:
            init_pose = obs_data["init_body_pose"][:, :, :J_BODY, :]
            init_pose_latent = self.pose2latent(init_pose)
        else:
            init_pose = torch.zeros(B, T, J_BODY, 3, device=device)
            init_pose_latent = torch.zeros(B, T, self.latent_pose_dim, device=device)

        if self.use_init and "init_hand_pose" in obs_data:
            # project hamer init to hand pose PCA space
            init_hands = obs_data["init_hand_pose"].view(B, T, -1)
            
            right_hand_mean = self.body_model.bm.right_hand_mean
            left_hand_mean = self.body_model.bm.left_hand_mean
            hand_mean = torch.cat([left_hand_mean, right_hand_mean])

            right_hand_components = self.body_model.bm.right_hand_components
            left_hand_components = self.body_model.bm.left_hand_components
            P_right_hand_components = torch.linalg.pinv(right_hand_components)
            P_left_hand_components = torch.linalg.pinv(left_hand_components)
            P = torch.block_diag(P_left_hand_components, P_right_hand_components)

            init_hands = torch.einsum('fp,btf->btp', P, init_hands - hand_mean[None,None])
        else:
            init_hands = torch.zeros(B, T, 2*self.num_pca_comps, device=device)

        # transform into world frame (T, 3, 3), (T, 3)
        R_w2c, t_w2c = cam_data["cam_R"], cam_data["cam_t"]
        R_c2w = R_w2c.transpose(-1, -2)
        t_c2w = -torch.einsum("tij,tj->ti", R_c2w, t_w2c)

        if self.use_init and "init_root_orient" in obs_data:
            init_rot = obs_data["init_root_orient"]  # (B, T, 3)
            init_rot_mat = angle_axis_to_rotation_matrix(init_rot)
            init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w, init_rot_mat)
            init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
        else:
            init_rot = (
                torch.tensor([np.pi, 0, 0], dtype=torch.float32)
                .reshape(1, 1, 3)
                .repeat(B, T, 1)
            )

        init_trans = torch.zeros(B, T, 3, device=device)
        if self.use_init and "init_trans" in obs_data:
            # must offset by the root location before applying camera to world transform
            pred_data = self.pred_smpl(init_trans, init_rot, init_pose, init_betas, init_scale, init_hands)
            root_loc = pred_data["joints3d"][..., 0, :]  # (B, T, 3)
            init_trans = obs_data["init_trans"]  # (B, T, 3)
            init_trans = (
                torch.einsum("tij,btj->bti", R_c2w, init_trans + root_loc)
                + t_c2w[None]
                - root_loc
            )
        else:
            # initialize trans with reprojected joints
            pred_data = self.pred_smpl(init_trans, init_rot, init_pose, init_betas)
            init_trans = estimate_initial_trans(
                init_pose,
                pred_data["joints3d_op"],
                obs_data["joints2d"],
                obs_data["intrins"][:, 0],
            )

        if self.opt_obj:
            # initialize object
            init_obj_trans = torch.zeros(B, T, 3, device=device)
            init_obj_rot = torch.eye(3, device=device).repeat(B, T, 1, 1)
            # init_obj_rot6d = matrix_to_rotation_6d(init_obj_rot)
            init_obj_rot6d = rotation_matrix_to_angle_axis(init_obj_rot)
            init_obj_verts = obs_data["init_obj_verts"]
            obj_faces = obs_data["obj_faces"]
            self.params.set_param("obj_rot6d", init_obj_rot6d)
            self.params.set_param("obj_trans", init_obj_trans)
            self.params.set_param("obj_verts", init_obj_verts)
            self.params.set_param("obj_faces", obj_faces)
            self.params.set_param("obj_scale", torch.ones(B, device=device))

        # Translate human to match object centroid
        with torch.no_grad():
            body_pose = self.latent2pose(init_pose_latent) if init_pose_latent is not None else init_pose
            pred_data = self.pred_smpl(init_trans, init_rot, body_pose, init_betas, init_scale, init_hands)
            human_verts = pred_data["points3d"]          # (B, T, V, 3)
            human_centroid = human_verts.mean(dim=2)      # (B, T, 3)

            # obj_verts shape: (B, T, V_obj, 3)
            obj_centroid = init_obj_verts.mean(dim=2)     # (B, T, 3)

            # shift translation so human centroid aligns with object centroid
            trans_offset = obj_centroid - human_centroid
            init_trans = init_trans + trans_offset
        Logger.log(f"Translated human to match object centroid, offset: {trans_offset[:, 0].detach().cpu()}")

        self.params.set_param("latent_pose", init_pose_latent)
        self.params.set_param("betas", init_betas)
        self.params.set_param("hum_scale", init_scale)
        self.params.set_param("hand_pose", init_hands)
        self.params.set_param("trans", init_trans)
        self.params.set_param("root_orient", init_rot)

    def get_optim_result(self, **kwargs):
        """
        Collect predicted outputs (latent_pose, trans, obj_trans, root_orient, betas, body pose) into dict
        """
        res = self.params.get_dict()
        if "latent_pose" in res:
            res["pose_body"] = self.latent2pose(self.params.latent_pose).detach()

        # add the cameras
        res["cam_R"], res["cam_t"], _, _ = self.params.get_cameras()
        res["intrins"] = self.params.intrins
        return {"world": res}

    def latent2pose(self, latent_pose):
        """
        Converts VPoser latent embedding to aa body pose.
        latent_pose : B x T x D
        body_pose : B x T x J*3
        """
        B, T, _ = latent_pose.size()
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose.reshape((-1, d_latent))
        body_pose = self.pose_prior.decode(latent_pose, output_type="matrot")
        body_pose = rotation_matrix_to_angle_axis(
            body_pose.reshape((B * T * J_BODY, 3, 3))
        ).reshape((B, T, J_BODY * 3))
        return body_pose

    def pose2latent(self, body_pose):
        """
        Encodes aa body pose to VPoser latent space.
        body_pose : B x T x J*3
        latent_pose : B x T x D
        """
        B, T = body_pose.shape[:2]
        body_pose = body_pose.reshape((-1, J_BODY * 3))
        latent_pose_distrib = self.pose_prior.encode(body_pose)
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose_distrib.mean.reshape((B, T, d_latent))
        return latent_pose

    def pred_smpl(self, trans, root_orient, body_pose, betas, scale, hand_pose):
        """
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        betas : B x D
        scale: B
        hand_pose: B x T x 2*C
        """
        smpl_out = run_smpl(self.body_model, trans, root_orient, body_pose, betas, scale, hand_pose)
        joints3d, points3d = smpl_out["joints"], smpl_out["vertices"]

        # select desired joints and vertices
        joints3d_body = joints3d[:, :, : len(SMPL_JOINTS), :]
        joints3d_op = joints3d[:, :, self.smpl2op_map, :]
        # hacky way to get hip joints that align with ViTPose keypoints
        # this could be moved elsewhere in the future (and done properly)
        joints3d_op[:, :, [9, 12]] = (
            joints3d_op[:, :, [9, 12]]
            + 0.25 * (joints3d_op[:, :, [9, 12]] - joints3d_op[:, :, [12, 9]])
            + 0.5
            * (
                joints3d_op[:, :, [8]]
                - 0.5 * (joints3d_op[:, :, [9, 12]] + joints3d_op[:, :, [12, 9]])
            )
        )
        verts3d = points3d[:, :, KEYPT_VERTS, :]

        return {
            "points3d": points3d,  # all vertices
            "verts3d": verts3d,  # keypoint vertices
            "joints3d": joints3d_body,  # smpl joints
            "joints3d_op": joints3d_op,  # OP joints
            "faces": smpl_out["faces"],  # index array of faces
        }

    def pred_params_smpl(self, reproj=True):
        body_pose = self.latent2pose(self.params.latent_pose)
        pred_data = self.pred_smpl(
            self.params.trans, self.params.root_orient, body_pose, self.params.betas, self.params.hum_scale, self.params.hand_pose
        )

        return pred_data
    
    def pred_obj(self, vertices, rot6d, trans, scale):
        """
        Forward pass of the object model and populates pred_data accordingly with
        verts3d.

        vertices: B x V x 3
        rot: B x 6
        trans : B x 3
        scale : B x 1
        """
        # Ensure translations is a B x 1 x 3 tensor to enable broadcasting
        B, T, _ = trans.shape
        trans = trans.unsqueeze(-2)

        # rotmat = rot6d_to_rotmat(rot6d).unsqueeze(0)
        rotmat = batch_rodrigues(rot6d[0]).unsqueeze(0)

        # Add translations to corresponding vertices in each batch
        mean_vertices = vertices.mean(dim=2, keepdim=True)
        scaled_vertices = torch.einsum('btij,btvj->btvi', scale * rotmat, vertices - mean_vertices) + mean_vertices
        translated_vertices = scaled_vertices + trans
        translated_vertices = translated_vertices.view(B, T, -1, 3)

        return {
            "obj_verts_op": translated_vertices,  # object vertices
        }

    def pred_params_obj(self):
        pred_data = self.pred_obj(
            self.params.obj_verts, self.params.obj_rot6d, self.params.obj_trans, self.params.obj_scale
        )

        return pred_data

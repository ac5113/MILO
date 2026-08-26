import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from body_model import smpl_to_openpose
from geometry.rotation import rotation_matrix_to_angle_axis
from geometry import camera as cam_util
from util.logger import Logger


class StageLoss(nn.Module):
    def __init__(self, loss_weights, **kwargs):
        super().__init__()
        self.cur_optim_step = 0
        self.set_loss_weights(loss_weights)
        self.setup_losses(loss_weights, **kwargs)

    def setup_losses(self, *args, **kwargs):
        raise NotImplementedError

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights
        Logger.log("Stage loss weights set to:")
        Logger.log(self.loss_weights)

class RootLoss(StageLoss):
    def setup_losses(
        self,
        loss_weights,
        ignore_op_joints=None,
        joints2d_sigma=100,
        use_chamfer=False,
        robust_loss="none",
        robust_tuning_const=4.6851,
    ):
        self.points3d_loss = Points3DLoss(use_chamfer, robust_loss, robust_tuning_const)

        # Precompute vertex-to-keypoint mapping for visibility masking
        if use_chamfer:
            self._smpl2op_map = smpl_to_openpose(
                'smplh', use_hands=True, use_face=False,
                use_face_contour=False, openpose_format='coco25',
            )
            smplh_data = np.load(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                             '_DATA', 'body_models', 'smplh', 'neutral', 'model.npz'),
                allow_pickle=True,
            )
            self._skinning_weights = smplh_data['weights']       # (6890, 52)
            self._kintree_table = smplh_data['kintree_table']    # (2, 52)

        self.correspondencekp_loss = CorrespondenceKPLoss()

    def forward(self, observed_data, pred_data, valid_mask=None):
        """
        For fitting just global root trans/orientation.
        Only computes joint/point/vert losses, i.e. no priors.
        """
        stats_dict = dict()
        loss = 0.0

        opt_human = True

        # All vertices to non-corresponding observed points in 3D space
        if opt_human:
            if (
                "init_obj_verts" in observed_data
                and "points3d" in pred_data
                and self.loss_weights["points3d"] > 0.0
            ):
                vert_vis_mask = None
                if (hasattr(self, '_smpl2op_map') and hasattr(self, '_skinning_weights')
                        and "kp3d_body_conf" in observed_data):
                    kp3d_body_conf = observed_data["kp3d_body_conf"]
                    kp3d_lhand_conf = observed_data.get("kp3d_lhand_conf")
                    kp3d_rhand_conf = observed_data.get("kp3d_rhand_conf")
                    if kp3d_lhand_conf is not None and kp3d_rhand_conf is not None:
                        T = pred_data["points3d"].size(1)
                        if kp3d_body_conf.ndim == 2:
                            kp3d_body_conf = kp3d_body_conf.unsqueeze(1).expand(-1, T, -1)
                            kp3d_lhand_conf = kp3d_lhand_conf.unsqueeze(1).expand(-1, T, -1)
                            kp3d_rhand_conf = kp3d_rhand_conf.unsqueeze(1).expand(-1, T, -1)
                        vert_vis_mask = build_vertex_visibility_mask(
                            kp3d_body_conf, kp3d_lhand_conf, kp3d_rhand_conf,
                            self._smpl2op_map, self._skinning_weights,
                            self._kintree_table,
                        )
                cur_loss = self.points3d_loss(
                    observed_data["init_obj_verts"], pred_data["points3d"],
                    vert_vis_mask=vert_vis_mask,
                )
                loss += self.loss_weights["points3d"] * cur_loss
                stats_dict["points3d"] = cur_loss

        # Human correspondence loss
        if opt_human:
            if (
                "joints3d_op" in pred_data
                and "kp3d_body" in observed_data
                and "kp3d_lhand" in observed_data
                and "kp3d_rhand" in observed_data
                and self.loss_weights["hum_correspondence"] > 0.0
            ):
                cur_loss = self.correspondencekp_loss(
                    pred_data["joints3d_op"],
                    observed_data["kp3d_body"], observed_data["kp3d_lhand"],
                    observed_data["kp3d_rhand"],
                    kp3d_body_conf=observed_data.get("kp3d_body_conf"),
                    kp3d_lhand_conf=observed_data.get("kp3d_lhand_conf"),
                    kp3d_rhand_conf=observed_data.get("kp3d_rhand_conf"),
                )
                loss += self.loss_weights["hum_correspondence"] * cur_loss
                stats_dict["hum_correspondence"] = cur_loss

        # If we're optimizing cameras, camera reprojection loss
        if "bg2d_err" in pred_data and self.loss_weights["bg2d"] > 0.0:
            cur_loss = pred_data["bg2d_err"]
            loss += self.loss_weights["bg2d"] * cur_loss
            stats_dict["bg2d_err"] = cur_loss

        # camera smoothness
        if "cam_R" in pred_data and self.loss_weights["cam_R_smooth"] > 0.0:
            cam_R = pred_data["cam_R"]  # (T, 3, 3)
            cur_loss = rotation_smoothness_loss(cam_R[1:], cam_R[:-1])
            loss += self.loss_weights["cam_R_smooth"] * cur_loss
            stats_dict["cam_R_smooth"] = cur_loss

        if "cam_t" in pred_data and self.loss_weights["cam_t_smooth"] > 0.0:
            cam_t = pred_data["cam_t"]  # (T, 3, 3)
            cur_loss = translation_smoothness_loss(cam_t[1:], cam_t[:-1])
            loss += self.loss_weights["cam_t_smooth"] * cur_loss
            stats_dict["cam_t_smooth"] = cur_loss

        return loss, stats_dict


def rotation_smoothness_loss(R1, R2):
    R12 = torch.einsum("...ij,...jk->...ik", R2, R1.transpose(-1, -2))
    aa12 = rotation_matrix_to_angle_axis(R12)
    return torch.sum(aa12**2)


def translation_smoothness_loss(t1, t2):
    return torch.sum((t2 - t1) ** 2)


def camera_smoothness_loss(R1, t1, R2, t2):
    """
    :param R1, t1 (N, 3, 3), (N, 3)
    :param R2, t2 (N, 3, 3), (N, 3)
    """
    R12, t12 = cam_util.compose_cameras(R2, t2, *cam_util.invert_camera(R1, t1))
    aa12 = rotation_matrix_to_angle_axis(R12)
    return torch.sum(aa12**2) + torch.sum(t12**2)


"""
Losses are cumulative
SMPLLoss setup is same as RootLoss
"""


class SMPLLoss(RootLoss):
    def forward(self, observed_data, pred_data, nsteps, valid_mask=None):
        """
        For fitting full shape and pose of SMPL.
        nsteps used to scale single-step losses
        """
        loss, stats_dict = super().forward(
            observed_data, pred_data, valid_mask=valid_mask
        )

        # prior to keep latent pose likely
        if "latent_pose" in pred_data and self.loss_weights["pose_prior"] > 0.0:
            cur_loss = pose_prior_loss(pred_data["latent_pose"], valid_mask)
            loss += self.loss_weights["pose_prior"] * cur_loss
            stats_dict["pose_prior"] = cur_loss

        # prior to keep latent pose likely
        if "hand_pose" in pred_data and self.loss_weights["hand_pose_prior"] > 0.0:
            cur_loss = pose_prior_loss(pred_data["hand_pose"], valid_mask)
            loss += self.loss_weights["hand_pose_prior"] * cur_loss
            stats_dict["hand_pose_prior"] = cur_loss

        # prior to keep PCA shape likely
        if "betas" in pred_data and self.loss_weights["shape_prior"] > 0.0:
            cur_loss = shape_prior_loss(pred_data["betas"])
            loss += self.loss_weights["shape_prior"] * nsteps * cur_loss
            stats_dict["shape_prior"] = cur_loss

        if "hand_pose" in pred_data and self.loss_weights["pose_prior"] > 0.0:
            cam_t = pred_data["hand_pose"]  # (T, 3, 3)
            cur_loss = translation_smoothness_loss(cam_t[1:], cam_t[:-1])
            loss += self.loss_weights["pose_prior"] * cur_loss
            stats_dict["hand_pose_smooth"] = cur_loss

        # Regularize latent pose — only penalize large deviations from root_fit       
        if (
            "init_fitted_latent_pose" in observed_data 
            and "latent_pose" in pred_data
            and self.loss_weights.get("init_pose_reg", 0.0) > 0.0
        ):                
            residual = pred_data["latent_pose"] - observed_data["init_fitted_latent_pose"]
            per_elem = residual ** 2
            threshold_sq = 2.5  # allows per-element deviation up to 1.0 before penalizing
            cur_loss = torch.sum(F.relu(per_elem - threshold_sq))
            loss += self.loss_weights["init_pose_reg"] * cur_loss
            stats_dict["init_pose_reg"] = cur_loss

        return loss, stats_dict


def hand_pose_smoothness_loss(h1, h2):
    return torch.sum((h2 - h1) ** 2)

def joints3d_loss(joints3d_obs, joints3d_pred, mask=None):
    """
    :param joints3d_obs (B, T, J, 3)
    :param joints3d_pred (B, T, J, 3)
    :param mask (optional) (B, T)
    """
    B, T, *dims = joints3d_obs.shape
    vis_mask = get_visible_mask(joints3d_obs)
    if mask is not None:
        vis_mask = vis_mask & mask.reshape(B, T, *(1,) * len(dims)).bool()
    loss = (joints3d_obs[vis_mask] - joints3d_pred[vis_mask]) ** 2
    loss = 0.5 * torch.sum(loss)
    return loss


def verts3d_loss(verts3d_obs, verts3d_pred, mask=None):
    """
    :param verts3d_obs (B, T, V, 3)
    :param verts3d_pred (B, T, V, 3)
    :param mask (optional) (B, T)
    """
    B, T, *dims = verts3d_obs.shape
    vis_mask = get_visible_mask(verts3d_obs)
    if mask is not None:
        assert mask.shape == (B, T)
        vis_mask = vis_mask & mask.reshape(B, T, *(1,) * len(dims)).bool()
    loss = (verts3d_obs[vis_mask] - verts3d_pred[vis_mask]) ** 2
    loss = 0.5 * torch.sum(loss)
    return loss


def get_visible_mask(obs_data):
    """
    Given observed data gets the mask of visible data (that actually contributes to the loss).
    """
    return torch.logical_not(torch.isinf(obs_data))


def build_vertex_visibility_mask(kp3d_body_conf, kp3d_lhand_conf, kp3d_rhand_conf, 
                                 smpl2op_map, skinning_weights, kintree_table, conf_threshold=0.0):
      """
      Build a per-vertex visibility mask based on GT keypoint confidence.
      Unmapped joints inherit visibility from their parent in the kinematic tree.
      """
      num_verts, num_joints = skinning_weights.shape  # (6890, 52)
      dominant_joint = skinning_weights.argmax(axis=1)  # (6890,)

      all_conf = torch.cat([kp3d_body_conf, kp3d_lhand_conf, kp3d_rhand_conf], dim=-1)
      B, T, _ = all_conf.shape
      device = all_conf.device

      # Step 1: Build per-joint visibility from keypoints (B, T, num_joints)
      joint_vis = torch.zeros(B, T, num_joints, dtype=torch.bool, device=device)

      joints_with_keypoints = set()
      for kp_idx in range(len(smpl2op_map)):
          joint_idx = int(smpl2op_map[kp_idx])
          if joint_idx >= num_joints:
              continue
          joints_with_keypoints.add(joint_idx)
          kp_visible = all_conf[..., kp_idx] > conf_threshold  # (B, T)
          joint_vis[..., joint_idx] |= kp_visible

      # Step 2: Unmapped joints inherit visibility from parent (topological order)
      kintree_parent = kintree_table[0]  # parent index for each joint
      for j in range(num_joints):
          if j not in joints_with_keypoints:
              parent = int(kintree_parent[j])
              if parent < num_joints:
                  joint_vis[..., j] = joint_vis[..., parent]
              else:
                  joint_vis[..., j] = True  # root with no mapping defaults to visible

      # Step 3: Map joint visibility -> vertex visibility via dominant joint
      dominant_joint_tensor = torch.from_numpy(dominant_joint).long().to(device)
      vert_vis_mask = joint_vis[..., dominant_joint_tensor]  # (B, T, V)

      return vert_vis_mask


class Points3DLoss(nn.Module):
    def __init__(
        self,
        use_chamfer=False,
        robust_loss="bisquare",
        robust_tuning_const=4.6851,
    ):
        super().__init__()

        if not use_chamfer:
            self.active = False
            return

        self.active = True

        robust_choices = ["none", "bisquare", "gm"]
        if robust_loss not in robust_choices:
            Logger.log(
                "Not a valid robust loss: %s. Please use %s"
                % (robust_loss, str(robust_choices))
            )
            exit()

        from utils.chamfer_distance import ChamferDistance

        self.chamfer_dist = ChamferDistance()

        self.robust_loss = robust_loss
        self.robust_tuning_const = robust_tuning_const

    def forward(self, points3d_obs, points3d_pred, vert_vis_mask=None):
        if not self.active:
            return torch.tensor(0.0, dtype=torch.float32, device=points3d_obs.device)

        # one-way chamfer
        B, T, N_obs, _ = points3d_obs.size()
        N_pred = points3d_pred.size(2)
        points3d_obs = points3d_obs.reshape((B * T, -1, 3))
        points3d_pred = points3d_pred.reshape((B * T, -1, 3))

        # obs2pred_sqr_dist, pred2obs_sqr_dist = self.chamfer_dist(
        #     points3d_obs, points3d_pred
        # )

        # Mask out invisible vertices by moving them far away
        # so they are never nearest neighbors in chamfer distance
        if vert_vis_mask is not None:
            vert_vis_mask_flat = vert_vis_mask.reshape((B * T, -1))
            num_visible = vert_vis_mask_flat.sum().item()
            num_total = vert_vis_mask_flat.numel()
            points3d_pred = points3d_pred.clone()
            points3d_pred[~vert_vis_mask_flat] = 1e8

        obs2pred_sqr_dist, pred2obs_sqr_dist = self.chamfer_dist(
            points3d_obs, points3d_pred
        )

        obs2pred_sqr_dist = obs2pred_sqr_dist.reshape((B, T * N_obs))
        pred2obs_sqr_dist = pred2obs_sqr_dist.reshape((B, T * N_pred))

        weighted_obs2pred_sqr_dist, w = apply_robust_weighting(
            obs2pred_sqr_dist.sqrt(),
            robust_loss_type=self.robust_loss,
            robust_tuning_const=self.robust_tuning_const,
        )

        # loss = torch.sum(weighted_obs2pred_sqr_dist)
        loss = torch.sum(weighted_obs2pred_sqr_dist) / (B * T * N_obs)
        # loss = 0.5 * loss
        return loss


def pose_prior_loss(latent_pose_pred, mask=None):
    """
    :param latent_pose_pred (B, T, D)
    :param mask (optional) (B, T)
    """
    # prior is isotropic gaussian so take L2 distance from 0
    loss = latent_pose_pred**2
    if mask is not None:
        loss = loss[mask.bool()]
    loss = torch.sum(loss)
    return loss


def shape_prior_loss(betas_pred):
    # prior is isotropic gaussian so take L2 distance from 0
    loss = betas_pred**2
    loss = torch.sum(loss)
    # loss = torch.exp(1 - betas_pred[:, 0]).square().sum()
    return loss


def joints3d_smooth_loss(joints3d_pred, mask=None):
    """
    :param joints3d_pred (B, T, J, 3)
    :param mask (optional) (B, T)
    """
    # minimize delta steps
    B, T, *dims = joints3d_pred.shape
    loss = (joints3d_pred[:, 1:, :, :] - joints3d_pred[:, :-1, :, :]) ** 2
    if mask is not None:
        mask = mask.bool()
        mask = mask[:, 1:] & mask[:, :-1]
        loss = loss[mask]
    loss = 0.5 * torch.sum(loss)
    return loss


def apply_robust_weighting(
    res, robust_loss_type="bisquare", robust_tuning_const=4.6851
):
    """
    Returns robustly weighted squared residuals.
    - res : torch.Tensor (B x N), take the MAD over each batch dimension independently.
    """
    robust_choices = ["none", "bisquare"]
    if robust_loss_type not in robust_choices:
        print(
            "Not a valid robust loss: %s. Please use %s"
            % (robust_loss_type, str(robust_choices))
        )

    w = None
    detach_res = (
        res.clone().detach()
    )  # don't want gradients flowing through the weights to avoid degeneracy
    if robust_loss_type == "none":
        w = torch.ones_like(detach_res)
    elif robust_loss_type == "bisquare":
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w


def robust_std(res):
    """
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (B x N)

    Returns:
    - std : B x 1
    """
    B = res.size(0)
    med = torch.median(res, dim=-1)[0].reshape((B, 1))
    abs_dev = torch.abs(res - med)
    MAD = torch.median(abs_dev, dim=-1)[0].reshape((B, 1))
    std = MAD / 0.67449
    return std


def bisquare_robust_weights(res, tune_const=4.6851):
    """
    Bisquare (Tukey) loss.
    See https://www.mathworks.com/help/curvefit/least-squares-fitting.html

    - residuals
    """
    # print(res.size())
    norm_res = res / (robust_std(res) * tune_const)
    # NOTE: this should use absolute value, it's ok right now since only used for 3d point cloud residuals
    #   which are guaranteed positive, but generally this won't work)
    outlier_mask = norm_res >= 1.0

    # print(torch.sum(outlier_mask))
    # print('Outlier frac: %f' % (float(torch.sum(outlier_mask)) / res.size(1)))

    w = (1.0 - norm_res**2) ** 2
    w[outlier_mask] = 0.0

    return w


def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


class CorrespondenceKPLoss(nn.Module):
    """
    Computes robust correspondence loss between predicted and observed 3D joints.
    Uses Geman-McClure robust loss function to handle outliers.
    Ignores keypoints that failed triangulation (marked as [0, 0, 0]).
    """
    def __init__(self, body_sigma=0.3, hand_sigma=0.1, zero_threshold=1e-6):
        super().__init__()
        self.body_sigma = body_sigma
        self.hand_sigma = hand_sigma  # Lower sigma for more robust hand tracking
        self.zero_threshold = zero_threshold  # Threshold to detect failed triangulations

    def forward(self, joints3d_pred, kp3d_body, kp3d_lhand, kp3d_rhand,
                kp3d_body_conf=None, kp3d_lhand_conf=None, kp3d_rhand_conf=None):
        
        # Extract body and hand joints from predictions
        joints3d_body = joints3d_pred[:, :, :25, :]
        joints3d_lhand = joints3d_pred[:, :, -42:-21, :]
        joints3d_rhand = joints3d_pred[:, :, -21:, :]

        # Validity masks (zero-vector check)
        valid_mask_body = ~(torch.abs(kp3d_body).sum(-1) < self.zero_threshold)
        valid_mask_lhand = ~(torch.abs(kp3d_lhand).sum(-1) < self.zero_threshold)
        valid_mask_rhand = ~(torch.abs(kp3d_rhand).sum(-1) < self.zero_threshold)

        # Compute residuals
        residual_body = joints3d_body - kp3d_body
        residual_lhand = joints3d_lhand - kp3d_lhand
        residual_rhand = joints3d_rhand - kp3d_rhand

        # Apply robust loss
        robust_body = gmof(residual_body, self.body_sigma)
        robust_lhand = gmof(residual_lhand, self.hand_sigma)
        robust_rhand = gmof(residual_rhand, self.hand_sigma)

        # Build per-keypoint weights: validity * confidence
        # conf shape: (B, T, K) or (B, K) -> expand to (B, T, K, 1)
        weight_body = valid_mask_body.float().unsqueeze(-1)
        weight_lhand = valid_mask_lhand.float().unsqueeze(-1)
        weight_rhand = valid_mask_rhand.float().unsqueeze(-1)

        if kp3d_body_conf is not None:
            # Ensure conf has same leading dims as the mask
            conf_b = kp3d_body_conf
            if conf_b.ndim == weight_body.ndim - 1:
                conf_b = conf_b.unsqueeze(-1)
            weight_body = weight_body * conf_b

        if kp3d_lhand_conf is not None:
            conf_l = kp3d_lhand_conf
            if conf_l.ndim == weight_lhand.ndim - 1:
                conf_l = conf_l.unsqueeze(-1)
            weight_lhand = weight_lhand * conf_l

        if kp3d_rhand_conf is not None:
            conf_r = kp3d_rhand_conf
            if conf_r.ndim == weight_rhand.ndim - 1:
                conf_r = conf_r.unsqueeze(-1)
            weight_rhand = weight_rhand * conf_r

        loss = (torch.sum(weight_body * robust_body)
                + torch.sum(weight_lhand * robust_lhand)
                + torch.sum(weight_rhand * robust_rhand))

        return loss

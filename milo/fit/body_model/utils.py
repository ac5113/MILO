import torch
from .specs import SMPL_JOINTS


def run_smpl(body_model, trans, root_orient, body_pose, betas=None, scale=None, hand_pose=None):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    scale: (optional) B
    hand_pose : (optional) B x T x pca_comps*2
    """
    B, T, _ = trans.shape
    bm_batch_size = body_model.bm.batch_size
    assert bm_batch_size % B == 0
    seq_len = bm_batch_size // B
    bm_num_betas = body_model.bm.num_betas
    bm_num_pca_comps = body_model.bm.num_pca_comps
    J_BODY = len(SMPL_JOINTS) - 1  # all joints except root
    if hand_pose is None:
        hand_pose = torch.zeros(B, T, 2*bm_num_pca_comps, device=trans.device)
        Thand = T
    else:
        _, Thand, _ = hand_pose.shape
    if T == 1:
        # must expand to use with body model (flatten any (B,T,J,3) pose first)
        trans = trans.reshape(B, T, 3).expand(B, seq_len, 3)
        root_orient = root_orient.reshape(B, T, 3).expand(B, seq_len, 3)
        body_pose = body_pose.reshape(B, T, -1).expand(B, seq_len, J_BODY * 3)
    elif T != seq_len:
        trans, root_orient, body_pose = zero_pad_tensors(
            [trans, root_orient, body_pose], seq_len - T
        )
    if Thand == 1:
        hand_pose = hand_pose.expand(B, seq_len, bm_num_pca_comps * 2)
    elif Thand!= seq_len:
        hand_pose = zero_pad_tensors(
            [hand_pose], seq_len - Thand
        )[0]

    if betas is None:
        betas = torch.zeros(B, bm_num_betas, device=trans.device)
    betas = betas.reshape((B, 1, bm_num_betas)).expand((B, seq_len, bm_num_betas))
    transl = trans.reshape((B * seq_len, -1))
    smpl_body = body_model(
        pose_body=body_pose.reshape((B * seq_len, -1)),
        pose_hand=hand_pose.reshape((B * seq_len, -1)),
        betas=betas.reshape((B * seq_len, -1)),
        root_orient=root_orient.reshape((B * seq_len, -1)),
        trans=transl * 0.,
        scale=torch.Tensor([1.0]).to(body_pose.device),
        # scale=scale,
    )

    points3d = smpl_body.v.reshape(B, seq_len, -1, 3)[:, :T]
    joints3d = smpl_body.Jtr.reshape(B, seq_len, -1, 3)[:, :T]
    
    # scale outside
    if scale:
        scaled_points3d = points3d * scale
        scaled_joints3d = joints3d * scale
    else:
        scaled_points3d = points3d
        scaled_joints3d = joints3d

    # scale lbs
    # scaled_points3d = points3d
    # scaled_joints3d = joints3d

    if scaled_points3d.shape[1] == 1:
        transl = transl[[0]]
    scaled_points3d += transl.unsqueeze(dim=1)
    scaled_joints3d += transl.unsqueeze(dim=1)

    return {
        "joints": scaled_joints3d,
        "vertices": scaled_points3d,
        "faces": smpl_body.f,
    }

# def run_obj(obj_trans, obj_scale, init_obj_verts, obj_faces):
#     """
#     Translating the object vertices

#     obj_trans : B x T x 3
#     obj_scale : B x T
#     init_obj_verts : B x T x V x 3
#     obj_faces : B x T x F x 3
#     """
#     B, T, _ = obj_trans.shape
#     init_obj_verts = init_obj_verts.clone()
#     obj_trans = obj_trans.clone()
#     obj_scale = obj_scale.clone()
#     obj_faces = obj_faces.clone()
#     obj_trans = obj_trans.view(-1, 1, 3)

#     # Center, apply scale and move back
#     obj_verts = init_obj_verts - init_obj_verts.mean(dim=2, keepdim=True)
#     obj_verts = obj_verts * obj_scale.view(-1, 1, 1, 1) + init_obj_verts.mean(dim=2, keepdim=True)
#     obj_verts = obj_verts + obj_trans
#     obj_verts = obj_verts.view(B, T, -1, 3)
#     return {
#         "vertices": obj_verts,
#         "faces": obj_faces,
#     }

def run_obj(obj_trans, obj_scale, init_obj_verts, obj_faces, obj_rot6d=None):
      B, T, _ = obj_trans.shape
      init_obj_verts = init_obj_verts.clone()
      obj_trans = obj_trans.view(B, T, 1, 3)

      mean_vertices = init_obj_verts.mean(dim=2, keepdim=True)
      centered = init_obj_verts - mean_vertices

      if obj_rot6d is not None:
          from geometry.rotation import batch_rodrigues
          rotmat = batch_rodrigues(obj_rot6d[0]).unsqueeze(0)  # (1, T, 3, 3)
          scale_val = obj_scale.view(B, 1, 1, 1) if obj_scale.ndim == 1 else obj_scale.view(-1, 1, 1, 1)
          centered = torch.einsum('btij,btvj->btvi', scale_val * rotmat, centered)
      else:
          scale_val = obj_scale.view(B, 1, 1, 1) if obj_scale.ndim == 1 else obj_scale.view(-1, 1, 1, 1)
          centered = centered * scale_val

      obj_verts = centered + mean_vertices + obj_trans
      return {"vertices": obj_verts, "faces": obj_faces}

def zero_pad_tensors(pad_list, pad_size):
    """
    Assumes tensors in pad_list are B x T x D and pad temporal dimension
    """
    B = pad_list[0].size(0)
    new_pad_list = []
    for pad_idx, pad_tensor in enumerate(pad_list):
        padding = torch.zeros((B, pad_size, pad_tensor.size(2))).to(pad_tensor)
        new_pad_list.append(torch.cat([pad_tensor, padding], dim=1))
    return new_pad_list

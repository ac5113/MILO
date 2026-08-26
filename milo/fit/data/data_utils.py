import cv2
import torch
import numpy as np

def obj_param_processing(obj_param, cam_param, center, do_flip, rot):
    pose, trans, obj_name = obj_param['pose'], obj_param['trans'], obj_param['name']

    if do_flip: assert "Object cannot be flipped!"

    # apply camera extrinsic (rotation)
    # merge pose/trans and camera rotation 
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
        pose, _ = cv2.Rodrigues(pose)
        pose, _ = cv2.Rodrigues(np.dot(R,pose))
        trans = np.dot(R, trans.T).reshape(-1)
    if 't' in cam_param:
        trans = trans + np.array(cam_param['t'], dtype=np.float32).reshape(-1)
    
    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)

    pose, _ = cv2.Rodrigues(np.array(pose))
    pose = cv2.Rodrigues(np.dot(rot_aug_mat,pose))[0].reshape(-1)
    trans = np.dot(rot_aug_mat, trans).reshape(-1)    
    trans = trans - center
    return pose, trans, obj_name

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, shift=(0, 0), inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    t_x, t_y = src_w*shift[0], src_h*shift[1]
    src_center = np.array([c_x+t_x, c_y+t_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        inv_trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    trans, inv_trans = trans.astype(np.float32), inv_trans.astype(np.float32)
    return trans, inv_trans

def generate_patch_image(cvimg, bbox, scale, rot, shift, do_flip, out_shape):
    img = cvimg.copy()
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans, inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    
    if do_flip:
        img_patch = img_patch[:, ::-1, :]

    return img_patch, trans, inv_trans

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB': img = img[:,:,::-1]
    img = img.astype(np.float32)
    return img

def img_processing(img, bbox, img_shape):
    scale, rot, shift, color_scale, blur_sigma, do_flip = 1.0, 0.0, (0.0, 0.0), np.array([1,1,1]), 0, False

    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, shift, do_flip, img_shape)
    img = img * color_scale[None,None,:]
    if blur_sigma > 0 : img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    img = np.clip(img, 0, 255)
    return img, trans, inv_trans, rot, do_flip

def process_bbox(bbox, input_shape, target_shape, expand_ratio=1.25, do_sanitize=False):
    if do_sanitize:
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((input_shape[1] - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((input_shape[0] - 1, y1 + np.max((0, h - 1))))
        if w*h > 0 and x2 > x1 and y2 > y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1])
        else:
            return None
    
    # aspect ratio preserving bbox
    bbox = np.array(bbox)
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = target_shape[1] / target_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*expand_ratio
    bbox[3] = h*expand_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def flip_3d_joint(kp, flip_pairs):
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    kp[:,0] = - kp[:,0]
    return kp

def coord3D_processing(coord, r, f, flip_pairs=[]):
    # in-plane rotation
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        
    coord = np.einsum('ij,kj->ki', rot_mat, coord)

    # flip the x coordinates
    if f:
        coord = flip_3d_joint(coord, flip_pairs)
    coord = coord.astype('float32')

    return coord
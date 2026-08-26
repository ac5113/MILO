import cv2
import json
import trimesh
import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
import torch
from torchvision import transforms
import copy

from .data_utils import process_bbox, load_img, img_processing, coord3D_processing, obj_param_processing, batch_rodrigues

# Eval/training-only dataset roots (BEHAVE / CONTHO object models + annotations).
# Unused by single-image inference; set these only to run the BEHAVE/CONTHO
# eval+training paths below. With them unset the asset loaders below raise
# (caught by the lazy guard on `obj_dict`).
_BEHAVE_ROOT = os.environ.get("MILO_BEHAVE_ROOT")
_CONTHO_ROOT = os.environ.get("MILO_CONTHO_ROOT")

joint_set = {
    'name': 'BEHAVE',
    'joint_num': 73,
    'joints_name': (
        'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
        'L_BigToe', 'L_SmallToe', 'L_Heel', 'R_BigToe',  'R_SmallToe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear'
        ),
    'flip_pairs': (
        (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
        (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46), (32, 47), (33, 48), (34, 49), (35, 50), (36, 51),
        (52, 55), (53, 56), (54, 57), (58, 63), (59, 64), (60, 65), (61, 66), (62, 67), (69, 70), (71, 72)
        ),
    'skeleton': (
        (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (9, 13), (13, 16), (16, 18), (18, 20), (9, 12), (12, 15),
        (20, 22), (22, 23), (23, 24), (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36),
        (21, 37), (37, 38), (38, 39), (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51),
        (7, 52), (7, 53), (7, 54), (8, 55), (8, 56), (8, 57), (36, 58), (24, 59), (27, 60), (33, 61), (30, 62), (51, 63), (39, 64), (42, 65), (48, 66), (45, 67), (12, 68), (68, 69), (68, 70), (69, 71), (70, 72)
    )
}

class InteractObject(object):
    def __init__(self, data):
        _base = osp.join(_CONTHO_ROOT, 'data', 'base_data') if _CONTHO_ROOT else 'data/base_data'
        self.path, self.verts64 = data['path'].replace('data/base_data', _base), np.array(data['kps']).astype(np.float32)
        obj = trimesh.load(self.path, process=False, maintain_order=True)
        self.vertices, self.faces = obj.vertices, obj.faces
        self.vertex_num = len(self.vertices)

    def load_template(self):
        template = trimesh.load(self.path, process=False, maintain_order=True)
        return template
    
    def load_verts(self):
        return self.verts64

    def transform_object(self, verts, pose, trans, scale=1.0):
        if pose.ndim == 1: rot, _ = cv2.Rodrigues(pose)
        else: rot = pose
        verts = np.matmul(np.array(verts), rot.T) + trans
        return scale * verts


class InteractObjectDict(object):
    def __init__(self):
        if _BEHAVE_ROOT is None:
            raise FileNotFoundError("MILO_BEHAVE_ROOT not set (BEHAVE object models)")
        obj_info_path = osp.join(_BEHAVE_ROOT, 'objects', '_info.json')

        with open(obj_info_path) as f:
            self.obj_info = json.load(f)

        self.obj_names = []
        for k, v in self.obj_info.items():
            obj_name = v['path'].split('/')[-1].replace('.obj', '')
            
            obj_info = InteractObject(v)
            setattr(self, obj_name, obj_info)
            self.obj_names.append(obj_name)
        self.obj_num = len(self.obj_names)
    
    def get_obj_info(self):
        return self.obj_info

    def get_obj_id(self, name):
        return self.obj_names.index(name)

    def get_obj_verts(self, key):
        return getattr(self, key).vertices64

    def __getitem__(self, key):
        if type(key) is int: key = self.obj_names[key]
        return getattr(self, key)
    
    def transform_object(self, verts, pose, trans, scale=1.0):
        if pose.ndim == 1: rot, _ = cv2.Rodrigues(pose)
        else: rot = pose
        verts = np.matmul(np.array(verts), rot.T) + trans
        return scale * verts

# Eval/training-only asset (BEHAVE/CONTHO object models). Not used by single-image
# inference; instantiated lazily so importing this module never fails on a machine
# without those datasets.
try:
    obj_dict = InteractObjectDict()
except (FileNotFoundError, OSError):
    obj_dict = None

def load_data():
    if _BEHAVE_ROOT is None:
        raise FileNotFoundError("MILO_BEHAVE_ROOT not set (BEHAVE training set)")
    annot_path = osp.join(_BEHAVE_ROOT, 'behave_train.json')
    db = COCO(annot_path)

    datalist = []
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]
        img_path = osp.join(_BEHAVE_ROOT, 'sequences', img['file_name'])

        bbox = process_bbox(ann['bbox'], (img['height'], img['width']), (512, 512), expand_ratio=1.3) 
        if bbox is None: continue

        h2d_keypoints = np.array(ann['h2d_keypoints'], dtype=np.float32).reshape(-1, 2)
        h2d_keypoints_valid = np.ones((len(h2d_keypoints), 1))
        h2d_keypoints = np.concatenate((h2d_keypoints, h2d_keypoints_valid), axis=-1).astype(np.float32)
    
        h3d_keypoints = np.array(ann['h3d_keypoints'], dtype=np.float32).reshape(-1, 3)
        h3d_keypoints_valid = np.ones((len(h3d_keypoints), 1))
        h3d_keypoints = np.concatenate((h3d_keypoints, h3d_keypoints_valid), -1).astype(np.float32)

        cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
        smpl_param = {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k,v in ann['smpl_param'].items()}
        obj_param = {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k,v in ann['obj_param'].items()}

        h_contacts = np.array(ann['h_contacts']).astype(np.float32)
        o_contacts = np.array(ann['o_contacts']).astype(np.float32)

        datalist.append({
            'ann_id': aid,
            'img_id': image_id,
            'img_path': img_path,
            'img_shape': (img['height'], img['width']),
            'bbox': bbox,
            'h2d_keypoints': h2d_keypoints, 
            'h3d_keypoints': h3d_keypoints,
            'h_contacts': h_contacts,
            'o_contacts': o_contacts,
            'cam_param': cam_param,
            'smpl_param': smpl_param,
            'obj_param': obj_param
            })

    return datalist

# datalist = load_data()
# transform = transforms.ToTensor()

def get_item(datalist, index):
    data = copy.deepcopy(datalist[index])

    # image
    img_path, bbox = data['img_path'], data['bbox']
    img = load_img(img_path)
    full_img = img.copy()
    img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, (512, 512))

    # h3d_keypoints
    h3d_keypoints, h3d_keypoints_valid = data['h3d_keypoints'][:,:3], data['h3d_keypoints'][:,-1]
    h3d_keypoints = coord3D_processing(h3d_keypoints, rot, do_flip, joint_set['flip_pairs'])
    root3d_keypoint = h3d_keypoints[0]
    h3d_keypoints = h3d_keypoints - root3d_keypoint
    h3d_keypoints = np.concatenate((h3d_keypoints, h3d_keypoints_valid[:,None]), -1).astype(np.float32)

    # OBJ_param
    obj_pose, obj_trans, obj_name = obj_param_processing(data['obj_param'], data['cam_param'], root3d_keypoint, do_flip, rot)
    obj_pose = batch_rodrigues(torch.tensor(obj_pose).reshape(-1, 3)).squeeze().numpy()
    obj_id = np.array([obj_dict.get_obj_id(obj_name)])

    # Get contacts
    h_contacts, o_contacts = data['h_contacts'], data['o_contacts']

    # Post processing
    img = transform(img.astype(np.float32)/255.0)

    obj_verts = obj_dict[obj_name].load_verts()
    obj_verts = obj_dict.transform_object(obj_verts, obj_pose, obj_trans)
    obj_verts = np.concatenate((obj_verts, np.ones_like(obj_verts[:,[0]])), -1)

    obj_full_mesh = obj_dict[obj_name].load_template()
    obj_full_verts, obj_full_faces = obj_full_mesh.vertices, obj_full_mesh.faces
    obj_full_verts = obj_dict.transform_object(obj_full_verts, obj_pose, obj_trans)
    obj_full_verts = np.concatenate((obj_full_verts, np.ones_like(obj_full_verts[:,[0]])), -1)

    human_mask_path = img_path.replace('.color.jpg', '.person_mask.jpg')
    obj_mask_path = img_path.replace('.color.jpg', '.obj_rend_mask.jpg')
        
    human_mask = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)
    human_mask = cv2.warpAffine(human_mask, img2bb_trans, (512, 512), flags=cv2.INTER_LINEAR)
    human_mask[human_mask<128] = 0; human_mask[human_mask>=128] = 1
    
    obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
    obj_mask = cv2.warpAffine(obj_mask, img2bb_trans, (512, 512), flags=cv2.INTER_LINEAR)
    obj_mask[obj_mask<128] = 0; obj_mask[obj_mask>=128] = 1

    human_mask, obj_mask = torch.tensor(human_mask).float(), torch.tensor(obj_mask).float()
    img = torch.cat((img, human_mask[None], obj_mask[None]))

    inputs = {'img': img, 'full_img': full_img, 'obj_id': obj_id}
    targets = {'h3d_keypoints': h3d_keypoints, 'root3d_keypoint': root3d_keypoint,
                'obj_pose': obj_pose, 'obj_trans': obj_trans, 'obj_verts': obj_verts, 'obj_full_verts': obj_full_verts, 'obj_full_faces': obj_full_faces,
                'h_contacts': h_contacts, 'o_contacts': o_contacts}
    meta_info = {'ann_id': data['ann_id'], 'bbox':bbox, 'img2bb_trans': img2bb_trans, 'bb2img_trans': bb2img_trans, 'img_path': img_path, 'obj_name': obj_name, 'cam_param': data['cam_param']}

    return inputs, targets, meta_info
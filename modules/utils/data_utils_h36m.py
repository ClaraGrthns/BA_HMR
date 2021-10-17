import numpy as np
import os.path as osp
import os
import pickle as pkl
import json
import torch
from tqdm import tqdm
from PIL import Image

from ..smpl_model.config_smpl import *
from ..smpl_model.smpl_pose2mesh import SMPL


def get_data_list_h36m(annot_dir:str,
                    subject_list:list,
                    fitting_thr:int,
                    load_from_pkl:str=None,
                    store_as_pkl:bool=False,
                    out_dir:str=None,
                    ):
        if load_from_pkl is not None:
            with open(load_from_pkl , "rb") as fp:
                datalist = pkl.load(fp)
        else:
            images = []
            cameras = {}
            smpl_params = {}
            joints = {}
            bboxes = {}
            for subject in subject_list:
                ### Load image annotations
                with open(osp.join(annot_dir, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                    annotations = json.load(f)
                images.extend(annotations['images'])
                ### Load cameras
                with open(osp.join(annot_dir, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                    cams = json.load(f)
                cameras[str(subject)] = {cam_id: get_cam_pose_intr(cam) for cam_id, cam in cams.items()}
                ### Load fitted smpl parameter
                with open(osp.join(annot_dir, 'Human36M_subject' + str(subject) + '_smpl_param.json'), 'r') as f:
                    smpl_params[str(subject)] = json.load(f)
                ### Load 3d Joint ground truth (17x3)
                with open(osp.join(annot_dir, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                    joints[str(subject)] = json.load(f)   

            with open(osp.join(annot_dir, 'h36m_bboxes.json'), 'r') as f:
                    bboxes = json.load(f)

            id_to_imgs = {} # Maps img-id to img file
            id_to_imgs = {img['id']: img for img in images}

            datalist = []
            num_smpl_param = 0
            
            for img_id, img in tqdm(id_to_imgs.items(), total=len(id_to_imgs.items())):
                img_name =  img['file_name']  
                subject = img['subject']
                action = img['action_idx']
                subaction = img['subaction_idx']
                frame = img['frame_idx']
                ### check if smpl parameters exist
                try:
                    smpl_param = smpl_params[str(subject)][str(action)][str(subaction)][str(frame)]
                except KeyError:
                    continue
                ### check threshhold of h36m gt and smpl-mesh h36m joints
                joint3d_smpl = np.array(smpl_param['fitted_3d_pose'], np.float32)
                joint3d_h36m_gt = np.array(joints[str(subject)][str(action)][str(subaction)][str(frame)],
                                    dtype=np.float32)
                if get_fitting_error(joint3d_h36m_gt, joint3d_smpl) > fitting_thr: 
                    continue
                    
                cam_id = img['cam_idx']
                if action !=4 or cam_id != 1 or subaction !=1:
                    continue
                cam_param = cameras[str(subject)][str(cam_id)]
                cam_pose, cam_intr = torch.FloatTensor(cam_param['cam_pose']), torch.FloatTensor(cam_param['cam_intr'])
                
                beta = torch.FloatTensor(smpl_param['shape'])
                pose = torch.FloatTensor(smpl_param['pose'])
                trans = torch.FloatTensor(smpl_param['trans']) 
                ### World coordinate --> Camera coordinate
                
                bbox = bboxes[str(img_id)]
                datalist.append({
                    'img_name': img_name,
                    'img_id': img_id,
                    'zarr_id': num_smpl_param,
                    'betas': beta,
                    'poses': pose,
                    'trans': trans,
                    'bbox': bbox,
                    'cam_id': cam_id,
                    'subject': subject,
                    'cam_pose': cam_pose,
                    'cam_intr': cam_intr,
                    })
                
                num_smpl_param += 1
            datalist = sorted(datalist, key=lambda x: x['img_id'])
        if store_as_pkl and out_dir is not None:
            sub_str = f'{min(subject_list)}to{max(subject_list)}'
            with open(osp.join(out_dir, f'datalist_h36m_thr{fitting_thr}_{sub_str}subj.pickle'), 'wb') as fp:
                pkl.dump(datalist, fp)
        return datalist

def get_cam_pose_intr(cam_dict):
    cam_pose = torch.cat((torch.FloatTensor(cam_dict['R']), torch.FloatTensor(cam_dict['t'])[:,None]/1000.), dim = 1)
    cam_pose = torch.cat((cam_pose, torch.FloatTensor([[0, 0, 0, 1]])), dim=0)
    cam_intr = torch.zeros(3,3)
    cam_intr[0,0], cam_intr[1,1] = cam_dict['f']
    cam_intr[0:2,2] = torch.tensor(cam_dict['c'])
    cam_intr[2,2] = 1.
    return {'cam_pose': cam_pose, 'cam_intr': cam_intr}

def get_fitting_error(joint3d_h36m_gt, joint3d_smpl):
    joint3d_h36m_gt = joint3d_h36m_gt - joint3d_h36m_gt[H36M_J17_NAME.index('Pelvis'), None,:] # root-relative
    # translation alignment
    joint3d_smpl = joint3d_smpl - np.mean(joint3d_smpl,0)[None,:] + np.mean(joint3d_h36m_gt,0)[None,:]
    error = np.sqrt(np.sum((joint3d_h36m_gt - joint3d_smpl)**2, 1)).mean()
    return error

def get_bbox_from_json(root_dir='../H36M', border_scale=1.3, store_as_json=None, load_from_json=None): 
    if load_from_json is not None:
        with open(load_from_json, 'r') as json_file:
            bboxes = json.load(json_file)
    else:
        bboxes={}
        annotation_files =  [image_name for image_name in os.listdir(osp.join(root_dir, 'annotations')) if image_name.endswith('_data.json')]
        for annotation_file in annotation_files:
            with open(osp.join(root_dir, 'annotations', annotation_file)) as json_file:
                annotations = json.load(json_file)
            for image, annotation in zip(annotations['images'], annotations['annotations']):
                x_border = image['width']
                y_border = image['height']
                x_min, y_min, width, height = annotation['bbox']
                x_max = x_min+width
                y_max = y_min+height
                delta_x = width * ((border_scale-1) / 2)
                delta_y = height * ((border_scale-1) / 2)
                delta = max(delta_x, delta_y)
                x_min = max(0, int(x_min-delta))
                y_min = max(0, int(y_min-delta))
                x_max = min(x_border, int(x_max+delta))
                y_max = min(y_border, int(y_max+delta))
                bboxes[image['id']] = [int(x_min), int(y_min), int(x_max), int(y_max)]
        if store_as_json is not None:
            with open(osp.join(store_as_json, 'h36m_bboxes.json'), 'w') as fp:
                json.dump(bboxes, fp)
    return bboxes
def get_backgrounds_from_folder(background_fp):
    bground_paths = [osp.join(background_fp, bg_img) for bg_img in os.listdir(background_fp)if bg_img.endswith('jpg')]
    return [np.array(Image.open(img_path)) for img_path in bground_paths]

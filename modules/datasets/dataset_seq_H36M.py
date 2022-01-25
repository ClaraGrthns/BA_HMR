import os
import os.path as osp
from re import sub
import time
import pickle as pkl
import numpy as np
from PIL import Image
import torch
import zarr
import copy


from ..utils.image_utils import to_tensor, transform, transform_visualize, crop_box
from ..utils.data_utils_h36m import get_data_chunk_list_h36m, get_background
from ..utils.geometry import get_smpl_coord_torch


class SequenceWiseH36M(torch.utils.data.Dataset):
    def __init__(self,
                data_path:str='../H36M',
                subject_list:list=[],
                smpl=None,
                load_from_zarr:list=None,
                load_seq_datalist:str=None, 
                load_datalist:list=None,
                len_chunks:int=None,
                img_size:int=224,
                mask:bool = False,
                fitting_thr:int=25,
                store_images:bool=True,
                backgrounds:list=None,
                ):
        super(SequenceWiseH36M, self).__init__()
        self.img_dir = osp.join(data_path, 'images')
        self.annot_dir = osp.join(data_path,'annotations')
        self.load_from_zarr = load_from_zarr
        self.backgrounds = backgrounds
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.smpl = smpl.to(self.device)
        self.mask = mask
        self.store_images = store_images
        self.len_chunks = len_chunks

        self.subject_list= subject_list
        
        chunks, seq_datalist = get_data_chunk_list_h36m(annot_dir=self.annot_dir,
                                                        subject_list=self.subject_list,
                                                        fitting_thr=fitting_thr,
                                                        len_chunks=len_chunks,
                                                        load_seq_datalist=load_seq_datalist,
                                                        load_datalist=load_datalist,
                                                        store_as_pkl=False,
                                                        )
        self.chunks = chunks
        self.seq_datalist = seq_datalist
        if self.load_from_zarr is not None:
            self.imgs = {subj: torch.from_numpy(zarr.load(zarr_path)) for subj, zarr_path in zip(self.subject_list, self.load_from_zarr) }
        else: 
            self.img_size = img_size
            if self.store_images: 
                dat_length = sum([len(seq_dat) for seq_dat in seq_datalist])
                self.img_cache_indicator = torch.zeros(dat_length, dtype=torch.bool)
                self.img_cache = torch.empty(dat_length, 3, img_size, img_size, dtype=torch.float32)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        chunk = copy.deepcopy(self.chunks[index])
        img_paths = [item['img_name'] for item in chunk]
        zarr_ids = [item['zarr_id'] for item in chunk]
        if self.load_from_zarr is not None:
            subject = chunk[0]['subject']
            zarr_ids = [item['zarr_id'] for item in chunk]
            imgs_tensor = self.imgs[subject][zarr_ids]
        elif self.store_images and torch.all(self.img_cache_indicator[zarr_ids]):
            imgs_tensor = self.img_cache[zarr_ids]
        else:
            imgs_tensor = torch.zeros(len(img_paths), 3, self.img_size, self.img_size)
            for idx, item in enumerate(chunk):
                img = np.array(Image.open(img_paths[idx]))
                if self.mask:
                    sub_dir, img_name = osp.split(item['img_name'])
                    mask_name = img_name.split('.')[-2]+'_mask.jpg'
                    mask_path = osp.join(self.img_dir, sub_dir, mask_name)
                    mask = np.round(np.array(Image.open(mask_path))/255-1)
                ## Cut out mask and use different backgrounds
                img[np.nonzero(mask)] = get_background(img_shape = img.shape, backgrounds=self.backgrounds)[np.nonzero(mask)]
            x_min, y_min, x_max, y_max = item['bbox']
            img = img[y_min:y_max, x_min:x_max]
            img_tensor = to_tensor(img)
            img_tensor = transform(img_tensor, img_size=self.img_size)
            imgs_tensor[idx]=img_tensor
            if self.store_images:
                self.img_cache[zarr_ids[idx]] = img_tensor
                self.img_cache_indicator[zarr_ids[idx]] = True
    
        poses = torch.empty((0, 72), dtype=torch.float32)
        betas = torch.empty((0, 10), dtype=torch.float32)
        transs = torch.empty((0, 3), dtype=torch.float32)
        vertices = torch.zeros(self.len_chunks, 6890, 3, dtype=torch.float32)
        cam_poses = torch.empty((0, 4, 4), dtype=torch.float32)
        cam_intr = torch.FloatTensor(chunk[0]['cam_intr'])

        for item in chunk:
            poses = torch.cat((poses, item['poses'][None]), 0)
            betas = torch.cat((betas, item['betas'][None]), 0)
            transs = torch.cat((transs,item['trans']), 0)
            cam_poses = torch.cat((cam_poses, item['cam_pose'][None]),0)

        poses = poses.to(self.device)
        betas = betas.to(self.device)
        transs = transs.to(self.device)
        cam_poses = cam_poses.to(self.device)

        for idx, (beta, pose, trans, cam_pose) in enumerate(zip(betas, poses, transs, cam_poses)):
            print(beta.device, pose.device, trans.device, cam_pose.device)
            verts, trans, pose = get_smpl_coord_torch(pose=pose[None], beta=beta[None], trans=trans[None], root_idx=0, cam_pose=cam_pose, smpl=self.smpl)
            vertices[idx]= verts
            poses[idx]=pose
            transs[idx] = trans
            
        betas = torch.mean(betas.view(-1, betas.shape[1]), dim=0)
        data = {}
        data['img_paths'] = img_paths
        data['imgs'] = imgs_tensor.to(self.device)
        data['betas'] = betas.expand(self.len_chunks, -1)
        data['poses'] = poses
        data['trans'] = transs
        data['vertices'] = vertices
        data['cam_pose'] = cam_poses
        data['cam_intr'] = cam_intr
        return data
   
    def set_chunks(self):
        chunks,_ = get_data_chunk_list_h36m(seq_datalist=self.seq_datalist)
        self.seq_chunks = chunks

        
        
def get_data(data_path,
            subject_list,
            load_from_zarr,
            load_seq_datalist,
            img_size,
            len_chunks,
            fitting_thr,
            mask,
            smpl,
            backgrounds,
            store_images,
            ):
    return SequenceWiseH36M(data_path=data_path,
                        subject_list=subject_list,
                        load_from_zarr=load_from_zarr,
                        load_seq_datalist = load_seq_datalist,
                        img_size=img_size,
                        mask=mask,
                        fitting_thr=fitting_thr,
                        smpl=smpl,
                        len_chunks=len_chunks,
                        backgrounds=backgrounds,
                        store_images=store_images
                        )
   
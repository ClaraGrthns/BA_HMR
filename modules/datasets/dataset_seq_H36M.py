import os
import os.path as osp
from re import sub
import time
import pickle as pkl
import numpy as np
from PIL import Image
import torch
from torch._C import float32
import zarr
import copy


from ..utils.image_utils import to_tensor, transform, transform_visualize, crop_box
from ..utils.data_utils_h36m import get_data_chunk_list_h36m, get_background
from ..utils.geometry import get_smpl_coord


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
        self.fitting_thr = fitting_thr  # milimeter --> Threshhold joints from smpl mesh to h36m gt
        self.smpl = smpl
        self.mask = mask
        self.store_images = store_images
        self.len_chunks = len_chunks

        self.subject_list= subject_list
        chunks = []
        seq_datalist = []
        
        for idx, subj_list in enumerate(load_seq_datalist):
            sub_chunks, sub_seq_datalist = get_data_chunk_list_h36m(self.annot_dir,
                    self.img_dir,
                    subject_list=[self.subject_list[idx]],
                    fitting_thr=25,
                    len_chunks=8,
                    load_seq_datalist=subj_list,
                    load_datalist=load_datalist,
                    store_as_pkl=False,
                    )
            chunks.append(sub_chunks)
            seq_datalist.append(sub_seq_datalist)
        self.chunks = [chunk for sub_chunks in chunks for chunk in sub_chunks]
        self.seq_datalist = [seq for sub_seq_datalist in seq_datalist for seq in sub_seq_datalist]

        if self.load_from_zarr is not None:
            self.imgs = {subj: torch.from_numpy(zarr.load(zarr_path)) for subj, zarr_path in zip(self.subject_list, self.load_from_zarr) }
             ### Load array into memory

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
        img_paths = [osp.join(self.img_dir, item['img_name']) for item in chunk]
        zarr_ids = [item['zarr_id'] for item in chunk]
        if self.load_from_zarr is not None:
            subject = chunk[0]['subject']
            zarr_ids = [item['zarr_id'] for item in chunk]
            imgs_tensor = self.imgs[subject][zarr_ids]
        elif self.store_images and torch.all(self.img_cache_indicator[zarr_ids]):
            img_tensor = self.img_cache[zarr_ids]
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
        data = {}
        data['img_path'] = img_paths
        data['img'] = imgs_tensor
        # To-Do:  
        data['cam_pose'] = item['cam_pose']
        data['cam_intr'] = item['cam_intr']
 
        beta = item['betas']
        pose = item['poses']
        trans = item['trans']
        vertices, trans = get_smpl_coord(pose=pose, beta=beta, trans=trans, root_idx=0, cam_pose=data['cam_pose'], smpl=self.smpl)
        data['betas'] = beta
        data['poses'] = pose
        data['trans'] = trans
        data['vertices'] = vertices
        return data

        
        
def get_data(data_path,
            subject_list,
            load_from_zarr,
            load_datalist,
            img_size,
            mask,
            smpl,
            backgrounds,
            store_images,
            ):
    return ImageWiseH36M(data_path=data_path,
                        subject_list=subject_list,
                        load_from_zarr=load_from_zarr,
                        load_datalist=load_datalist,
                        img_size=img_size,
                        mask=mask,
                        smpl=smpl,
                        backgrounds=backgrounds,
                        store_images=store_images
                        )
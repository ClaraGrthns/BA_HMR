import torch
import os.path as osp
import os
import zarr
from PIL import Image
import numpy as np
import copy
import os, psutil

from ..utils.data_utils_h36m import get_data_list_h36m, get_background
from ..utils.image_utils import to_tensor, transform
from ..smpl_model.smpl_pose2mesh import SMPL
from ..utils.geometry import get_smpl_coord

class ImageWiseH36M(torch.utils.data.Dataset):
    def __init__(self,
                data_path:str='../H36M',
                subject_list:list=[],
                smpl=None,
                load_from_zarr:list=None,
                load_datalist:str=None,
                img_size:int=224,
                mask:bool = False,
                fitting_thr:int=25,
                store_images:bool=False,
                backgrounds:list=None,
                ):
        super(ImageWiseH36M, self).__init__()
        self.img_dir = osp.join(data_path, 'images')
        self.annot_dir = osp.join(data_path,'annotations')
        self.load_from_zarr = load_from_zarr
        self.backgrounds = backgrounds
        self.fitting_thr = fitting_thr  # milimeter --> Threshhold joints from smpl mesh to h36m gt
        self.smpl = smpl
        self.mask = mask
        self.store_images = False
        self.subject_list = subject_list
        print('datasets initialized')
        self.datalist = get_data_list_h36m(annot_dir=self.annot_dir,
                                        subject_list=self.subject_list,
                                        fitting_thr=self.fitting_thr,
                                        load_from_pkl=load_datalist,
                                        store_as_pkl=False,
                                        out_dir=None,) 
        process = psutil.Process(os.getpid())
        print('datalist h36m, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')

        if self.load_from_zarr is not None:
            imgs = {}
            for subj, zarr_path in zip(self.subject_list, self.load_from_zarr):
                print(subj, zarr_path)
                process = psutil.Process(os.getpid())
                print('data h36m, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')
                imgs[subj]= torch.from_numpy(zarr.load(zarr_path))
                print(imgs[subj].device)
            self.imgs = imgs
            #self.imgs = {torch.from_numpy(zarr.load(zarr_path)) for subj, zarr_path in zip(self.subject_list, self.load_from_zarr) }
            ### Load array into memory
        else: 
            self.img_size = img_size
            if self.store_images: 
                self.img_cache_indicator = torch.zeros(self.__len__(), dtype=torch.bool)
                self.img_cache = torch.empty(self.__len__(), 3, img_size, img_size, dtype=torch.float32)
       
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        item = copy.deepcopy(self.datalist[index])
        img_path = osp.join(self.img_dir, item['img_name'])
        if self.load_from_zarr is not None:
            subject = item['subject']
            zarr_id = item['zarr_id']
            img_tensor = self.imgs[subject][zarr_id]
        elif self.store_images and self.img_cache_indicator[index]:
            img_tensor = self.img_cache[index]
        else:
            img = np.array(Image.open(img_path))
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
            if self.store_images:
                self.img_cache[index] = img_tensor
                self.img_cache_indicator[index] = True
        data = {}
        data['img_path'] = img_path
        data['img'] = img_tensor.to(self.device)
        data['cam_pose'] = item['cam_pose'].to(self.device)
        data['cam_intr'] = item['cam_intr']
        beta = item['betas'].to(self.device)
        pose = item['poses'].to(self.device)
        trans = item['trans'].to(self.device)
        vertices, trans, pose = get_smpl_coord(pose=pose, beta=beta, trans=trans, root_idx=0, cam_pose=data['cam_pose'], smpl=self.smpl)
        data['betas'] = beta
        data['poses'] = pose
        data['trans'] = trans
        data['vertices'] = vertices
        #data['joints_3d'] = item['joints_3d']/1000
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
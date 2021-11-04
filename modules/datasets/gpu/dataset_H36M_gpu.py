import torch
import os.path as osp
import zarr
from PIL import Image
import numpy as np
import copy
import random

from ...utils.data_utils_h36m import get_data_list_h36m
from ...utils.image_utils import to_tensor, transform, transform_visualize, crop_box, lcc
from ...smpl_model.smpl_pose2mesh import SMPL
from ...utils.geometry import get_smpl_coord

class ImageWiseH36M(torch.utils.data.Dataset):
    def __init__(self,
                data_path:str='../H36M',
                split:str='train',
                smpl=None,
                load_from_zarr:str=None,
                load_datalist:str=None,
                img_size:int=224,
                mask:bool = False,
                fitting_thr:int=25,
                store_images:bool=True,
                backgrounds:list=None,
                ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(ImageWiseH36M, self).__init__()
        self.img_dir = osp.join(data_path, 'images')
        self.annot_dir = osp.join(data_path,'annotations')
        self.load_from_zarr = load_from_zarr
        self.backgrounds = backgrounds
        self.fitting_thr = fitting_thr  # milimeter --> Threshhold joints from smpl mesh to h36m gt
        self.smpl = smpl
        self.mask = mask
        self.store_images = store_images

        if split == 'full':
            self.subject_list= [1, 5, 6, 7, 8, 9, 11]
        elif split == 'train':
            self.subject_list = [1, 5, 6, 7, 8]
        elif split == 'validation': 
            self.subject_list = [9,11]
        else:
            self.subject_list=[1]
            
        self.datalist = get_data_list_h36m(annot_dir=self.annot_dir,
                                            subject_list=self.subject_list,
                                            fitting_thr=self.fitting_thr,
                                            load_from_pkl=load_datalist,
                                            store_as_pkl=False,
                                            out_dir=None,)
        a = self.load_from_zarr is not None

        if self.load_from_zarr is not None:
            self.imgs = torch.from_numpy(zarr.load(self.load_from_zarr)) ### Load array into memory
        elif self.store_images:
            self.img_size = img_size
            self.img_cache_indicator = torch.zeros(self.__len__(), dtype=torch.bool)
            self.img_cache = torch.empty(self.__len__(), 3, img_size, img_size, dtype=torch.float32)
  
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = copy.deepcopy(self.datalist[index])
        img_path = osp.join(self.img_dir, data['img_name'])
        img = np.array(Image.open(img_path))            
        if self.load_from_zarr is not None:
            img_tensor = self.imgs[index].to(self.device)
        elif self.store_images and self.img_cache_indicator[index]:
            img_tensor = self.img_cache[index].to(self.device)
        else:
            img = np.array(Image.open(img_path))
            if self.mask:
                sub_dir, img_name = osp.split(data['img_name'])
                mask_name = img_name.split('.')[-2]+'_mask.jpg'
                mask_path = osp.join(self.img_dir, sub_dir, mask_name)
                mask = np.round(np.array(Image.open(mask_path))/255-1)
                ## Cut out mask and use different backgrounds
                img[np.nonzero(mask)] = self.get_background(img.shape)[np.nonzero(mask)]
            x_min, y_min, x_max, y_max = data['bbox']
            img = img[y_min:y_max, x_min:x_max]
            img_tensor = to_tensor(img).to(self.device)
            img_tensor = transform(img_tensor, img_size=self.img_size)
            if self.store_images:
                self.img_cache[index] = img_tensor.to('cpu')
                self.img_cache_indicator[index] = True
        beta = data['betas']
        pose = data['poses']
        trans = data['trans']
        cam_pose = data['cam_pose']
        vertices, trans = get_smpl_coord(pose=pose, beta=beta, trans=trans, root_idx=0, cam_pose=cam_pose, smpl=self.smpl)
        data['vertices'] = vertices
        data['trans'] = trans
        data['img_path'] = img_path
        data['img'] = img_tensor
        return data
    def get_background(self, img_shape):
        height, width,_ = img_shape
        mask = random.choice(self.backgrounds)[:height, :width]
        return mask
        
        
def get_data(data_path,
            split,
            load_from_zarr,
            load_datalist,
            img_size,
            mask,
            smpl,
            backgrounds,
            store_images,
            ):
    return ImageWiseH36M(data_path=data_path,
                        split=split,
                        load_from_zarr=load_from_zarr,
                        load_datalist=load_datalist,
                        img_size=img_size,
                        mask=mask,
                        smpl=smpl,
                        backgrounds=backgrounds,
                        store_images=store_images,
                        a)
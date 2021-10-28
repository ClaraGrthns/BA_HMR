import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import time
import json
from PIL import Image
import torch
import zarr
from .image_utils import to_tensor, transform, transform_visualize, crop_box
from .data_utils_3dpw import get_ids_imgspaths_seq
from .data_utils_h36m import get_data_list_h36m, get_backgrounds_from_folder, get_background

def save_img_zarr_3dpw(data_path:str,
                  zarr_path:str,
                  split:str='train',
                  num_required_keypoints:int=0,
                  num_chunks=None,
                  img_size=224,
                  load_from_pkl=None,
                  padding=False,):  
    id_img_list = get_ids_imgspaths_seq(data_path=data_path, 
                                    split=split,
                                    num_required_keypoints=num_required_keypoints,
                                    load_from_pkl=load_from_pkl,
                                    store_as_pkl=False)
    image_paths = id_img_list['image_paths']
    person_ids = id_img_list['person_ids']
    sequences = id_img_list['sequences']

    ## create imgsx3xHxW zarr array
    if num_chunks is None:
        num_chunks= len(image_paths)//10                 
    img_zarr = zarr.open(zarr_path, mode='w', shape=(len(image_paths), 3, img_size, img_size), chunks=(num_chunks, None), dtype='float32')

    for index, img_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        ## for each img: save tranformed img in zarr array
        _, img_name = os.path.split(img_path)
        img = np.array(Image.open(img_path))
        img_tensor = to_tensor(img)

        seq_name = img_path.split('/')[-2]  
        index_seq = int((img_name.split('.')[0]).split('_')[1])
        person_id = person_ids[index]
        
        seq_pose = sequences[seq_name]['poses2d'][person_id][index_seq]  
        pose2d =  torch.tensor(seq_pose, dtype=torch.float32)
        img_tensor, _ = crop_box(img_tensor=img_tensor, pose2d=pose2d, padding=padding)
        img_tensor = transform(img_tensor, img_size)
        img_zarr[index] = img_tensor

def save_img_zarr_h36m(data_path:str,
                    zarr_path:str,
                    subject_list:list,
                    img_size:int,
                    fitting_thr:int,
                    load_from_pkl:str,
                    num_chunks:int=None,
                    ):
    annot_dir = osp.join(data_path,'annotations')
    img_dir = osp.join(data_path, 'images')
    datalist = get_data_list_h36m(annot_dir=annot_dir, 
                                subject_list=subject_list, 
                                fitting_thr=fitting_thr,
                                load_from_pkl=load_from_pkl,
                                store_as_pkl=False)

    backgrounds = get_backgrounds_from_folder(osp.join(data_path, 'backgrounds'))

    if num_chunks is None:
        num_chunks= len(datalist)//10

    img_zarr = zarr.open(zarr_path, mode='w', shape=(len(datalist), 3, img_size, img_size), chunks=(num_chunks, None), dtype='float32')
    
    for data in datalist:
        #open image
        img_path = osp.join(img_dir, data['img_name'])
        zarr_id = data['zarr_id']
        img = np.array(Image.open(img_path))
        sub_dir, img_name = osp.split(data['img_name'])
        #open mask and apply mask
        mask_name = img_name.split('.')[-2]+'_mask.jpg'
        mask_path = osp.join(img_dir, sub_dir, mask_name)
        mask = np.round(np.array(Image.open(mask_path))/255-1)
        img[np.nonzero(mask)] = get_background(img_shape=img.shape, backgrounds=backgrounds)[np.nonzero(mask)]
        #apply bbox
        x_min, y_min, x_max, y_max = data['bbox']
        img = img[y_min:y_max, x_min:x_max]
        #transform img
        img_tensor = to_tensor(img)
        img_tensor = transform(img_tensor, img_size=img_size)
        img_zarr[zarr_id] = img_tensor
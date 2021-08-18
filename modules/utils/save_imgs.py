import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import time
import pickle as pkl
from PIL import Image
import torch
import zarr
from .image_utils import to_tensor, transform, transform_visualize, crop_box
from .data_utils import get_ids_imgspaths_seq

def save_img_zarr(root_path:str,
                  zarr_path:str,
                  split:str='train',
                  num_required_keypoints:int=0,
                  num_chunks=None,
                  img_size=224,
                  save_id_paths=False,
                  load_from_pkl=None,
                  padding=False,):
    id_img_list = get_ids_imgspaths_seq(root_path=root_path, 
                                    split=split,
                                    load_from_pkl=load_from_pkl,
                                    store_as_pkl=False)
    image_paths = id_img_list['image_paths']
    person_ids = id_img_list['person_ids']
    sequences = id_img_list['sequences']

    ## create imgsx3x244x244 zarr array
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
        img_tensor = transform(img_tensor,img_size)
        img_zarr[index] = img_tensor

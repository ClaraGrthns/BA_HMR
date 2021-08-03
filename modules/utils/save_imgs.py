import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import time
import pickle as pkl
import numpy as np
from PIL import Image
import torch
import zarr

from .image_utils import to_tensor, transform, transform_visualize, crop_box
from .data_utils import get_relevant_keypoints

def save_img_zarr(root_path:str,
                  zarr_path:str=None,
                  split:str='train',
                  num_required_keypoints:int=0,
                  num_chunks=None,
                  img_size=224,
                  save_id_paths=False,
                  out_path=None):
                    
    id_img_list = get_ids_imgspaths(root_path=root_path, 
                                    out_path=out_path,
                                    split=split,
                                    num_required_keypoints=num_required_keypoints,
                                    save_ids_paths=save_id_paths)
    image_paths = id_img_list['image_paths']
    person_ids = id_img_list['person_ids']
    seq_poses = id_img_list['seq_poses']

    ## create imgsx3x244x244 zarr array
    if zarr_path is None:
        zarr_path = osp.join('../../', 'data', f'imgs_3dpw_min{num_required_keypoints}_kps_{split}.zarr')
        print(zarr_path)
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
        
        seq_pose = seq_poses[seq_name][person_id][index_seq]  
        pose2d =  torch.tensor(seq_pose, dtype=torch.float32)
        img_tensor, _ = crop_box(img_tensor=img_tensor, pose2d=pose2d)
        img_tensor = transform(img_tensor,img_size)

        img_zarr[index] = img_tensor

def get_ids_imgspaths(root_path:str="../3DPW",
                      out_path:str="./data/ids_imgpaths",
                      split:str='train',
                      num_required_keypoints:int=8,
                      save_ids_paths=True,):
    
    sequence_path = osp.join(root_path, 'sequenceFiles', split)
    seq_names = [seq_name.split('.pkl')[0] for seq_name in sorted(os.listdir(sequence_path))]
    num_required_keypoints = 8
    image_paths = []
    person_ids = []
    seq_poses = {}
    for seq_name in seq_names:
        ## loop through all sequences and filter out those where kp are missing
        img_dir = osp.join(root_path, 'imageFiles', seq_name)

        seq_file_name = os.path.join(sequence_path, f'{seq_name}.pkl')
        with open(seq_file_name, 'rb') as f:
            seq = pkl.load(f, encoding='latin1')
        seq_poses[seq_name] = seq['poses2d']
        num_people = len(seq['poses'])

        for img_idx, img_name in enumerate(sorted(os.listdir(img_dir))):
            ## for each person: save img_path 
            image_path = osp.join(img_dir,img_name)
            for person_id in range(num_people):
                pose2d = seq['poses2d'][person_id][img_idx]
                relevant_poses2d = get_relevant_keypoints(pose2d)
                if len(relevant_poses2d) >= num_required_keypoints:
                    image_paths.append(image_path)
                    person_ids.append(person_id)

    id_img_list = {"person_ids": person_ids, "image_paths": image_paths}
    if save_ids_paths:
        with open(osp.join(out_path, f'ids_paths_{split}_min{num_required_keypoints}_kps_ids_img_list.pickle'), 'wb') as fp:
            pkl.dump(id_img_list, fp)
    id_img_list['seq_poses'] = seq_poses
    return id_img_list
import os
import os.path as osp
import time
import pickle as pkl
import numpy as np
from PIL import Image
import torch
import zarr

from .utils.image_utils import to_tensor, transform, transform_visualize, crop_box
from .utils.data_utils import get_chunks_seq, rand_partition


class SequenceWise3DPW(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path:str,
        split:str='train',
        num_required_keypoints:int=0,
        len_chunks:int=8,
        store_sequences=True,
        store_images=True,
        load_from_zarr:str=None,
        img_size=224,
        load_chunks_seq=None,
    ):
        super(SequenceWise3DPW, self).__init__()
        self.split = split
        self.num_required_keypoints = num_required_keypoints
        self.store_sequences = store_sequences
        self.store_images = store_images
        self.load_from_zarr = load_from_zarr
        self.len_chunks = len_chunks
 
        chunks, img_list, img_paths, sequences = get_chunks_seq(data_path=data_path,
                                                                     split=self.split,
                                                                     load_from_pkl=load_chunks_seq,
                                                                     len_chunks=self.len_chunks,
                                                                     num_required_keypoints=self.num_required_keypoints,
                                                                    )
        self.seq_chunks = chunks
        self.img_paths = img_paths
        self.img_list = img_list
        
        if self.store_sequences:
            self.sequences = sequences
        else: 
            self.sequence_path = osp.join(data_path, 'sequenceFiles', split)

        if self.load_from_zarr is not None:
            self.imgs = torch.from_numpy(zarr.load(self.load_from_zarr)) ### Load array into memory
        elif self.store_images:
            self.img_size = img_size
            self.img_cache_indicator = torch.zeros(len(self.img_paths), dtype=torch.bool)
            self.img_cache = torch.empty(len(self.img_paths), 3, img_size, img_size, dtype=torch.float32)
        self.timers = {
            'load_sequence': 0,
            'load_image': 0,
            'out': 0,
        }
        
    def __len__(self):
        return len(self.seq_chunks)
        
    def __getitem__(self, index):
        t_start = time.time()

        # load sequence
        seq_chunk = self.seq_chunks[index]
        img_paths = [img_path[0] for img_path in seq_chunk]
        seq_indices = [int(os.path.split(img_path)[1].split('.')[0].split('_')[1]) for img_path in img_paths]
        seq_name = img_paths[0].split('/')[-2]
        person_id = seq_chunk[0][1]
        
        if self.store_sequences:
            seq = self.sequences[seq_name]
        else:
            seq_file_name = os.path.join(self.sequence_path, f'{seq_name}.pkl')
            with open(seq_file_name, 'rb') as f:
                seq = pkl.load(f, encoding='latin1')
        
        poses2d = torch.tensor(seq['poses2d'][person_id][seq_indices], dtype=torch.float32)    
        poses3d = torch.tensor(seq['jointPositions'][person_id][seq_indices], dtype=torch.float32) 
        poses3d = poses3d.view(-1, 24,3)
        
        t_load_sequence = time.time()
    
        img_indices = [chunk[-1] for chunk in seq_chunk]
        if self.load_from_zarr is not None:
            imgs_tensor = self.imgs[img_indices] ### Read array from memory
        elif self.store_images and torch.all(self.img_cache_indicator[img_indices]):
            imgs_tensor = self.img_cache[img_indices]
        else:
            imgs_tensor = torch.zeros(len(img_paths), 3, self.img_size, self.img_size)
            for idx, img_path in enumerate(img_paths):
                img = np.array(Image.open(img_path))
                img_tensor = to_tensor(img)
                img_tensor, _ = crop_box(img_tensor=img_tensor, pose2d=poses2d[idx])
                img_tensor = transform(img_tensor, img_size=self.img_size)
                imgs_tensor[idx] = img_tensor
                if self.store_images:
                    self.img_cache[img_indices[idx]] = img_tensor
                    self.img_cache_indicator[img_indices[idx]] = True
                
        t_load_image = time.time()
        
        data = {}
        data['img_path'] = img_paths
        data['img'] = imgs_tensor
        data['betas'] = torch.tensor(seq['betas'][person_id][None,:10], dtype=torch.float32)
        data['cam_pose'] = torch.tensor(seq['cam_poses'][seq_indices], dtype=torch.float32)    
        data['poses'] = torch.tensor(seq['poses'][person_id][None, seq_indices], dtype=torch.float32) 
        data['poses2d'] = poses2d 
        data['poses3d'] = poses3d
        data['cam_pose'] = torch.tensor(seq['cam_poses'][seq_indices], dtype=torch.float32)  
        data['cam_intr'] = torch.tensor(seq['cam_intrinsics'], dtype=torch.float32)
        data['trans'] = torch.tensor(seq['trans'][person_id][None, seq_indices], dtype=torch.float32)
        
        t_out = time.time()
        
        self.timers['load_sequence'] += t_load_sequence - t_start
        self.timers['load_image'] += t_load_image - t_load_sequence
        self.timers['out'] += t_out - t_load_image

        return data         
    def set_chunks(self):
        self.seq_chunks = get_chunks_seq(img_list=self.img_list)
    
def get_train_val_data(data_path:str,
                       num_required_keypoints:int=0,
                       len_chunks:int=8,
                       store_sequences:bool=True,
                       store_images:bool=True,
                       load_from_zarr_trn:str=None,
                       load_from_zarr_val:str=None,
                       img_size:int=224,
                       load_chunks_seq_val:str=None,
                       load_chunks_seq_trn:str=None):

    train_data = SequenceWise3DPW(data_path=data_path,
                                  num_required_keypoints=num_required_keypoints,
                                  len_chunks = len_chunks,
                                  store_sequences=store_sequences,
                                  store_images=store_images,
                                  img_size=img_size,
                                  load_from_zarr=load_from_zarr_trn,
                                  load_chunks_seq=load_chunks_seq_trn,
                               )

    val_data = SequenceWise3DPW(data_path=data_path, 
                                split = 'validation',
                                num_required_keypoints=num_required_keypoints,
                                len_chunks = len_chunks,
                                store_sequences=store_sequences,
                                store_images=store_images,
                                img_size=img_size,
                                load_from_zarr=load_from_zarr_val,
                                load_chunks_seq=load_chunks_seq_val,
                             )
    
    return train_data, val_data
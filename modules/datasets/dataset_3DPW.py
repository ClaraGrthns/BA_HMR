import os
import os.path as osp
import time
import pickle as pkl
import numpy as np
from PIL import Image
import torch
import zarr
import copy

from ..utils.image_utils import to_tensor, transform, transform_visualize, crop_box
from ..utils.data_utils_3dpw import get_ids_imgspaths_seq 
from ..utils.geometry import get_smpl_coord

class ImageWise3DPW(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path:str,
        split:str = 'train',
        num_required_keypoints:int = 0,
        smpl=None,
        store_sequences=True,
        store_images=True,
        load_from_zarr:str=None,
        img_size=224,
        load_ids_imgpaths_seq=None,
    ):
        super(ImageWise3DPW, self).__init__()
        self.split = split
        self.num_required_keypoints = num_required_keypoints
        self.store_sequences = store_sequences
        self.store_images = store_images
        self.load_from_zarr = load_from_zarr
        self.smpl = smpl
 
        ids_imgpaths_seq = get_ids_imgspaths_seq(data_path=data_path,
                                                split=self.split,
                                                load_from_pkl=load_ids_imgpaths_seq,
                                                num_required_keypoints=self.num_required_keypoints,
                                                store_as_pkl=False)
        self.image_paths = ids_imgpaths_seq['image_paths']
        self.person_ids = ids_imgpaths_seq['person_ids']
        if self.store_sequences:
            self.sequences = ids_imgpaths_seq['sequences']
        else: 
            self.sequence_path = osp.join(data_path, 'sequenceFiles', split)
        if self.load_from_zarr is not None:
            self.imgs = torch.from_numpy(zarr.load(self.load_from_zarr)) ### Load array into memory
        else:
            self.img_size = img_size
            if self.store_images:
                self.img_cache_indicator = torch.zeros(self.__len__(), dtype=torch.bool)
                self.img_cache = torch.empty(self.__len__(), 3, img_size, img_size, dtype=torch.float32)
        self.timers = {
            'load_sequence': 0,
            'load_image': 0,
            'out': 0,
        }
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        t_start = time.time()

        # load sequence
        img_path = self.image_paths[index]
        _, img_name = os.path.split(img_path)
        seq_name = img_path.split('/')[-2]
        
        if self.store_sequences:
            seq = self.sequences[seq_name]
        else:
            seq_file_name = os.path.join(self.sequence_path, f'{seq_name}.pkl')
            with open(seq_file_name, 'rb') as f:
                seq = pkl.load(f, encoding='latin1')
            
        index_seq = int((img_name.split('.')[0]).split('_')[1])
        person_id = self.person_ids[index]
        poses2d = torch.tensor(seq['poses2d'][person_id][index_seq], dtype=torch.float32)    
        #poses3d = torch.tensor(seq['jointPositions'][person_id][index_seq], dtype=torch.float32) 
        #poses3d = poses3d.view(-1, 24,3)
        
        t_load_sequence = time.time()
    
        # Resize Image to img_sizeximg_size format with padding (hrnet: 256x256)
        if self.load_from_zarr is not None:
            img_tensor = self.imgs[index] ### Read array from memory
        elif self.store_images and self.img_cache_indicator[index]:
            img_tensor = self.img_cache[index]
        else:
            img = np.array(Image.open(img_path))
            img_tensor = to_tensor(img)
            img_tensor, _ = crop_box(img_tensor=img_tensor, pose2d=poses2d)
            img_tensor = transform(img_tensor, img_size=self.img_size)
            if self.store_images:
                self.img_cache[index] = img_tensor
                self.img_cache_indicator[index] = True
                
        t_load_image = time.time()
        
        data = {}
        data['img_path'] = img_path
        data['img'] = img_tensor
        data['cam_pose'] = torch.FloatTensor(seq['cam_poses'][index_seq]) 
        data['cam_intr'] = torch.tensor(seq['cam_intrinsics'])
   
        beta = copy.deepcopy(torch.FloatTensor(seq['betas'][person_id][:10]))
        pose = copy.deepcopy(torch.FloatTensor(seq['poses'][person_id][index_seq]))
        trans = copy.deepcopy(torch.FloatTensor(seq['trans'][person_id][index_seq]))
        vertices, trans = get_smpl_coord(pose=pose, beta=beta, trans=trans, root_idx=0, cam_pose=data['cam_pose'], smpl=self.smpl)
        data['betas'] = beta
        data['poses'] = pose
        data['trans'] = trans
        data['vertices'] = vertices

        #data['poses2d'] = poses2d 
        #data['poses3d'] = poses3d
        
        t_out = time.time()
        self.timers['load_sequence'] += t_load_sequence - t_start
        self.timers['load_image'] += t_load_image - t_load_sequence
        self.timers['out'] += t_out - t_load_image
        return data
    
def get_train_val_data(data_path, 
                       num_required_keypoints, 
                       store_sequences,
                       store_images,
                       load_from_zarr_trn,
                       load_from_zarr_val,
                       img_size,
                       load_ids_imgpaths_seq_trn,
                       load_ids_imgpaths_seq_val,
                       smpl,
                      ): 
    
    train_data = ImageWise3DPW(data_path=data_path,
                               num_required_keypoints=num_required_keypoints,
                               store_sequences=store_sequences,
                               store_images=store_images,
                               load_from_zarr=load_from_zarr_trn,
                               img_size=img_size,
                               load_ids_imgpaths_seq=load_ids_imgpaths_seq_trn,
                               smpl=smpl,
                               )

    val_data = ImageWise3DPW(data_path=data_path, 
                             split = 'validation',
                             num_required_keypoints=num_required_keypoints,
                             store_sequences=store_sequences,
                             store_images=store_images,
                             load_from_zarr=load_from_zarr_val,
                             img_size=img_size,
                             load_ids_imgpaths_seq=load_ids_imgpaths_seq_val,
                             smpl=smpl,
                             )
    
    return train_data, val_data

def get_data(data_path,
            split,
            num_required_keypoints, 
            store_sequences,
            store_images,
            load_from_zarr,
            img_size,
            load_ids_imgpaths_seq,
            smpl,
            ):
    return ImageWise3DPW(data_path=data_path,
                            split = split,
                            num_required_keypoints=num_required_keypoints,
                            store_sequences=store_sequences,
                            store_images=store_images,
                            load_from_zarr=load_from_zarr,
                            img_size=img_size,
                            load_ids_imgpaths_seq=load_ids_imgpaths_seq,
                            smpl=smpl,
                            )

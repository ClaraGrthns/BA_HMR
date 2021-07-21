import os
import os.path as osp
import time
import pickle as pkl
import numpy as np
from PIL import Image
import torch

from .utils.image_utils import to_tensor, transform, transform_visualize, crop_box
from .utils.data_utils import get_relevant_keypoints


class ImageWise3DPW(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path:str,
        split:str = 'train',
        num_required_keypoints:int = 0,
        store_sequences=True,
        store_images=True,
        load_from_zarr=False,
    ):
        super(ImageWise3DPW, self).__init__()
        self.sequence_path = osp.join(root_path, 'sequenceFiles', split)
        self.seq_names = [seq_name.split('.pkl')[0] for seq_name in sorted(os.listdir(self.sequence_path))]
        self.split = split
        self.num_required_keypoints = num_required_keypoints
        self.store_sequences = store_sequences
        self.sequences = {}
        self.store_images = store_images

        person_ids = []
        image_paths = []
        for seq_name in self.seq_names:
            img_dir = osp.join(root_path, 'imageFiles', seq_name)
            
            seq_file_name = os.path.join(self.sequence_path, f'{seq_name}.pkl')
            with open(seq_file_name, 'rb') as f:
                seq = pkl.load(f, encoding='latin1')
            if store_sequences:
                self.sequences[seq_name] = seq
                
            num_people = len(seq['poses'])
            
            for img_idx, img_name in enumerate(sorted(os.listdir(img_dir))):
                image_path = osp.join(img_dir,img_name)
                for person_id in range(num_people):
                    pose2d = seq['poses2d'][person_id][img_idx]
                    relevant_poses2d = get_relevant_keypoints(pose2d)
                    if len(relevant_poses2d) >= self.num_required_keypoints:
                        image_paths.append(image_path)
                        person_ids.append(person_id)
            
        self.image_paths = image_paths
        self.person_ids = person_ids
        
        
        if self.store_images:
            self.img_cache_indicator = torch.zeros(self.__len__(), dtype=torch.bool)
            self.img_cache = torch.empty(self.__len__(), 3, 224, 224, dtype=torch.float32)
        
        self.timers = {
            'load_image': 0,
            'load_sequence': 0,
            #'image_to_tensor': 0,
            #'crop_box': 0,
            'transform_image': 0,
            'out': 0,
        }
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        # TODO use zarr to store image-crops with the right resulution on file system.
        # First use the actual implementation of this class with store_images = True to generate a tensor of cropped images (self.img_cache).
        # Then convert this tensor to a np.ndarray and use zarr to store it in the local file system.
        # You should end up with a .zarr file containing all the image crops at their respective index.
        #
        # Adapt this class (or write a new one) to load images from the zarr file.

        t_start = time.time()
        
        # load image
        # img = np.array(Image.open(img_path))
        t_load_image = time.time()

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
        
        poses3d = torch.tensor(seq['jointPositions'][person_id][index_seq], dtype=torch.float32) 
        poses3d = poses3d.view(-1, 24,3)
        
        t_load_sequence = time.time()
    
        # Resize Image to 224x224 format with padding
        if self.store_images and self.img_cache_indicator[index]:
            img_tensor = self.img_cache[index]
        else:
            img = np.array(Image.open(img_path))
            img_tensor = to_tensor(img)
        
            img_tensor, _ = crop_box(img_tensor=img_tensor, pose2d=poses2d)
        
            img_tensor = transform(img_tensor)
            
            if self.store_images:
                self.img_cache[index] = img_tensor
                self.img_cache_indicator[index] = True
                
        t_preprocess_image = time.time()
        
        data = {}
        data['img_path'] = img_path
        data['img'] = img_tensor
        data['betas'] = torch.tensor(seq['betas'][person_id][:10], dtype=torch.float32)
        data['cam_pose'] = torch.tensor(seq['cam_poses'][index_seq], dtype=torch.float32)    
        data['poses'] = torch.tensor(seq['poses'][person_id][index_seq], dtype=torch.float32) 
        data['poses2d'] = poses2d 
        data['poses3d'] = poses3d
        data['cam_pose'] = torch.tensor(seq['cam_poses'][index_seq], dtype=torch.float32)  
        data['cam_intr'] = torch.tensor(seq['cam_intrinsics'], dtype=torch.float32)
        data['trans'] = torch.tensor(seq['trans'][person_id][index_seq], dtype=torch.float32)
        
        t_out = time.time()
        
        self.timers['load_image'] += t_load_image - t_start
        self.timers['load_sequence'] += t_load_sequence - t_load_image
        #self.timers['image_to_tensor'] += t_image_to_tensor - t_load_sequence
        #self.timers['crop_box'] += t_crop_box - t_image_to_tensor
        self.timers['transform_image'] += t_preprocess_image - t_load_sequence
        
        self.timers['out'] += t_out - t_preprocess_image

        return data
    
def get_train_val_data(data_path, num_required_keypoints=8): 
    train_data = ImageWise3DPW(
            root_path=data_path,
            num_required_keypoints=num_required_keypoints,
        )

    val_data = ImageWise3DPW(
        root_path=data_path, 
        split = 'validation')
    return train_data, val_data
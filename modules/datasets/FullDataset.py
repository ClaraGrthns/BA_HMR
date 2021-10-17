import torch
import numpy as np

from .dataset_3DPW import get_data as get_data_3dpw
from .dataset_H36M import get_data as get_data_h36m
from modules.smpl_model.smpl_pose2mesh import SMPL

class ImgWiseFullDataset(torch.utils.data.Dataset):
    """Combination of Human 3.6M and 3DPW Dataset"""
    def __init__(self, *datasets):
        super(ImgWiseFullDataset, self).__init__()
        self.datasets = [dataset for dataset in datasets if len(dataset)!= 0]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.lengths.insert(0,0)
        self.length = sum(self.lengths)
        self.partition = np.array(self.lengths).cumsum()

    def __getitem__(self, i):
        part = np.argmax(self.partition > i)
        return self.datasets[part-1][i - self.partition[part-1]]
                            
    def __len__(self):
        return self.length

def get_full_train_val_data(
        dataset:str='full',
        data_path_3dpw:str='../3DPW',
        num_required_keypoints:int = 0,
        store_sequences=True,
        store_images=True,
        load_from_zarr_3dpw_trn:str=None,
        load_from_zarr_3dpw_val:str=None,
        img_size=224,
        load_ids_imgpaths_seq_trn=None,
        load_ids_imgpaths_seq_val=None,
        data_path_h36m:str='../H36M',
        load_from_zarr_h36m_trn:str=None,
        load_from_zarr_h36m_val:str=None,
        load_datalist_trn:str=None,
        load_datalist_val:str=None,
        backgrounds:list=None,
        mask:bool=True,
        val_on_h36m:bool=False,
    ): 
    smpl = SMPL()
    smpl.layer['neutral'].th_shapedirs= smpl.layer['neutral'].th_shapedirs[:,:,:10]
    smpl.layer['neutral'].th_betas= smpl.layer['neutral'].th_betas[:,:10]

    train_data_3dpw = val_data_3dpw = train_data_h36m = val_data_h36m = []
    if dataset == 'full' or dataset == '3dpw':
        train_data_3dpw = get_data_3dpw(data_path=data_path_3dpw, 
                                        split='train',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images,
                                        load_from_zarr=load_from_zarr_3dpw_trn,
                                        img_size=img_size,
                                        load_ids_imgpaths_seq=load_ids_imgpaths_seq_trn,
                                        smpl=smpl.layer['neutral'],)
                                        
        val_data_3dpw = get_data_3dpw(data_path=data_path_3dpw, 
                                        split='validation',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images,
                                        load_from_zarr=load_from_zarr_3dpw_val,
                                        img_size=img_size,
                                        load_ids_imgpaths_seq=load_ids_imgpaths_seq_val,
                                        smpl=smpl.layer['neutral'],)
    if dataset == 'full' or dataset == 'h36m':
        train_data_h36m = get_data_h36m(data_path=data_path_h36m,
                                split='train',
                                load_from_zarr=load_from_zarr_h36m_trn,
                                load_datalist=load_datalist_trn,
                                img_size=img_size,
                                mask=mask,
                                backgrounds=backgrounds,
                                smpl=smpl.layer['neutral'],
                                )
        if val_on_h36m:
            val_data_h36m = get_data_h36m(data_path=data_path_h36m,
                                    split='validation',
                                    load_from_zarr=load_from_zarr_h36m_val,
                                    load_datalist=load_datalist_val,
                                    img_size=img_size,
                                    mask=mask,
                                    backgrounds=backgrounds,
                                    smpl=smpl.layer['neutral'],)

    train_data = ImgWiseFullDataset(train_data_3dpw, train_data_h36m)
    val_data = ImgWiseFullDataset(val_data_3dpw, val_data_h36m)
    return train_data, val_data
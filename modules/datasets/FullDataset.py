import torch
import numpy as np

from .dataset_3DPW import get_data as get_data_3dpw
from .dataset_3DPW_wo_trans import get_data as get_data_3dpw_wo
from .dataset_H36M import get_data as get_data_h36m
from .dataset_H36M_wo_trans import get_data as get_data_h36m_wo

from .dataset_seq_3DPW import get_data as get_data_3dpw_seq
from .dataset_seq_H36M import get_data as get_data_h36m_seq

from ..smpl_model.smpl_pose2mesh import SMPL
import os, psutil

class FullDataset(torch.utils.data.Dataset):
    """Combination of Human 3.6M and 3DPW Dataset"""
    def __init__(self, *datasets):
        super(FullDataset, self).__init__()
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
        store_images_3dpw=False,
        load_from_zarr_3dpw_trn:str=None,
        load_from_zarr_3dpw_val:str=None,
        img_size=224,
        load_ids_imgpaths_seq_trn=None,
        load_ids_imgpaths_seq_val=None,
        data_path_h36m:str='../H36M',
        store_images_h36m=False,
        load_from_zarr_h36m_trn:list=None,
        load_from_zarr_h36m_val:list=None,
        load_datalist_trn:str=None,
        load_datalist_val:str=None,
        backgrounds:list=None,
        mask:bool=True,
        val_on_h36m:bool=False,
        subject_list_trn:list=[],
        subject_list_val:list=[],
        fitting_thr=25,
    ): 
    print('initialize smpl model')
    smpl = SMPL()
    smpl.layer['neutral'].th_shapedirs = smpl.layer['neutral'].th_shapedirs[:,:,:10]
    smpl.layer['neutral'].th_betas = smpl.layer['neutral'].th_betas[:,:10]
    train_data_3dpw = val_data_3dpw = train_data_h36m = val_data_h36m = []

    if dataset == 'full' or dataset == '3dpw':
        train_data_3dpw = get_data_3dpw(data_path=data_path_3dpw, 
                                        split='train',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images_3dpw,
                                        load_from_zarr=load_from_zarr_3dpw_trn,
                                        img_size=img_size,
                                        load_ids_imgpaths_seq=load_ids_imgpaths_seq_trn,
                                        smpl=smpl.layer['neutral'],)
                                        
        val_data_3dpw = get_data_3dpw(data_path=data_path_3dpw, 
                                        split='validation',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images_3dpw,
                                        load_from_zarr=load_from_zarr_3dpw_val,
                                        img_size=img_size,
                                        load_ids_imgpaths_seq=load_ids_imgpaths_seq_val,
                                        smpl=smpl.layer['neutral'],)
        process = psutil.Process(os.getpid())
        print('3dpw loaded, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')
    if dataset == 'full' or dataset == 'h36m':
        train_data_h36m = get_data_h36m(data_path=data_path_h36m,
                                subject_list=subject_list_trn,
                                load_from_zarr=load_from_zarr_h36m_trn,
                                load_datalist=load_datalist_trn,
                                img_size=img_size,
                                mask=mask,
                                backgrounds=backgrounds,
                                smpl=smpl.layer['neutral'],
                                store_images=store_images_h36m
                                )
        process = psutil.Process(os.getpid())
        print('h36m train loaded, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')
        if val_on_h36m or dataset == 'h36m':
            val_data_h36m = get_data_h36m(data_path=data_path_h36m,
                                    subject_list=subject_list_val, 
                                    load_from_zarr=load_from_zarr_h36m_val,
                                    load_datalist=load_datalist_val,
                                    img_size=img_size,
                                    mask=mask,
                                    backgrounds=backgrounds,
                                    smpl=smpl.layer['neutral'],
                                    store_images=store_images_h36m,
                                    )
            process = psutil.Process(os.getpid())
            print('h36m valid loaded, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')

    train_data = FullDataset(train_data_3dpw, train_data_h36m)
    val_data = [dataset for dataset in [val_data_3dpw, val_data_h36m] if len(dataset) != 0]

    print(f'length train data: 3dpw: {len(train_data_3dpw)}, h36m: {len(train_data_h36m)}, total: {len(train_data_3dpw)+len(train_data_h36m)}')
    print(f'length validation data: 3dpw: {len(val_data_3dpw)}, h36m: {len(val_data_h36m)}, total: {len(val_data_3dpw)+len(val_data_h36m)}')

    return train_data, val_data

def get_full_train_val_data_wo_trans(
        dataset:str='full',
        data_path_3dpw:str='../3DPW',
        num_required_keypoints:int = 0,
        store_sequences=True,
        store_images_3dpw=False,
        load_from_zarr_3dpw_trn:str=None,
        load_from_zarr_3dpw_val:str=None,
        img_size=224,
        load_ids_imgpaths_seq_trn=None,
        load_ids_imgpaths_seq_val=None,
        data_path_h36m:str='../H36M',
        store_images_h36m=False,
        load_from_zarr_h36m_trn:list=None,
        load_from_zarr_h36m_val:list=None,
        load_datalist_trn:str=None,
        load_datalist_val:str=None,
        backgrounds:list=None,
        mask:bool=True,
        val_on_h36m:bool=False,
        subject_list_trn:list=[],
        subject_list_val:list=[],
        fitting_thr=25,
    ): 
    print('initialize smpl model')
    smpl = SMPL()
    smpl.layer['neutral'].th_shapedirs = smpl.layer['neutral'].th_shapedirs[:,:,:10]
    smpl.layer['neutral'].th_betas = smpl.layer['neutral'].th_betas[:,:10]
    train_data_3dpw = val_data_3dpw = train_data_h36m = val_data_h36m = []

    if dataset == 'full' or dataset == '3dpw':
        train_data_3dpw = get_data_3dpw_wo(data_path=data_path_3dpw, 
                                        split='train',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images_3dpw,
                                        load_from_zarr=load_from_zarr_3dpw_trn,
                                        img_size=img_size,
                                        load_ids_imgpaths_seq=load_ids_imgpaths_seq_trn,
                                        smpl=smpl.layer['neutral'],)
                                        
        val_data_3dpw = get_data_3dpw_wo(data_path=data_path_3dpw, 
                                        split='validation',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images_3dpw,
                                        load_from_zarr=load_from_zarr_3dpw_val,
                                        img_size=img_size,
                                        load_ids_imgpaths_seq=load_ids_imgpaths_seq_val,
                                        smpl=smpl.layer['neutral'],)
        process = psutil.Process(os.getpid())
        print('3dpw loaded, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')
    if dataset == 'full' or dataset == 'h36m':
        train_data_h36m = get_data_h36m_wo(data_path=data_path_h36m,
                                subject_list=subject_list_trn,
                                load_from_zarr=load_from_zarr_h36m_trn,
                                load_datalist=load_datalist_trn,
                                img_size=img_size,
                                mask=mask,
                                backgrounds=backgrounds,
                                smpl=smpl.layer['neutral'],
                                store_images=store_images_h36m
                                )
        process = psutil.Process(os.getpid())
        print('h36m train loaded, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')
        if val_on_h36m or dataset == 'h36m':
            val_data_h36m = get_data_h36m_wo(data_path=data_path_h36m,
                                    subject_list=subject_list_val, 
                                    load_from_zarr=load_from_zarr_h36m_val,
                                    load_datalist=load_datalist_val,
                                    img_size=img_size,
                                    mask=mask,
                                    backgrounds=backgrounds,
                                    smpl=smpl.layer['neutral'],
                                    store_images=store_images_h36m,
                                    )
            process = psutil.Process(os.getpid())
            print('h36m valid loaded, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')

    train_data = FullDataset(train_data_3dpw, train_data_h36m)
    val_data = [dataset for dataset in [val_data_3dpw, val_data_h36m] if len(dataset) != 0]

    print(f'length train data: 3dpw: {len(train_data_3dpw)}, h36m: {len(train_data_h36m)}, total: {len(train_data_3dpw)+len(train_data_h36m)}')
    print(f'length validation data: 3dpw: {len(val_data_3dpw)}, h36m: {len(val_data_h36m)}, total: {len(val_data_3dpw)+len(val_data_h36m)}')

    return train_data, val_data

def get_full_seq_train_val_data(
        dataset:str='full',
        data_path_3dpw:str='../3DPW',
        len_chunks=8,
        num_required_keypoints:int = 0,
        store_sequences=True,
        store_images_3dpw=False,
        load_from_zarr_3dpw_trn:str=None,
        load_from_zarr_3dpw_val:str=None,
        img_size=224,
        load_chunks_seq_trn=None,
        load_chunks_seq_val=None,
        data_path_h36m:str='../H36M',
        store_images_h36m=False,
        load_from_zarr_h36m_trn:list=None,
        load_from_zarr_h36m_val:list=None,
        load_seq_datalist_trn:str=None,
        load_seq_datalist_val:str=None,
        backgrounds:list=None,
        mask:bool=True,
        val_on_h36m:bool=False,
        subject_list_trn:list=[],
        subject_list_val:list=[],
        fitting_thr=25,
    ): 
    print('initialize smpl model')
    smpl = SMPL()
    smpl.layer['neutral'].th_shapedirs = smpl.layer['neutral'].th_shapedirs[:,:,:10]
    smpl.layer['neutral'].th_betas = smpl.layer['neutral'].th_betas[:,:10]
    train_data_3dpw = val_data_3dpw = train_data_h36m = val_data_h36m = []
    if dataset == 'full' or dataset == '3dpw':
        print('get 3dpw train data')
        train_data_3dpw = get_data_3dpw_seq(data_path=data_path_3dpw, 
                                        split='train',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images_3dpw,
                                        load_from_zarr=load_from_zarr_3dpw_trn,
                                        img_size=img_size,
                                        load_chunks_seq=load_chunks_seq_trn,
                                        smpl=smpl.layer['neutral'],
                                        len_chunks=len_chunks,
                                        )
        print('get 3dpw validation data')    
        val_data_3dpw = get_data_3dpw_seq(data_path=data_path_3dpw, 
                                        split='validation',
                                        num_required_keypoints=num_required_keypoints,
                                        store_sequences=store_sequences,
                                        store_images=store_images_3dpw,
                                        load_from_zarr=load_from_zarr_3dpw_val,
                                        img_size=img_size,
                                        load_chunks_seq=load_chunks_seq_val,
                                        smpl=smpl.layer['neutral'],
                                        len_chunks=len_chunks,
                                        )
    if dataset == 'full' or dataset == 'h36m':
        print('get h36m train data')    
        train_data_h36m = get_data_h36m_seq(data_path=data_path_h36m,
                            subject_list=subject_list_trn,
                            load_from_zarr=load_from_zarr_h36m_trn,
                            load_seq_datalist = load_seq_datalist_trn,
                            img_size=img_size,
                            mask=mask,
                            fitting_thr=fitting_thr,
                            smpl=smpl.layer['neutral'],
                            len_chunks=len_chunks,
                            backgrounds=backgrounds,
                            store_images=store_images_h36m,
                            )
        if val_on_h36m:
            print('get h36m validation data')    
            val_data_h36m = get_data_h36m_seq(data_path=data_path_h36m,
                                subject_list=subject_list_val,
                                load_from_zarr=load_from_zarr_h36m_val,
                                load_seq_datalist = load_seq_datalist_val,
                                img_size=img_size,
                                mask=mask,
                                fitting_thr=fitting_thr,
                                smpl=smpl.layer['neutral'],
                                len_chunks=len_chunks,
                                backgrounds=backgrounds,
                                store_images=store_images_h36m
                                )

    print(f'length train data: 3dpw: {len(train_data_3dpw)}, h36m: {len(train_data_h36m)}, total: {len(train_data_3dpw)+len(train_data_h36m)}')
    print(f'length validation data: 3dpw: {len(val_data_3dpw)}, h36m: {len(val_data_h36m)}, total: {len(val_data_3dpw)+len(val_data_h36m)}')
    train_data = FullDataset(train_data_3dpw, train_data_h36m)
    val_data = [dataset for dataset in [val_data_3dpw, val_data_h36m] if len(dataset) != 0]
    return train_data, val_data



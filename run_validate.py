import argparse
import pprint
from torch.utils.tensorboard import SummaryWriter
import torch
import os.path as osp
import os, psutil

from modules.utils.data_utils import mk_dir_checkpoint
from modules.models import get_model
from modules.train.training import val_loop as val_loop_img
from modules.train.seq_training_verts_smpl import val_loop as val_loop_seq
from modules.losses_metrics import get_criterion_dict, get_metrics_dict
from modules.datasets.FullDataset import get_full_train_val_data as get_full_train_val_data_img
from modules.datasets.FullDataset import get_full_train_val_data as get_full_train_val_data_img
from modules.datasets.FullDataset import get_full_seq_train_val_data as get_full_seq_train_val_data_seq

from modules.utils.data_utils_h36m import get_backgrounds_from_folder
from configs.config import update_cfg
from hrnet_model_imgnet.config.default import update_hrnet_cfg
from modules.smpl_model._smpl import SMPL, Mesh


def parse_args():

    parser = argparse.ArgumentParser(description="Start Training")
    parser.add_argument('--cfg', type=str, help='cfg file path to yaml')
    parser.add_argument('--pretrained', type=str, help='cfg file path to model weight')
    parser.add_argument('--model', type=str, help='cfg file path to model weight')
    parser.add_argument('--opts', nargs="+", help='configuration options (key must already exist!)')
    parser.add_argument('--cfg_hrnet', type=str, help='when using hrnets: path to cfg yaml',
                        default='configs/experiments_hrnet_imgnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
    
    args = parser.parse_args()
    cfg = update_cfg(args.cfg, args.opts)
    cfg_hrnet = None
    if "hrnet" in cfg.MODEL.ENCODER:
        cfg_hrnet = update_hrnet_cfg(args.cfg_hrnet)
    return cfg, cfg_hrnet, args.pretrained, args.model

def main(cfg, cfg_hrnet, pretrained, model):
    print('start training')
    process = psutil.Process(os.getpid())
    print('start training 1, current memory', process.memory_info().rss/(1024*2024*1024), 'GB')
    print('config as it should be:')
    pprint.pprint( cfg)

    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
            
    writer = SummaryWriter(cfg.LOGGING.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0) 

    metrics = get_metrics_dict(cfg.METRIC) 
    criterion = get_criterion_dict(cfg.LOSS)

    #backgrounds = get_backgrounds_from_folder(osp.join(cfg.DATASETS.H36M, 'backgrounds'))
    load_from_zarr_h36m_trn = [cfg.H36M.LOAD_FROM_ZARR+f'_{subj}to{subj}subj.zarr' for subj in cfg.H36M.SUBJ_LIST.TRN]
    load_from_zarr_h36m_val = [cfg.H36M.LOAD_FROM_ZARR+f'_{subj}to{subj}subj.zarr' for subj in cfg.H36M.SUBJ_LIST.VAL]

    if model == 'img':
        _, val_data = get_full_train_val_data_img(
            dataset=cfg.DATASET_OPT,
            data_path_3dpw= cfg.DATASETS.THREEDPW,
            num_required_keypoints = cfg.TRAIN.NUM_REQUIRED_KPS,
            store_sequences=cfg.THREEDPW.STORE_SEQUENCES,
            store_images_3dpw=cfg.THREEDPW.STORE_IMAGES,
            load_from_zarr_3dpw_trn=cfg.THREEDPW.LOAD_FROM_ZARR.TRN,
            load_from_zarr_3dpw_val=cfg.THREEDPW.LOAD_FROM_ZARR.VAL,
            img_size=cfg.IMG_SIZE,
            load_ids_imgpaths_seq_trn=cfg.THREEDPW.LOAD_IDS_IMGPATHS_SEQ.TRN,
            load_ids_imgpaths_seq_val=cfg.THREEDPW.LOAD_IDS_IMGPATHS_SEQ.VAL,
            data_path_h36m=cfg.DATASETS.H36M,
            load_from_zarr_h36m_trn=load_from_zarr_h36m_trn,
            load_from_zarr_h36m_val=load_from_zarr_h36m_val,
            load_datalist_trn=cfg.H36M.LOAD_DATALIST.TRN,
            load_datalist_val=cfg.H36M.LOAD_DATALIST.VAL,
            backgrounds=None,
            mask=cfg.H36M.MASK,
            store_images_h36m=cfg.H36M.STORE_IMAGES,
            val_on_h36m=cfg.H36M.VAL_ON_H36M,
            subject_list_trn=cfg.H36M.SUBJ_LIST.TRN,
            subject_list_val=cfg.H36M.SUBJ_LIST.VAL,
        )
    else: 
        _, val_data = get_full_seq_train_val_data(dataset= cfg.DATASET_OPT,
                                data_path_3dpw= cfg.DATASETS.THREEDPW,
                                len_chunks=cfg.TRAIN.LEN_CHUNKS,
                                num_required_keypoints = cfg.TRAIN.NUM_REQUIRED_KPS,
                                store_sequences=cfg.THREEDPW.STORE_SEQUENCES,
                                store_images_3dpw=cfg.THREEDPW.STORE_IMAGES,
                                load_from_zarr_3dpw_trn=cfg.THREEDPW.LOAD_FROM_ZARR.TRN,
                                load_from_zarr_3dpw_val=cfg.THREEDPW.LOAD_FROM_ZARR.VAL,
                                img_size=cfg.IMG_SIZE,
                                load_chunks_seq_trn=cfg.THREEDPW.LOAD_CHUNKS_SEQ.TRN,
                                load_chunks_seq_val=cfg.THREEDPW.LOAD_CHUNKS_SEQ.VAL,
                                data_path_h36m=cfg.DATASETS.H36M,
                                load_from_zarr_h36m_trn=load_from_zarr_h36m_trn,
                                load_from_zarr_h36m_val=load_from_zarr_h36m_val,
                                load_seq_datalist_trn=cfg.H36M.LOAD_SEQ_DATALIST.TRN,
                                load_seq_datalist_val=cfg.H36M.LOAD_SEQ_DATALIST.VAL,
                                backgrounds=None,
                                mask=cfg.H36M.MASK,
                                store_images_h36m=cfg.H36M.STORE_IMAGES,
                                val_on_h36m=cfg.H36M.VAL_ON_H36M,
                                subject_list_trn=cfg.H36M.SUBJ_LIST.TRN,
                                subject_list_val=cfg.H36M.SUBJ_LIST.VAL,
                            )

    if cfg.TRAIN.BATCH_SIZE_VAL is None:
        batch_size_val = cfg.TRAIN.BATCH_SIZE_TRN
    else: 
        batch_size_val = cfg.TRAIN.BATCH_SIZE_TRN

    loader_val = [torch.utils.data.DataLoader(dataset=data, batch_size=batch_size_val, shuffle=False,) for data in  val_data]
    smpl = SMPL().to(device) 

    model = get_model(cfg.MODEL.DIM_Z, cfg.MODEL.ENCODER, cfg_hrnet)
    model = model.to(device)

    checkpoint = torch.load(pretrained, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('config as it is:')
    pprint.pprint(checkpoint['config_model']) 
    process = psutil.Process(os.getpid())
    print('datasets loaded, current memory', process.memory_info().rss/(1024*1024*1024), 'GB')
    if True:
        loss_val, min_mpve = val_loop_img(model=model, 
                                    loader_val=loader_val,
                                    criterion=criterion, 
                                    metrics=metrics, 
                                    epoch=0, 
                                    writer=writer, 
                                    log_steps=cfg.LOGGING.LOG_STEPS, 
                                    device=device,
                                    smpl=smpl,
                                    scale = False)
    else:
        mesh_sampler = Mesh()
        loss_val, min_mpve = val_loop_seq(model=model, 
                                    loader_val=loader_val,
                                    criterion=criterion, 
                                    metrics=metrics, 
                                    epoch=0, 
                                    writer=writer, 
                                    log_steps=cfg.LOGGING.LOG_STEPS, 
                                    device=device,
                                    smpl=smpl,
                                    mesh_sampler=mesh_sampler,
                                    scale = False)

    print(f'Epoch 0: Loss Val: {loss_val}, min mpve: {min_mpve}')

if __name__ == '__main__':
    cfg, cfg_hrnet, pretrained, model = parse_args()
    pprint.pprint(cfg)
    main(cfg, cfg_hrnet, pretrained, model)
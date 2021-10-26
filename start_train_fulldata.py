import argparse
import pprint
from torch.utils.tensorboard import SummaryWriter
import torch
import os.path as osp


from modules.models import get_model
from modules.train.training import train_model
from modules.losses_metrics import get_criterion_dict, get_metrics_dict
from modules.datasets.FullDataset import get_full_train_val_data
from modules.utils.data_utils_h36m import get_backgrounds_from_folder
from configs.config import update_cfg
from hrnet_model_imgnet.config.default import update_hrnet_cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Start Training")
    parser.add_argument('--cfg', type=str, help='cfg file path to yaml')
    parser.add_argument('--dataset', type=str, help="'full', 'h36m', '3dpw'", default='full')
    parser.add_argument('--opts', nargs="+", help='configuration options (key must already exist!)')
    parser.add_argument('--cfg_hrnet', type=str, help='when using hrnets: path to cfg yaml',
                        default='configs/experiments_hrnet_imgnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
    
    args = parser.parse_args()
    cfg = update_cfg(args.cfg, args.opts)
    cfg_hrnet = None
    if "hrnet" in cfg.MODEL.ENCODER:
        cfg_hrnet = update_hrnet_cfg(args.cfg_hrnet)
    return cfg, cfg_hrnet, args.dataset

def main(cfg, cfg_hrnet, dataset_opt):
    pprint.pprint(cfg)
    writer = SummaryWriter(cfg.LOGGING.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0) 

    metrics = get_metrics_dict(cfg.METRIC) 
    criterion = get_criterion_dict(cfg.LOSS)

    backgrounds = get_backgrounds_from_folder(osp.join(cfg.DATASETS.H36M, 'backgrounds'))

    train_data, val_data = get_full_train_val_data(
        dataset= dataset_opt,
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
        load_from_zarr_h36m_trn=cfg.H36M.LOAD_FROM_ZARR.TRN,
        load_from_zarr_h36m_val=cfg.H36M.LOAD_FROM_ZARR.VAL,
        load_datalist_trn=cfg.H36M.LOAD_DATALIST.TRN,
        load_datalist_val=cfg.H36M.LOAD_DATALIST.VAL,
        backgrounds=backgrounds,
        mask=cfg.H36M.MASK,
        store_images_h36m=cfg.H36M.STORE_IMAGES,
        val_on_h36m=cfg.H36M.VAL_ON_H36M,
    )

    print("length train and val data:", len(train_data), len(val_data))
    model = get_model(cfg.MODEL.DIM_Z, cfg.MODEL.ENCODER, cfg_hrnet)
    
    dummy_input = next(iter(torch.utils.data.DataLoader((train_data))))["img"]
    writer.add_graph(model, dummy_input)

    train_model(
        model=model,
        num_epochs=cfg.TRAIN.NUM_EPOCHS,
        data_trn=train_data,
        data_val=val_data,
        criterion=criterion,
        metrics=metrics,
        batch_size_trn=cfg.TRAIN.BATCH_SIZE_TRN,
        batch_size_val=cfg.TRAIN.BATCH_SIZE_VAL,
        learning_rate=cfg.TRAIN.LEARNING_RATE,
        writer=writer,
        log_steps = cfg.LOGGING.LOG_STEPS,
        checkpoint_dir=cfg.OUT_DIR,
        cfgs=(cfg, cfg_hrnet),
    )
    
if __name__ == '__main__':
    cfg, cfg_hrnet, dataset_opt = parse_args()
    main(cfg, cfg_hrnet, dataset_opt)
import torchvision.models as models
import argparse
import pprint
from torch.utils.tensorboard import SummaryWriter
import torch

from modules.models import get_model
from modules.training import train_model
from modules.losses_metrics import get_criterion_dict, get_metrics_dict
from modules.dataset_3DPW import get_train_val_data
from configs.config import update_cfg
from hrnet_model_imgnet.config.default import update_hrnet_cfg
from modules.training import val_loop

def parse_args():

    parser = argparse.ArgumentParser(description="Start Training")
    parser.add_argument('--cfg', type=str, help='cfg file path to yaml')
    parser.add_argument('--opts', nargs="+", help='configuration options (key must already exist!)')
    parser.add_argument('--cfg_hrnet', type=str, help='when using hrnets: path to cfg yaml',
                        default='configs/experiments_hrnet_imgnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
    
    args = parser.parse_args()
    cfg = update_cfg(args.cfg, args.opts)
    cfg_hrnet = None
    if "hrnet" in cfg.MODEL.ENCODER:
        cfg_hrnet = update_hrnet_cfg(args.cfg_hrnet)
    return cfg, cfg_hrnet

def main(cfg, cfg_hrnet):
    device = 'cuda'
    writer = SummaryWriter(cfg.LOGGING.LOGDIR)
    metrics = get_metrics_dict() 
    criterion = get_criterion_dict(cfg.LOSS)
    _, val_data = get_train_val_data(data_path=cfg.DATASETS.THREEDPW, 
                                    num_required_keypoints=cfg.TRAIN.NUM_REQUIRED_KPS, 
                                    store_sequences=cfg.STORE_SEQUENCES,
                                    store_images=cfg.STORE_IMAGES,
                                    load_from_zarr_trn=cfg.LOAD_FROM_ZARR.TRN,
                                    load_from_zarr_val=cfg.LOAD_FROM_ZARR.VAL,
                                    img_size=cfg.IMG_SIZE,
                                    load_ids_imgpaths_seq_trn=cfg.LOAD_IDS_IMGPATHS_SEQ.TRN,
                                    load_ids_imgpaths_seq_val=cfg.LOAD_IDS_IMGPATHS_SEQ.VAL,
                                    )

    print("length of val data:", len(val_data))
    model = get_model(cfg.MODEL.DIM_Z, cfg.MODEL.ENCODER, cfg_hrnet)
    loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=cfg.TRAIN.BATCH_SIZE_VAL,
                                             shuffle=False,
                                             )
    min_mpve = float('inf') 

    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        loss_val, min_mpve = val_loop(model=model, 
                                    loader_val=loader_val,
                                    criterion=criterion, 
                                    metrics=metrics, 
                                    epoch=epoch, 
                                    writer=writer, 
                                    log_steps=cfg.LOGGING.LOG_STEPS, 
                                    device=device,
                                    checkpoint_dir=cfg.OUT_DIR,
                                    cfgs=(cfg, cfg_hrnet),
                                    min_mpve=min_mpve,)
        print(f'Epoch: {epoch}; Loss Val: {loss_val}, min mpve: {min_mpve}')

if __name__ == '__main__':
    cfg, cfg_hrnet = parse_args()
    main(cfg, cfg_hrnet)
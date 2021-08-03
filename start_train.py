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

def parse_args():

    parser = argparse.ArgumentParser(description="Start Training")
    parser.add_argument('--cfg', type=str, help='cfg file path to yaml')
    parser.add_argument('--opts', nargs="+", help='configuration options (key must already exist!)')
    parser.add_argument('--cfg_hrnet', type=str, help='when using hrnets: path to cfg yaml',
                        default='configs/experiments_hrnet_imgnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml')
    
    args = parser.parse_args()
    cfg = update_cfg(args.cfg, args.opts)
    cfg_hrnet = None
    if cfg.MODEL.ENCODER == "hrnet":
        cfg_hrnet = update_hrnet_cfg(args.cfg_hrnet)
    return cfg, cfg_hrnet

def main(cfg, cfg_hrnet):

    pprint.pprint(cfg)
    writer = SummaryWriter(cfg.LOGGING.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    metrics = get_metrics_dict() 
    criterion = get_criterion_dict(cfg.LOSS)
    train_data, val_data = get_train_val_data(data_path=cfg.DATASETS.THREEDPW, 
                                              num_required_keypoints=cfg.TRAIN.NUM_REQUIRED_KPS, 
                                              store_sequences=cfg.STORE_SEQUENCES,
                                              store_images=cfg.STORE_IMAGES,
                                              load_from_zarr_trn=cfg.LOAD_FROM_ZARR.TRN,
                                              load_from_zarr_val=cfg.LOAD_FROM_ZARR.VAL,
                                              img_size=cfg.IMG_SIZE,
                                              load_ids_imgpaths_trn=cfg.LOAD_IDS_IMGPATHS.TRN,
                                              load_ids_imgpaths_val=cfg.LOAD_IDS_IMGPATHS.VAL,
                                             )

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
    )
    
if __name__ == '__main__':
    cfg, cfg_hrnet = parse_args()
    print(cfg)
    main(cfg, cfg_hrnet)
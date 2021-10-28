import argparse
import pprint 
from torch.utils.tensorboard import SummaryWriter
import torch

from modules.models import get_model_seq
from modules.train.seq_training import train_model
from modules.losses_metrics import get_criterion_dict, get_metrics_dict
from modules.datasets.dataset_seq_3DPW import get_train_val_data
from modules.utils.data_utils import mk_dir_checkpoint
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
    if "hrnet" in cfg.MODEL.ENCODER:
        cfg_hrnet = update_hrnet_cfg(args.cfg_hrnet)
    return cfg, cfg_hrnet

def main(cfg, cfg_hrnet):
    pprint.pprint(cfg)
    writer = SummaryWriter(cfg.LOGGING.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    metrics = get_metrics_dict(cfg.METRIC) 
    criterion = get_criterion_dict(cfg.LOSS)
    print(metrics, criterion)
    train_data, val_data = get_train_val_data(data_path=cfg.DATASETS.THREEDPW, 
                                              num_required_keypoints=cfg.TRAIN.NUM_REQUIRED_KPS, 
                                              len_chunks=cfg.TRAIN.LEN_CHUNKS,
                                              store_sequences=cfg.THREEDPW.STORE_SEQUENCES,
                                              store_images=cfg.THREEDPW.STORE_IMAGES,
                                              load_from_zarr_trn=cfg.THREEDPW.LOAD_FROM_ZARR.TRN,
                                              load_from_zarr_val=cfg.THREEDPW.LOAD_FROM_ZARR.VAL,
                                              img_size=cfg.IMG_SIZE,
                                              load_chunks_seq_trn=cfg.THREEDPW.LOAD_CHUNKS_SEQ.TRN,
                                              load_chunks_seq_val=cfg.THREEDPW.LOAD_CHUNKS_SEQ.VAL,
                                              )
    print("length train and val data:", len(train_data), len(val_data))
    model = get_model_seq(dim_z_pose = cfg.MODEL.DIM_Z_POSE, dim_z_shape= cfg.MODEL.DIM_Z_SHAPE ,encoder=cfg.MODEL.ENCODER, cfg_hrnet= cfg_hrnet)
    
    dummy_input = next(iter(torch.utils.data.DataLoader((train_data))))["img"]
    writer.add_graph(model, dummy_input)

    checkpoint_dir = mk_dir_checkpoint(cfg.OUT_DIR, (cfg, cfg_hrnet), type='seqwise' )

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
        checkpoint_dir = checkpoint_dir,
        cfgs=(cfg, cfg_hrnet),
    )
    
if __name__ == '__main__':
    cfg, cfg_hrnet = parse_args()
    main(cfg, cfg_hrnet)
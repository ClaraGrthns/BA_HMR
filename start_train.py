import torchvision.models as models
import argparse
import pprint
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch
from modules.models import get_model
from modules.training import train_model
from modules.losses_metrics import get_criterion_dict, get_metrics_dict
from modules.dataset_3DPW import get_train_val_data
from configs.config import get_cfg_defaults, update_cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Start Training")
    parser.add_argument('--cfg', type=str, help='cfg file path')
    args = parser.parse_args()
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()
    return cfg

def main(cfg):

    writer = SummaryWriter(cfg.LOGGING.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)


    metrics = get_metrics_dict() 
    criterion = get_criterion_dict(cfg.LOSS)
    train_data, val_data = get_train_val_data(cfg.DATASETS.ThreeDPW , cfg.TRAIN.NUM_REQUIRED_KPS)
    model = get_model(cfg.MODEL.DIM_Z, cfg.MODEL.ENCODER)
    
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
    cfg = parse_args()
    main(cfg)
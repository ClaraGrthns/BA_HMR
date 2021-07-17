import torchvision.models as models
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

from modules.models import get_model
from modules.training import train_model
from modules.losses_metrics import get_criterion_dict, get_metrics_dict
from modules.dataset_3DPW import get_train_val_data

def main(cfg):

    data_path = cfg['DATASETS']['3DPW']      
    num_epochs=cfg['TRAIN']['NUM_EPOCHS']
    batch_size_trn=cfg['TRAIN']['BATCH_SIZE_TRN']
    batch_size_val=cfg['TRAIN']['BATCH_SIZE_VAL']
    learning_rate= cfg['TRAIN']['LEARNING_RATE']
    num_required_keypoints = cfg['NUM_REQUIRED_KPS']   
    dim_z = cfg['MODEL']['DIM_Z']
    encoder = cfg['MODEL']['ENCODER']
    logdir = cfg['LOGDIR']

    metrics = get_metrics_dict() 
    criterion = get_criterion_dict()
    train_data, val_data = get_train_val_data(data_path, num_required_keypoints)

    model = get_model(dim_z, encoder)
    print(data_path, num_epochs, batch_size_trn, batch_size_val, learning_rate, num_required_keypoints, dim_z, encoder)

    writer = SummaryWriter(logdir)

    train_model(
        model=model,
        num_epochs=num_epochs,
        data_trn=train_data,
        data_val=val_data,
        criterion=criterion,
        metrics=metrics,
        batch_size_trn=batch_size_trn,
        batch_size_val=batch_size_val,
        learning_rate=learning_rate,
        writer=writer,
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path', default='./configs/default_config.yaml')
    args = parser.parse_args()
    print(args)
    with open(args.cfg, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.CLoader)
    main(cfg)
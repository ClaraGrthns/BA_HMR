import os.path as osp
import torch
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt
import random 

 
def rand_partition(list_in, n, len_chunks):
#Source: https://stackoverflow.com/questions/3352737/how-to-randomly-partition-a-list-into-n-nearly-equal-parts
    ''' Devides List into n random sublists 
    Args: 
        list_in: List that will be devided
        n: number of sublists
        len_chunks: length of each sublist
    Returns:
        List of n sublists of length len_chunks
    '''
    rand_chunks = list_in.copy()
    random.shuffle(rand_chunks)
    return [rand_chunks[i::n][:len_chunks] for i in range(n)]

def save_checkpoint(model, optimizer, loss, name, epoch, checkpoint_dir, cfgs ):
    # Save model checkpoint in subdir corresponding to date of training and model/training params
    filepath = osp.join(checkpoint_dir, f'checkpoint_{name}_epoch_{epoch}.pt')
    save_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'running_loss': loss,
            'config_model': cfgs,
            }
    if name == 'latest_ckpt':
        save_model['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(save_model, filepath)

def mk_dir_checkpoint(checkpoint_dir, cfgs, type='imgwise' ):
    today = date.today()
    dt_string = today.strftime("%m_%d_%Y")
    model_string = f'{type}_{cfgs[0].MODEL.ENCODER}_bs{cfgs[0].TRAIN.BATCH_SIZE_TRN}_lr{cfgs[0].TRAIN.LEARNING_RATE}_minkp{cfgs[0].TRAIN.NUM_REQUIRED_KPS}'
    checkpoint_subdir = osp.join(checkpoint_dir, dt_string, model_string)
    Path(checkpoint_subdir).mkdir(parents=True, exist_ok=True)
    return checkpoint_subdir

def log_loss_and_metrics(writer, loss, metrics, log_steps, iteration, name):
    # ...log the running loss
    for loss_key in loss.keys():
        writer.add_scalar(f'{name} loss: {loss_key}', loss[loss_key]/log_steps, iteration)
    # ...log the metrics
    for metr_key in metrics.keys():
        writer.add_scalar(f'{name} metrics: {metr_key}', metrics[metr_key]/log_steps, iteration)

def plotErrors(errors, xrange, save_plot=None):
    plt.figure(figsize=(8, 5))
    plt.plot(xrange, errors, color="forestgreen")
    plt.show()
    plt.close()


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
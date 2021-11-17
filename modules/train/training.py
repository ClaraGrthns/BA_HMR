import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import os, psutil

from ..smpl_model._smpl import SMPL, H36M_J17_NAME
from ..utils.data_utils import save_checkpoint
import matplotlib.pyplot as plt


def plotErrors(errors, xrange, save_plot=None):
    plt.figure(figsize=(8, 5))
    plt.plot(xrange, errors, color="forestgreen")
    plt.show()
    plt.close()


def _loop(
    name,
    train,
    model,
    optimizer,
    loader,
    criterion,
    metrics,
    epoch,
    writer,
    log_steps,
    device,
):
    if train:
        model.train() 
    else:
        model.eval()
    running_loss = dict.fromkeys(criterion.keys(), 0.)
    epoch_loss = 0
    running_metrics = dict.fromkeys(metrics.keys(), 0.)
    smpl = SMPL().to(device) 

    #for i, batch in tqdm(enumerate(loader), total = len(loader), desc= f'Epoch {epoch}: {name}-loop'):
    print(f'start {name} loop!')
    for i, batch in enumerate(loader): 
        if i == 50:
            break
        print(f'{i} / {len(loader)}') 
        img = batch["img"].to(device)
        betas_gt = batch["betas"].to(device)
        poses_gt = batch["poses"].to(device)
        vertices_gt = batch['vertices'].to(device)
        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        #### Forward ####
        prediction = model(img)
   
        # Calculate Vertices with SMPL-model
        betas_pred, poses_pred = prediction  
        vertices_pred = smpl(beta=betas_pred, pose=poses_pred)

        # Get 3d Joints from smpl-model (dim: 17x3) and normalize with Pelvis
        joints3d_gt = smpl.get_h36m_joints(vertices_gt)
        joints3d_pred = smpl.get_h36m_joints(vertices_pred)
        pelvis_gt = joints3d_gt[:,H36M_J17_NAME.index('Pelvis'),:]
        pelvis_pred = joints3d_pred[:, H36M_J17_NAME.index('Pelvis'),:] 
        vertices_gt = vertices_gt - pelvis_gt[:, None, :]
        vertices_pred = vertices_pred - pelvis_pred[:, None, :]
        
        # List of Preds and Targets for smpl-params, vertices, (2d-keypoints and 3d-keypoints)
        preds = {"SMPL": (betas_pred, poses_pred), "VERTS": vertices_pred}
        targets = {"SMPL": (betas_gt, poses_gt), "VERTS": vertices_gt}
        
        #### Losses: Maps keys to losses: loss_smpl, loss_verts, (loss_kp_2d, loss_kp_3d) ####
        loss_batch = 0
        for loss_key in criterion.keys():
            loss = criterion[loss_key][0](preds[loss_key], targets[loss_key]) 
            loss_batch += loss * criterion[loss_key][1] # add weighted loss to total loss of batch
            running_loss[loss_key] += loss.item()
            epoch_loss += loss_batch.item()
        
        if train:
            # backward
            loss_batch.backward()
            # optimize
            optimizer.step()

        process = psutil.Process(os.getpid())
        print('current loop, after train', process.memory_info().rss/(1024*1024*1024), 'GB')
        #### Metrics: Mean per vertex error ####
        for metr_key in metrics.keys():
            running_metrics[metr_key] += metrics[metr_key](preds[metr_key], targets[metr_key])
        if train and i % log_steps == log_steps-1:    # every "log_steps" mini-batches...
            running_loss, running_metrics = log_loss_and_metrics(writer=writer, 
                                                                loss=running_loss, 
                                                                metrics=running_metrics, 
                                                                log_steps=log_steps, 
                                                                iteration=epoch * len(loader) + i,
                                                                name=name,
                                                                train=train)
    if not train:
        log_loss_and_metrics(writer=writer, 
                            loss=running_loss, 
                            metrics=running_metrics, 
                            log_steps=len(loader),
                            iteration=epoch+1, 
                            name=name,
                            train=train)
    return epoch_loss, running_loss, running_metrics

def log_loss_and_metrics(writer, loss, metrics, log_steps, iteration, name, train):
    # ...log the running loss
    for loss_key in loss.keys():
        writer.add_scalar(f'{name} loss: {loss_key}', loss[loss_key]/log_steps, iteration)
        if train:
            loss[loss_key] = 0.0
    # ...log the metrics
    for metr_key in metrics.keys():
        writer.add_scalar(f'{name} metrics: {metr_key}', metrics[metr_key]/log_steps, iteration)
        if train: 
            metrics[metr_key] = 0
    if train:
        return loss, metrics

def trn_loop(model, optimizer, loader_trn, criterion, metrics, epoch, writer,log_steps, device):
    epoch_loss,_,_ =  _loop(
        name='train',
        train=True,
        model=model,
        optimizer=optimizer,
        loader=loader_trn,
        criterion=criterion,
        metrics=metrics,
        epoch=epoch,
        writer=writer,
        log_steps=log_steps, 
        device=device,
    )
    return epoch_loss/len(loader_trn)
    
def val_loop(model, loader_val, criterion, metrics, epoch, writer, log_steps, device):
    data_sets = ['3dpw', 'h36m']
    epoch_loss = 0
    epoch_losses = dict.fromkeys(criterion.keys(), 0)
    epoch_metrics = dict.fromkeys(metrics.keys(), 0)
    total_length = sum([len(loader) for loader in loader_val])

    for data_set, loader in zip(data_sets, loader_val):
        with torch.no_grad():
            name = f'validate on {data_set}'
            aux_loss, aux_losses, aux_metrics = _loop(
                name=name,
                train=False,
                model=model,
                optimizer=None,
                loader=loader,
                criterion=criterion,
                metrics=metrics,
                epoch=epoch,
                writer=writer,
                log_steps = log_steps, 
                device=device,
            )
            epoch_loss += aux_loss
            for key in epoch_losses.keys():
                epoch_losses[key] += aux_losses[key]
            for key in epoch_metrics.keys():
                epoch_metrics[key] += aux_metrics[key]  
                
    log_loss_and_metrics(writer=writer, 
                        loss=epoch_losses, 
                        metrics=epoch_metrics, 
                        log_steps=total_length,
                        iteration=epoch+1, 
                        name='validate on 3dpw & h36m',
                        train=False,)        
    return epoch_loss/total_length, epoch_metrics['VERTS']/total_length

def train_model(model, num_epochs, data_trn, data_val, criterion, metrics,
                batch_size_trn=1, batch_size_val=None, learning_rate=1e-4,
                writer=None, log_steps = 200, device='auto', checkpoint_dir=None, cfgs=None,):
    
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    loader_trn = torch.utils.data.DataLoader(
        dataset=data_trn,
        batch_size=batch_size_trn,
        shuffle=True,
    )
    if batch_size_val is None:
        batch_size_val = batch_size_trn    
    loader_val = [torch.utils.data.DataLoader(dataset=data,batch_size=batch_size_val, shuffle=False,) for data in  data_val]
    min_mpve = float('inf') 

    del(data_trn)
    del(data_val)

    #for epoch in range(num_epochs):
    for epoch in range(3):
        loss_trn = trn_loop(model=model, 
                            optimizer=optimizer, 
                            loader_trn=loader_trn, 
                            criterion=criterion, 
                            metrics=metrics, 
                            epoch=epoch, 
                            writer=writer, 
                            log_steps=log_steps,
                            device=device,            
                            ) 
        save_checkpoint(model=model, 
                        optimizer=optimizer,
                        loss=loss_trn,
                        name='latest_ckpt', 
                        epoch=epoch,
                        iteration=(epoch+1)*len(loader_trn),
                        checkpoint_dir=checkpoint_dir,
                        cfgs=cfgs,
                        ) 
        loss_val, mvpe = val_loop(model=model, 
                            loader_val=loader_val,
                            criterion=criterion, 
                            metrics=metrics, 
                            epoch=epoch, 
                            writer=writer, 
                            log_steps=log_steps, 
                            device=device,                            
                            )
        if mvpe < min_mpve:
            min_mpve = mvpe
            save_checkpoint(model=model, 
                            optimizer=optimizer,
                            loss= loss_val,
                            name='min_val_loss', 
                            epoch=epoch,
                            iteration=(epoch+1)*len(loader_val),
                            checkpoint_dir=checkpoint_dir,
                            cfgs=cfgs,
                            ) 
        print(f'Epoch: {epoch}; Loss Trn: {loss_trn}; Loss Val: {loss_val}, min Mpve: {min_mpve}')
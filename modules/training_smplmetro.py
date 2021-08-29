import torch
import numpy as np
from tqdm import tqdm
import os.path as osp

from .smpl_model._smpl import SMPL, H36M_J17_NAME



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
    checkpoint_dir=None,
    cfgs=None,
    min_mpve=None,
):
    
    if train:
        model.train() 
    else:
        model.eval()
    
    running_loss = dict.fromkeys(criterion.keys(), 0)
    epoch_loss = dict.fromkeys(criterion.keys(), 0)
    running_metrics = dict.fromkeys(metrics.keys(), 0)
    smpl = SMPL().to(device)    
    
    for i, batch in tqdm(enumerate(loader), total = len(loader), desc= f'Epoch {epoch}: {name}-loop'):
        
        img = batch["img"].to(device)
        betas_gt = batch["betas"].to(device)
        poses_gt = batch["poses"].to(device)
        trans_gt = batch["trans"].to(device)
        #poses2d = batch["poses2d"].to(device)
        #poses3d = batch["poses3d"].to(device)
        
        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        #### Forward ####
        prediction = model(img)
   
        # Calculate Vertices with SMPL-model
        betas_pred, poses_pred = prediction        
        vertices_gt = smpl(betas_gt, poses_gt, trans_gt)
        vertices_pred = smpl(betas_pred, poses_pred, trans_gt)

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
            epoch_loss[loss_key] += loss.item()
        
        if train:
            # backward
            loss_batch.backward()
            # optimize
            optimizer.step()
            
        #### Metrics: Mean per vertex error ####
        for metr_key in metrics.keys():
            running_metrics[metr_key] += metrics[metr_key](preds[metr_key], targets[metr_key])
       
        if name == "validate" and running_metrics['VERTS'] < min_mpve:
            save_checkpoint(model=model, 
                            optimizer=optimizer,
                            loss=sum(running_loss.values())/((i%log_steps)+1),
                            name='min_mpve', 
                            epoch=epoch,
                            iteration=(epoch * len(loader) + i),
                            checkpoint_dir=checkpoint_dir,
                            cfgs=cfgs,)
            min_mpve = running_metrics['VERTS']
    
        if i % log_steps == log_steps-1:    # every "log_steps" mini-batches...
                # ...log the running loss
                # ...log the metrics
            for loss_key in running_loss.keys():
                writer.add_scalar(f'{name} loss: {loss_key}' ,
                                running_loss[loss_key]/log_steps,
                                epoch * len(loader) + i)
                running_loss[loss_key] = 0.0

            for metr_key in running_metrics.keys():
                writer.add_scalar(f'{name} metrics: {metr_key}',
                                 running_metrics[metr_key]/log_steps,
                                 epoch * len(loader) + i)
                running_metrics[metr_key] = 0
    return sum(epoch_loss.values())/len(loader), min_mpve

def trn_loop(model, optimizer, loader_trn, criterion, metrics, epoch, writer,log_steps, device,):
    return _loop(
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
    
def val_loop(model, loader_val, criterion, metrics, epoch, writer, log_steps, device, checkpoint_dir, cfgs, min_mpve):
    with torch.no_grad():
        return _loop(
            name='validate',
            train=False,
            model=model,
            optimizer=None,
            loader=loader_val,
            criterion=criterion,
            metrics=metrics,
            epoch=epoch,
            writer=writer,
            log_steps = log_steps, 
            device=device,
            checkpoint_dir=checkpoint_dir,
            cfgs=cfgs,
            min_mpve=min_mpve,
        )

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
        
    loader_val = torch.utils.data.DataLoader(
        dataset=data_val,
        batch_size=batch_size_val,
        shuffle=False,
    )

    min_mpve = float('inf') 

    for epoch in range(num_epochs):
        loss_trn, _ = trn_loop(model=model, 
                            optimizer=optimizer, 
                            loader_trn=loader_trn, 
                            criterion=criterion, 
                            metrics=metrics, 
                            epoch=epoch, 
                            writer=writer, 
                            log_steps=log_steps,
                            device=device)

        save_checkpoint(model=model, 
                        optimizer=optimizer,
                        loss=loss_trn,
                        name='latest_ckpt', 
                        epoch=epoch,
                        iteration=(epoch+1)*len(loader_trn),
                        checkpoint_dir=checkpoint_dir,
                        cfgs=cfgs,)

        loss_val, min_mpve = val_loop(model=model, 
                            loader_val=loader_val,
                            criterion=criterion, 
                            metrics=metrics, 
                            epoch=epoch, 
                            writer=writer, 
                            log_steps=log_steps, 
                            device=device,
                            checkpoint_dir=checkpoint_dir,
                            cfgs=cfgs,
                            min_mpve=min_mpve,)
        
        print(f'Epoch: {epoch}; Loss Trn: {loss_trn}; Loss Val: {loss_val}, min Mpve: {min_mpve}')

def save_checkpoint(model, optimizer, loss, name, epoch, iteration, checkpoint_dir, cfgs):
    filepath = osp.join(checkpoint_dir, f'checkpoint_{name}_{epoch}_{iteration}.pt')
    save_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'running_loss': loss,
            'config_model': cfgs,
            }
    if name == 'latest_ckpt':
        save_model['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(save_model, filepath)
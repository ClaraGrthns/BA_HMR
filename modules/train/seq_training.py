import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
from ..utils.data_utils_3dpw import save_checkpoint
from ..smpl_model._smpl import SMPL, Mesh, H36M_J17_NAME
        
def _loop(
    name,
    train,
    model,
    optimizer,
    loader,
    criterion,
    metrics,
    smpl,
    mesh_sampler,
    epoch,
    writer,
    log_steps,
    device,
):
    
    if train:
        model.train() 
    else:
        model.eval()
    
    running_loss = dict.fromkeys(criterion.keys(), 0)
    epoch_loss = dict.fromkeys(criterion.keys(), 0)
    running_metrics = dict.fromkeys(metrics.keys(), 0)
    
    for i, batch in tqdm(enumerate(loader), total = len(loader), desc= f'Epoch {epoch}: {name}-loop'):
        
        img = batch["img"].to(device)
        betas_gt = batch["betas"].to(device)
        poses_gt = batch["poses"].to(device)
        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        #### Forward ####
        prediction = model(img)
        betas_pred, poses_pred, verts_sub2_pred, verts_sub_pred, verts_full_pred  = prediction  
        # --> dim verts: (bs*seqlen)x|V|x3
        
        # Create Groundtruth-Mesh and downsample it
        verts_gt_full = smpl(poses_gt.reshape(-1, 72), betas_gt.reshape(-1, 10))
        verts_gt_sub = mesh_sampler.downsample(verts_gt_full)
        verts_gt_sub2 = mesh_sampler.downsample(verts_gt_full, n1=0, n2=2)


        # Get 3d Joints from smpl-model (dim: 17x3) and normalize with Pelvis
        joints3d_gt = smpl.get_h36m_joints(verts_gt_full)
        joints3d_pred = smpl.get_h36m_joints(verts_full_pred)
        pelvis_gt = joints3d_gt[:,H36M_J17_NAME.index('Pelvis'),:]
        pelvis_pred = joints3d_pred[:, H36M_J17_NAME.index('Pelvis'),:] 

        #Normalize Groundtruth
        verts_gt_sub2 = verts_gt_sub2 - pelvis_gt[:, None, :]
        verts_gt_sub = verts_gt_sub - pelvis_gt[:, None, :]
        verts_gt_full = verts_gt_full - pelvis_gt[:, None, :]

        #Normalize Prediction
        verts_sub2_pred = verts_sub2_pred - pelvis_pred[:, None, :]
        verts_sub_pred = verts_sub_pred - pelvis_pred[:, None, :]
        verts_full_pred = verts_full_pred - pelvis_pred[:, None, :]
        
        # List of Preds and Targets for smpl-params, verts, (2d-keypoints and 3d-keypoints)
        preds = {"SMPL": (betas_pred, poses_pred), "VERTS_SUB2": verts_sub2_pred , "VERTS_SUB": verts_sub_pred, "VERTS_FULL": verts_full_pred}
        targets = {"SMPL": (betas_gt, poses_gt), "VERTS_SUB2": verts_gt_sub2, "VERTS_SUB": verts_gt_sub, "VERTS_FULL": verts_gt_full}
        
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
    return sum(epoch_loss.values())/len(loader)

def trn_loop(model, optimizer, loader_trn, criterion, metrics, smpl, mesh_sampler, epoch, writer,log_steps, device,):
    return _loop(
        name='train',
        train=True,
        model=model,
        optimizer=optimizer,
        loader=loader_trn,
        criterion=criterion,
        metrics=metrics,
        smpl=smpl,
        mesh_sampler=mesh_sampler,
        epoch=epoch,
        writer=writer,
        log_steps=log_steps, 
        device=device,
    )
    
def val_loop(model, loader_val, criterion, metrics, smpl, mesh_sampler, epoch, writer, log_steps, device):
    with torch.no_grad():
        return _loop(
            name='validate',
            train=False,
            model=model,
            optimizer=None,
            loader=loader_val,
            criterion=criterion,
            metrics=metrics,
            smpl=smpl,
            mesh_sampler=mesh_sampler,
            epoch=epoch,
            writer=writer,
            log_steps = log_steps, 
            device=device,
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

    smpl = SMPL().to(device)
    mesh_sampler = Mesh()

    min_mpve = float('inf') 

    for epoch in range(num_epochs):
        loss_trn = trn_loop(model=model, 
                            optimizer=optimizer, 
                            loader_trn=loader_trn, 
                            criterion=criterion, 
                            metrics=metrics, 
                            smpl=smpl,
                            mesh_sampler=mesh_sampler,
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

        loss_val = val_loop(model=model, 
                            loader_val=loader_val,
                            criterion=criterion, 
                            metrics=metrics,
                            smpl=smpl, 
                            mesh_sampler=mesh_sampler,
                            epoch=epoch, 
                            writer=writer, 
                            log_steps=log_steps, 
                            device=device,
                            )
        if loss_val < min_mpve:
            min_mpve = loss_val
            save_checkpoint(model=model, 
                            optimizer=optimizer,
                            loss=loss_val,
                            name='min_val_loss', 
                            epoch=epoch,
                            iteration=(epoch+1)*len(loader_val),
                            checkpoint_dir=checkpoint_dir,
                            cfgs=cfgs,)

        
        print(f'Epoch: {epoch}; Loss Trn: {loss_trn}; Loss Val: {loss_val}, min Mpve: {min_mpve}')
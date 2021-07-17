import torch
import numpy as np
from .smpl_model.smpl import SMPL
from .losses import criterion_dict, mean_per_vertex_error
import config as cfg

def _loop(
    name,
    train,
    model,
    optimizer,
    loader,
    criterion,
    metrics,
    epoch,
    device,
):
    
    if train:
        model.train()
        
    else:
        model.eval()
    
    running_loss = 0
    running_metrics = dict.fromkeys(metrics.keys(), 0)
    
    
    
    for i, batch in enumerate(iter(loader)):
        
        img = batch["img"].to(device) #.unsqueeze(0)
        betas_gt = batch["betas"].to(device)
        poses_gt = batch["poses"].to(device)
        trans_gt = batch["trans"].to(device)
        poses2d = batch["poses2d"].to(device)
        poses3d = batch["poses3d"].to(device)
        
    
        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        #### Forward
        prediction = model(img)
   
        
        # Calculate Vertices with SMPL-model
        betas_pred, poses_pred = prediction
        smpl = SMPL(cfg.SMPL_Neutral)
        smpl_out_gt = smpl(betas_gt, poses_gt[:, :3], poses_gt[:, 3:], trans_gt)
        smpl_out_pred = smpl(betas_pred, poses_pred[:, :3], poses_pred[:, 3:], trans_gt)
        
        vertices_gt = smpl_out_gt.vertices
        vertices_pred = smpl_out_pred.vertices
        
        # Get 3d Joints from smpl-model (dim: 17x3) and normalize with Pelvis
        joints3d_gt = smpl.get_h36m_joints(vertices_gt)
        joints3d_pred = smpl.get_h36m_joints(vertices_pred)

        pelvis_gt = joints3d_gt[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        pelvis_pred = joints3d_pred[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        
        vertices_gt = vertices_gt - pelvis_gt[:, None, :]
        vertices_pred = vertices_gt - pelvis_pred[:, None, :]
        
        # Lists Preds and Targets for smpl-params, vertices, (2d-keypoints and 3d-keypoints)
        preds = [(betas_pred, poses_pred), vertices_pred]
        targets = [(betas_gt, poses_gt), vertices_gt]
        #targets = [(betas_gt, poses_gt), vertices_gt, poses2d_gt, poses3d_gt]
        #prediction = [(betas_pred, poses_pred), vertices_pred] + list(prediction[2:]) 
        
        #### Losses: Maps keys to losses: loss_smpl, loss_verts, loss_kp_2d, loss_kp_3d
        loss_batch = dict.fromkeys(criterion.keys(), 0)
        for loss_key, pred, target in zip(loss_batch.keys(), preds, targets):
            loss_batch[loss_key] = criterion[loss_key](pred, target)
            running_loss += loss_batch["loss_verts"].item()
            #running_loss += loss_batch[loss_key].item()
            
        if train:
            # backward
            
            #loss_total = sum(loss_batch.values())
            #loss_total.bachward()
            loss_batch["loss_verts"].backward()
            
            # optimize
            optimizer.step()
            
        
        #### Metrics: Mean per vertex error
        for metr_key, pred, target in zip(metrics.keys(), preds, targets):
            running_metrics[metr_key] += metrics[metr_key](pred, target)
        
        if i % 1000 == 999:    # every 1000 mini-batches...
                # ...log the running loss
            writer.add_scalar(f'{name} loss',
                            running_loss / 1000,
                            epoch * len(loader) + i)
            running_loss = 0.0
                # ...log the metrics
            for metr_key in running_metrics.keys():
                writer.add_scalar(f'{name} metrics: {metr_key}',
                                 running_metrics[metr_key]/ 1000,
                                 epoch * len(loader) + i)
                running_metrics[metr_key] = 0
           
    return running_loss / (len(loader)//1000)

def trn_loop(model, optimizer, loader_trn, criterion, metrics, epoch, device):
    return _loop(
        name = 'train',
        train=True,
        model=model,
        optimizer=optimizer,
        loader=loader_trn,
        criterion=criterion,
        metrics=metrics,
        epoch=epoch,
        device=device,
    )
    
def val_loop(model, loader_val, criterion, metrics, epoch, device):
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
            device=device,
        )

def train_model(model, num_epochs, data_trn, data_val, criterion,
                metrics, batch_size_trn=1, batch_size_val=None, learning_rate=1e-4, device='auto'):
    
    if device is 'auto':
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
    
    for epoch in range(num_epochs):
        loss_trn = trn_loop(model=model, optimizer=optimizer, loader_trn=loader_trn, criterion=criterion, metrics=metrics, epoch=epoch, device=device)
        loss_val = val_loop(model=model, loader_val=loader_val, criterion=criterion, metrics=metrics, epoch=epoch, device=device)
        print(f'Epoch: {epoch}; Loss Trn: {loss_trn}; Loss Val: {loss_val}')
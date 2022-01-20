import torch
import numpy as np
from tqdm import tqdm

from ..smpl_model._smpl import SMPL, H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from ..utils.data_utils import save_checkpoint, log_loss_and_metrics




def _loop(
    name,
    train,
    model,
    optimizer,
    loader,
    criterion,
    metrics,
    smpl,
    epoch,
    writer,
    log_steps,
    device,
    scale=False,
):
    if train:
        model.train() 
    else:
        model.eval()

    running_loss = dict.fromkeys(criterion.keys(), 0.)
    epoch_loss = 0
    running_metrics = dict.fromkeys(metrics.keys(), 0.)

    print(f'start {name} loop!')
    for i, batch in tqdm(enumerate(loader), total = len(loader), desc = f'Epoch {epoch}: {name}-loop'):
        img = batch["img"].to(device)
        betas_gt = batch["betas"].to(device)
        poses_gt = batch["poses"].to(device)
        vertices_gt = batch["vertices"].to(device)
        #joints3d_gt = batch["joints_3d"].to(device) # Bx14x3, already standardized with pelvis joint

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        #### Forward ####
        prediction = model(img)
   
        # Calculate Vertices with SMPL-model
        betas_pred, poses_pred = prediction  
        vertices_pred = smpl(beta=betas_pred, pose=poses_pred)

        # Get 3d Joints from smpl-model (dim: 17x3) and normalize with Pelvis
        joints3d_smpl_gt = smpl.get_h36m_joints(vertices_gt)
        joints3d_pred = smpl.get_h36m_joints(vertices_pred)
        
        pelvis_gt = joints3d_smpl_gt[:, H36M_J17_NAME.index('Pelvis'),:]
        torso_gt = joints3d_smpl_gt[:,H36M_J17_NAME.index('Torso'),:]

        pelvis_pred = joints3d_pred[:, H36M_J17_NAME.index('Pelvis'),:] 
        torso_pred = joints3d_pred[:, H36M_J17_NAME.index('Torso'),:]

        # standardized vertices 
        vertices_gt = vertices_gt - pelvis_gt[:, None, :]
        vertices_pred = vertices_pred - pelvis_pred[:, None, :]
        
        # standardized predicted joints (gt joints are already standardized with pelvis)
        joints3d_pred = joints3d_pred[:, H36M_J17_TO_J14,:]
        joints3d_pred = joints3d_pred - pelvis_pred[:, None, :]
        joints3d_smpl_gt = joints3d_smpl_gt[:, H36M_J17_TO_J14,:]
        joints3d_smpl_gt = joints3d_smpl_gt - pelvis_gt[:, None, :]

        if scale:
            scale_smpl_gt = torch.torch.linalg.vector_norm((torso_gt-pelvis_gt), dim=-1, keepdim=True)[:, None, :]
            scale_pred = torch.torch.linalg.vector_norm((torso_pred-pelvis_pred), dim=-1, keepdim=True)[:, None, :] 

            vertices_pred = vertices_pred/scale_pred
            vertices_gt = vertices_gt/scale_smpl_gt
            joints3d_pred = joints3d_pred/scale_pred
            joints3d_smpl_gt = joints3d_smpl_gt/scale_smpl_gt

            # Scale Joints with Left and Right Hip
            #scale_gt = torch.torch.linalg.vector_norm((joints3d_gt[:, 2,:]-joints3d_gt[:, 3,:]), dim=-1, keepdim=True)[:, None, :]
            #scale_pred = torch.torch.linalg.vector_norm((joints3d_pred[:, 2,:]-joints3d_pred[:, 3,:]), dim=-1, keepdim=True)[:, None, :]
            #joints3d_pred = joints3d_pred/scale_pred
            #joints3d_gt = joints3d_gt/scale_gt


        # List of Preds and Targets for smpl-params, vertices, 3d-keypoints, (2d-keypoints)
        preds = {"SMPL": (betas_pred, poses_pred), "VERTS": vertices_pred, "KP_3D": joints3d_pred}
        targets = {"SMPL": (betas_gt, poses_gt), "VERTS": vertices_gt, "KP_3D": joints3d_smpl_gt}
        
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

        #### Metrics: Mean per vertex error ####
        for metr_key in metrics.keys():
            running_metrics[metr_key] += metrics[metr_key](preds[metr_key], targets[metr_key])

        if train and i % log_steps == log_steps-1:    # every "log_steps" mini-batches...
            log_loss_and_metrics(writer=writer, 
                loss=running_loss, 
                metrics=running_metrics, 
                log_steps=log_steps, 
                iteration=epoch*len(loader)+i,
                name=name,
                )
            running_loss = dict.fromkeys(running_loss, 0.)
            running_metrics = dict.fromkeys(running_metrics, 0.)
            writer.add_scalar('loss total, training', epoch_loss/i, epoch*len(loader)+i)

    return epoch_loss, running_loss, running_metrics



def trn_loop(model, optimizer, loader_trn, criterion, metrics, epoch, writer,log_steps, device, smpl, scale):
    epoch_loss,_,running_metrics =  _loop(
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
        smpl=smpl, 
        scale=scale,          
    )
    return epoch_loss/len(loader_trn)
    
def val_loop(model, loader_val, criterion, metrics, epoch, writer, log_steps, device, smpl, scale):
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
                log_steps=log_steps, 
                device=device,
                smpl=smpl,  
                scale=scale,         
            )
            epoch_loss += aux_loss
            for key in epoch_losses.keys():
                epoch_losses[key] += aux_losses[key]
            for key in epoch_metrics.keys():
                epoch_metrics[key] += aux_metrics[key] 

            log_loss_and_metrics(writer=writer, 
                        loss=aux_losses, 
                        metrics=aux_metrics, 
                        log_steps=len(loader),
                        iteration=epoch+1, 
                        name=name,
                        ) 
                
    log_loss_and_metrics(writer=writer, 
                        loss=epoch_losses, 
                        metrics=epoch_metrics, 
                        log_steps=total_length,
                        iteration=epoch+1, 
                        name='validate on 3dpw & h36m',
                        )
    writer.add_scalar('loss total, valid', epoch_loss/total_length, epoch+1)
    if "SMPL" in epoch_metrics.keys():
        print('smpl metric:', epoch_metrics['SMPL']/total_length)
    return epoch_loss/total_length, epoch_metrics['VERTS']/total_length

def train_model(model, num_epochs, data_trn, data_val, criterion, metrics,
                batch_size_trn=1, batch_size_val=None, learning_rate=1e-4,
                writer=None, log_steps = 200, device='auto', checkpoint_dir=None, scale=False, cfgs=None,):
    
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
        
    loader_val = [torch.utils.data.DataLoader(dataset=data, batch_size=batch_size_val, shuffle=False,) for data in  data_val]
    
    del(data_trn)
    del(data_val)

    smpl = SMPL().to(device) 
    min_mpve = float('inf') 
    
    for epoch in range(num_epochs):
        loss_trn = trn_loop(model=model, 
                            optimizer=optimizer, 
                            loader_trn=loader_trn, 
                            criterion=criterion, 
                            metrics=metrics, 
                            epoch=epoch, 
                            writer=writer, 
                            log_steps=log_steps,
                            device=device, 
                            smpl=smpl, 
                            scale=scale,          
                            ) 
        if epoch % 10 == 0:
            save_checkpoint(model=model, 
                        optimizer=optimizer,
                        loss=loss_trn,
                        name='latest_ckpt', 
                        epoch=epoch,
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
                            smpl=smpl, 
                            scale=scale,                
                            )
        if mvpe < min_mpve:
            min_mpve = mvpe
            save_checkpoint(model=model, 
                            optimizer=optimizer,
                            loss=loss_val,
                            name='min_val_loss', 
                            epoch=epoch,
                            checkpoint_dir=checkpoint_dir,
                            cfgs=cfgs,
                            ) 
        print(f'Epoch: {epoch}; Loss Trn: {loss_trn}; Loss Val: {loss_val}, min Mpve: {min_mpve}')
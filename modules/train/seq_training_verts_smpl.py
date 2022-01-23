import torch
from tqdm import tqdm
from ..utils.data_utils import save_checkpoint, log_loss_and_metrics
from ..smpl_model._smpl import SMPL, Mesh, H36M_J17_NAME, H36M_J17_TO_J14
        
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
    scale,
):
    
    if train:
        model.train() 
    else:
        model.eval()
    
    running_loss = dict.fromkeys(criterion.keys(), 0)
    epoch_loss = 0
    running_metrics = dict.fromkeys(metrics.keys(), 0)

    print(f'start {name} loop!')
    for i, batch in tqdm(enumerate(loader), total = len(loader), desc= f'Epoch {epoch}: {name}-loop'):
        img = batch["imgs"]#.to(device)
        betas_gt = batch["betas"]#.to(device)
        poses_gt = batch["poses"]#.to(device)
        verts_full_gt = batch["vertices"]#.to(device)
        verts_full_gt = verts_full_gt.reshape(-1, verts_full_gt.shape[-2], verts_full_gt.shape[-1])

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        #### Forward ####
        prediction = model(img)
        betas_pred, poses_pred, verts_sub2_pred, verts_sub_pred, verts_full_pred  = prediction  
        # --> dim verts: (bs*seqlen)x|V|x3
        verts_pred_smpl = smpl(beta = betas_pred, pose = poses_pred)

        # Create Groundtruth-Mesh and downsample it
        verts_sub_gt = mesh_sampler.downsample(verts_full_gt)
        verts_sub2_gt = mesh_sampler.downsample(verts_full_gt, n1=0, n2=2)


        # Get 3d Joints from smpl-model (dim: 17x3) and normalize with Pelvis
        joints3d_smpl_gt = smpl.get_h36m_joints(verts_full_gt)
        joints3d_pred = smpl.get_h36m_joints(verts_full_pred)
        joints3d_pred_smpl = smpl.get_h36m_joints(verts_pred_smpl)

        pelvis_gt = joints3d_smpl_gt[:, H36M_J17_NAME.index('Pelvis'),:]
        #torso_gt = joints3d_smpl_gt[:,H36M_J17_NAME.index('Torso'),:]

        pelvis_pred = joints3d_pred[:, H36M_J17_NAME.index('Pelvis'),:] 
        pelvis_pred_smpl = joints3d_pred_smpl[:, H36M_J17_NAME.index('Pelvis'),:] 
        #torso_pred = joints3d_pred[:, H36M_J17_NAME.index('Torso'),:]

        #Normalize Groundtruth
        verts_sub2_gt = verts_sub2_gt - pelvis_gt[:, None, :]
        verts_sub_gt = verts_sub_gt - pelvis_gt[:, None, :]
        verts_full_gt = verts_full_gt - pelvis_gt[:, None, :]

        #Normalize Prediction
        verts_sub2_pred = verts_sub2_pred - pelvis_pred[:, None, :]
        verts_sub_pred = verts_sub_pred - pelvis_pred[:, None, :]
        verts_full_pred = verts_full_pred - pelvis_pred[:, None, :]
        verts_pred_smpl = verts_pred_smpl - pelvis_pred_smpl[:, None, :]

        joints3d_pred = joints3d_pred[:, H36M_J17_TO_J14,:]
        joints3d_pred = joints3d_pred - pelvis_pred[:, None, :]
        joints3d_smpl_gt = joints3d_smpl_gt[:, H36M_J17_TO_J14,:]
        joints3d_smpl_gt = joints3d_smpl_gt - pelvis_gt[:, None, :]

        if scale:
            scale_smpl_gt = torch.torch.linalg.vector_norm((torso_gt-pelvis_gt), dim=-1, keepdim=True)[:, None, :]
            scale_pred = torch.torch.linalg.vector_norm((torso_pred-pelvis_pred), dim=-1, keepdim=True)[:, None, :]            
            
            verts_full_gt = verts_full_gt/scale_smpl_gt
            verts_sub2_gt = verts_sub2_gt/scale_smpl_gt
            verts_sub_gt = verts_sub_gt/scale_smpl_gt

            verts_full_pred = verts_full_pred/scale_pred
            verts_sub2_pred = verts_sub2_pred/scale_pred
            verts_sub_pred = verts_sub_pred/scale_pred
    
            joints3d_pred = joints3d_pred/scale_pred
            joints3d_smpl_gt = joints3d_smpl_gt/scale_smpl_gt

            '''# Scale Joints with Left and Right Hip
            scale_gt = torch.torch.linalg.vector_norm((joints3d_gt[:, 2,:]-joints3d_gt[:, 3,:]), dim=-1, keepdim=True)[:, None, :]
            scale_pred = torch.torch.linalg.vector_norm((joints3d_pred[:, 2,:]-joints3d_pred[:, 3,:]), dim=-1, keepdim=True)[:, None, :]
            joints3d_gt = joints3d_gt/scale_gt
            joints3d_pred = joints3d_pred/scale_pred'''

        # List of Preds and Targets for smpl-params, verts, (2d-keypoints and 3d-keypoints)
        preds = {"SMPL": (betas_pred, poses_pred), "VERTS_SUB2": verts_sub2_pred , "VERTS_SUB": verts_sub_pred, "VERTS_FULL": verts_full_pred, "KP_3D": joints3d_pred, "VERTS_SMPL": verts_pred_smpl}
        targets = {"SMPL": (betas_gt, poses_gt), "VERTS_SUB2": verts_sub2_gt, "VERTS_SUB": verts_sub_gt, "VERTS_FULL": verts_full_gt, "KP_3D": joints3d_smpl_gt, "VERTS_SMPL": verts_full_gt}
        
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
       
        if train and  i % log_steps == log_steps-1:    # every "log_steps" mini-batches...
            log_loss_and_metrics(writer=writer, 
                loss=running_loss, 
                metrics=running_metrics, 
                log_steps=log_steps, 
                iteration=epoch*len(loader)+i,
                name=name,
                )
            running_loss= dict.fromkeys(running_loss, 0.)
            running_metrics = dict.fromkeys(running_metrics, 0.)
            writer.add_scalar('loss total, training', epoch_loss/i, epoch*len(loader)+i)
    return epoch_loss, running_loss, running_metrics

def trn_loop(model, optimizer, loader_trn, criterion, metrics, smpl, mesh_sampler, epoch, writer,log_steps, device, scale):
    epoch_loss,_,_ =  _loop(
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
        scale=scale,
    )
    return epoch_loss/len(loader_trn)
    
def val_loop(model, loader_val, criterion, metrics, smpl, mesh_sampler, epoch, writer, log_steps, device, scale):
    datasets = ['3dpw', 'h36m']
    epoch_loss = 0
    epoch_losses = dict.fromkeys(criterion.keys(), 0)
    epoch_metrics = dict.fromkeys(metrics.keys(), 0)
    total_length = sum([len(loader) for loader in loader_val])
    for dataset, loader in zip(datasets, loader_val):
        with torch.no_grad():
            name = f'validate on {dataset}'
            aux_loss, aux_losses, aux_metrics = _loop(
                name=name,
                train=False,
                model=model,
                optimizer=None,
                loader=loader,
                criterion=criterion,
                metrics=metrics,
                smpl=smpl,
                mesh_sampler=mesh_sampler,
                epoch=epoch,
                writer=writer,
                log_steps = log_steps, 
                device=device,
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
                        name=f'validate on {datasets}',
                        )  
        writer.add_scalar('loss total, valid', epoch_loss/total_length, epoch+1)
    if "SMPL" in epoch_metrics.keys():
        print('smpl metrics:', epoch_metrics['SMPL']/total_length)
    return epoch_loss/total_length, epoch_metrics['VERTS_FULL']/total_length

                
def train_model(model, num_epochs, data_trn, data_val, criterion, metrics,
                batch_size_trn=1, batch_size_val=None, learning_rate=1e-4,
                writer=None, log_steps = 200, device='auto',
                checkpoint_dir=None, cfgs=None, scale=False,):
    
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    smpl = SMPL().to(device)
    mesh_sampler = Mesh()

    min_mpve = float('inf') 

    for epoch in range(num_epochs):

        for dataset in data_trn.datasets:
            dataset.set_chunks()   
        loader_trn = torch.utils.data.DataLoader(dataset=data_trn,
                                            batch_size=batch_size_trn,
                                            shuffle=True,)
        if batch_size_val is None:
            batch_size_val = batch_size_trn  
        for dataset in data_val:
            dataset.set_chunks()
        loader_val = [torch.utils.data.DataLoader(dataset=data, 
                                        batch_size=batch_size_val, 
                                        shuffle=False,) for data in data_val]

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
                            device=device,
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

        loss_val, mpve = val_loop(model=model, 
                            loader_val=loader_val,
                            criterion=criterion, 
                            metrics=metrics,
                            smpl=smpl, 
                            mesh_sampler=mesh_sampler,
                            epoch=epoch, 
                            writer=writer, 
                            log_steps=log_steps, 
                            device=device,
                            scale=scale,
                            )
        if mpve < min_mpve:
            min_mpve = mpve
            save_checkpoint(model=model, 
                            optimizer=optimizer,
                            loss=loss_val,
                            name='min_val_loss', 
                            epoch=epoch,
                            checkpoint_dir=checkpoint_dir,
                            cfgs=cfgs,
                            )
        print(f'Epoch: {epoch}; Loss Trn: {loss_trn}; Loss Val: {loss_val}, min Mpve: {min_mpve}')
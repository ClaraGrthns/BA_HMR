import torch
from tqdm import tqdm
from ..utils.data_utils import save_checkpoint, log_loss_and_metrics
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
    epoch_loss = 0
    running_metrics = dict.fromkeys(metrics.keys(), 0)

    print(f'start {name} loop!')
    for i, batch in tqdm(enumerate(loader), total = len(loader), desc= f'Epoch {epoch}: {name}-loop'):
        img = batch["imgs"].to(device)
        betas_gt = batch["betas"].to(device)
        poses_gt = batch["poses"].to(device)
        verts_full_gt = batch["vertices"].to(device)
        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        #### Forward ####
        prediction = model(img)
        betas_pred, poses_pred, verts_sub2_pred, verts_sub_pred, verts_full_pred  = prediction  
        # --> dim verts: (bs*seqlen)x|V|x3
        
        # Create Groundtruth-Mesh and downsample it
        verts_sub_gt = mesh_sampler.downsample(verts_full_gt)
        verts_sub2_gt = mesh_sampler.downsample(verts_full_gt, n1=0, n2=2)


        # Get 3d Joints from smpl-model (dim: 17x3) and normalize with Pelvis
        joints3d_gt = smpl.get_h36m_joints(verts_full_gt)
        joints3d_pred = smpl.get_h36m_joints(verts_full_pred)
        pelvis_gt = joints3d_gt[:,H36M_J17_NAME.index('Pelvis'),:]
        pelvis_pred = joints3d_pred[:, H36M_J17_NAME.index('Pelvis'),:] 

        #Normalize Groundtruth
        verts_sub2_gt = verts_sub2_gt - pelvis_gt[:, None, :]
        verts_sub_gt = verts_sub_gt - pelvis_gt[:, None, :]
        verts_full_gt = verts_full_gt - pelvis_gt[:, None, :]

        #Normalize Prediction
        verts_sub2_pred = verts_sub2_pred - pelvis_pred[:, None, :]
        verts_sub_pred = verts_sub_pred - pelvis_pred[:, None, :]
        verts_full_pred = verts_full_pred - pelvis_pred[:, None, :]
        
        # List of Preds and Targets for smpl-params, verts, (2d-keypoints and 3d-keypoints)
        preds = {"SMPL": (betas_pred, poses_pred), "VERTS_SUB2": verts_sub2_pred , "VERTS_SUB": verts_sub_pred, "VERTS_FULL": verts_full_pred}
        targets = {"SMPL": (betas_gt, poses_gt), "VERTS_SUB2": verts_sub2_gt, "VERTS_SUB": verts_sub_gt, "VERTS_FULL": verts_full_gt}
        
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
    return epoch_loss, running_loss, running_metrics

def trn_loop(model, optimizer, loader_trn, criterion, metrics, smpl, mesh_sampler, epoch, writer,log_steps, device,):
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
    )
    return epoch_loss/len(loader_trn)
    
def val_loop(model, loader_val, criterion, metrics, smpl, mesh_sampler, epoch, writer, log_steps, device):
    #data_sets = ['3dpw', 'h36m']
    epoch_loss = 0
    epoch_losses = dict.fromkeys(criterion.keys(), 0)
    epoch_metrics = dict.fromkeys(metrics.keys(), 0)
    total_length = sum([len(loader) for loader in loader_val])
    for loader in loader_val:
        with torch.no_grad():
            name = f'validate on {loader}'
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
    return epoch_loss/total_length, epoch_metrics['VERTS']/total_length

                
def train_model(model, num_epochs, data_trn, data_val, criterion, metrics,
                batch_size_trn=1, batch_size_val=None, learning_rate=1e-4,
                writer=None, log_steps = 200, device='auto',
                checkpoint_dir=None, cfgs=None,):
    
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

        loader_trn = torch.utils.data.DataLoader(
        dataset = data_trn.set_chunks(),
        batch_size=batch_size_trn,
        shuffle=True,
        )
        if batch_size_val is None:
            batch_size_val = batch_size_trn
            
        loader_val = [torch.utils.data.DataLoader(dataset=data.set_chunks(), 
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
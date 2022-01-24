import torch
import numpy as np
from .utils.geometry import batch_rodrigues

BETA_WEIGHTS = torch.FloatTensor([0.5045, 0.1355, 0.0959, 0.0678, 0.0430, 0.0397, 0.0376, 0.0281, 0.0258, 0.0223])

def criterion_smpl1(param_pred, param_gt):
    betas_pred, poses_pred = param_pred
    betas_gt, poses_gt = param_gt
    betas_pred = betas_pred.reshape(-1,10)
    betas_gt = betas_gt.reshape(-1, 10)
    # Angle Axis Represenation (Bx24x3)--> Rotation Matrix (Bx24x3x3) (-> Rodrigues' Rotation formula)
    if poses_pred.dim() <= 3:
        rotmat_pred = batch_rodrigues(poses_pred.reshape(-1,3)).reshape(-1, 24, 3, 3)
    else: 
        rotmat_pred = poses_pred.reshape(-1, 24, 3, 3)

    rotmat_gt = batch_rodrigues(poses_gt.reshape(-1,3)).reshape(-1, 24, 3, 3)

    if len(rotmat_pred) > 0:
        loss_pose = torch.nn.functional.mse_loss(rotmat_pred, rotmat_gt)
        loss_betas = (BETA_WEIGHTS*(torch.nn.functional.mse_loss(betas_pred, betas_gt,reduction='none')).sum(dim=0)).sum()

    else:
        loss_pose = torch.FloatTensor(1).fill_(0.).to(torch.device("cpu"))
        loss_betas = torch.FloatTensor(1).fill_(0.).to(torch.device("cpu"))

    return 0.001*loss_betas + loss_pose

def criterion_verts(verts_pred, verts_gt):
    if len(verts_gt) > 0:
        return torch.nn.functional.l1_loss(verts_pred, verts_gt)
    else:
        return torch.FloatTensor(1).fill_(0.) 


def criterion_kp_2d(kp_2d_pred, kp_2d_gt):
    criterion_keypoints2d = torch.nn.functional.mse_loss(reduction='none')
    if kp_2d_gt.shape[-1] != 3:
        kp_2d_gt = kp_2d_gt.transpose(2,1)
    conf = kp_2d_gt[:, :, -1].unsqueeze(-1).clone()
    
    loss_kp_2d = (conf * criterion_keypoints2d(kp_2d_pred, kp_2d_gt[:, :, :-1])).mean()
    return  loss_kp_2d



def criterion_kp_3d(kp_3d_pred, kp_3d_gt):
    #Input kp_3d_pred, kp_3d_gt: torch.Tensor([Bx14x3])
    #criterion_keypoints3d = torch.nn.functional.mse_loss()
    if len(kp_3d_gt) > 0:
        pelvis_gt = (kp_3d_gt[:, 2,:] + kp_3d_gt[:, 3,:]) / 2
        kp_3d_gt = kp_3d_gt - pelvis_gt[:, None, :]
        pelvis_pred = (kp_3d_pred[:, 2,:] + kp_3d_pred[:, 3,:]) / 2
        kp_3d_pred = kp_3d_pred - pelvis_pred[:, None, :]
        return torch.nn.functional.l1_loss(kp_3d_pred, kp_3d_gt)
    else:
        return torch.FloatTensor(1).fill_(0.).to(torch.device("cpu")) 

    
def mean_per_vertex_error(vert_pred, vert_gt):
    """
    Compute mPVE
    """
    with torch.no_grad():
        error = np.mean(torch.sqrt(((vert_pred - vert_gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy(), dtype=np.float32)
        return error 
    
def get_metrics_dict(metr_weights):
    metr_dict = {"VERTS": mean_per_vertex_error,
                "VERTS_SMPL": mean_per_vertex_error,
                "VERTS_FULL": mean_per_vertex_error,
                "VERTS_SUB2": mean_per_vertex_error,
                "VERTS_SUB": mean_per_vertex_error,
                "SMPL": criterion_smpl,}  
    return {key: metric for key, metric in metr_dict.items() if metr_weights[key] != 0 }


def get_criterion_dict(loss_weights):
    loss_dict = {"SMPL" : criterion_smpl, 
                 "VERTS": criterion_verts,
                 "VERTS_SMPL": criterion_verts,
                 "VERTS_FULL": criterion_verts,
                 "VERTS_SUB2": criterion_verts,
                 "VERTS_SUB": criterion_verts,
                 "KP_2D": criterion_kp_2d,
                 "KP_3D": criterion_kp_3d,}
    # Maps keys to criteria functions and corresponding weights (only for loss_weights ≠ 0)
    return {key: (criterion, loss_weights[key]) for key, criterion in loss_dict.items() if loss_weights[key] != 0 }

    




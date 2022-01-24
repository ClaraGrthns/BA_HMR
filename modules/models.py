from __future__ import division

import torch
from torch._C import dtype
import torchvision.models as models
from hrnet_model_imgnet.models.cls_hrnet import get_cls_net
import torch.nn.functional as F


class PoseNetXtreme(torch.nn.Module):
    def __init__(self, encoder, decoder, dim_z=256):
        super(PoseNetXtreme, self).__init__()
        self.dim_z = dim_z
        self.encoder = encoder
        self.linear = torch.nn.Linear(1000, self.dim_z)
        self.decoder = decoder
        
    def forward(self, x):
        z = self.linear(torch.nn.functional.relu(self.encoder(x)))
        beta, pose = self.decoder(z)
        return beta, pose

class PoseDecoder(torch.nn.Module):
    def __init__(self, *layers):
        super(PoseDecoder, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1], layers[1:])])
        # residual layer 
        self.linear_beta = torch.nn.Linear(layers[-1], 10)
        self.linear_pose = torch.nn.Linear(layers[-1], 72)
        
    def forward(self, z):
        for layer in self.layers:
            layer_out = layer(torch.relu(z))
            if layer_out.shape[-1] == z.shape[-1]:
                z = z + layer_out
            else:
                z = layer_out
        beta = self.linear_beta(z)
        pose = self.linear_pose(z)
        return beta, pose

def get_encoder(encoder:str, cfg_hrnet):
    if "resnet" in encoder:
        return models.__dict__[encoder](pretrained = True)
    else: 
        return get_cls_net(cfg_hrnet)

def get_model(dim_z, encoder, cfg_hrnet):
    encoder_pretrained = get_encoder(encoder, cfg_hrnet)
    model = PoseNetXtreme(
        encoder=encoder_pretrained,
        decoder=PoseDecoder(dim_z, dim_z, dim_z),
        dim_z=dim_z,
    )
    return model

class PoseSeqNetXtreme4(torch.nn.Module):
    def __init__(self, encoder, decoder, shape_pose_encoder, smpl_regressor, dim_z=128):
        super(PoseSeqNetXtreme4, self).__init__()
        self.encoder = encoder
        self.dim_z = dim_z
        self.linear = torch.nn.Linear(1000, self.dim_z)
        self.shape_pose_encoder = shape_pose_encoder
        self.decoder = decoder
        self.smpl_regressor = smpl_regressor
        
    def forward(self, images):
        batch_size = images.shape[0]
        img_size = images.shape[-2:]
        seq_len = images.shape[1]
        images = images.view(batch_size*seq_len, 3,img_size[0],img_size[1])
        features = self.linear(torch.nn.functional.relu(self.encoder(images)))
        betas, poses = self.shape_pose_encoder(features)
        betas = betas.view(batch_size, seq_len, -1) 
        poses = poses.view(batch_size, seq_len, -1) 
        betas = torch.mean(betas, dim=1, keepdim=True).expand(-1, seq_len, -1)
        z = torch.cat((betas, poses), dim=-1)

        verts_sub2, verts_sub, verts_full = self.decoder(z)
        poses, betas = self.smpl_regressor(verts_sub.reshape(seq_len*batch_size, -1))

        betas = betas.reshape(batch_size, seq_len, -1) 
        betas = torch.mean(betas, dim=1, keepdim=True).expand(-1, seq_len, -1)
        betas = betas.reshape(-1, 10)
        return betas, poses, verts_sub2.reshape(-1, 431, 3), verts_sub.reshape(-1, 1723, 3), verts_full.reshape(-1, 6890,3)

def get_model_seq4(dim_z, dim_z_pose , dim_z_shape,  encoder, cfg_hrnet):
    encoder_pretrained = get_encoder(encoder, cfg_hrnet)
    model = PoseSeqNetXtreme4(
        encoder=encoder_pretrained,
        shape_pose_encoder=PoseDecoder(dim_z, dim_z, dim_z),
        decoder=PoseSeqDecoder(dim_z_pose+dim_z_shape),
        dim_z=dim_z,
        smpl_regressor=SMPLParamRegressor(),
    )
    return model


class PoseSeqNetXtreme3(torch.nn.Module):
    def __init__(self, encoder, shape_pose_decoder, dim_z=128):
        super(PoseSeqNetXtreme3, self).__init__()
        self.encoder = encoder
        self.dim_z = dim_z
        self.linear = torch.nn.Linear(1000, self.dim_z)
        self.shape_pose_decoder = shape_pose_decoder
        
    def forward(self, images):
        batch_size = images.shape[0]
        img_size = images.shape[-2:]
        seq_len = images.shape[1]
        images = images.view(batch_size*seq_len, 3,img_size[0],img_size[1])
        features = self.linear(torch.nn.functional.relu(self.encoder(images)))
        betas, poses = self.shape_pose_decoder(features)
        betas = betas.view(batch_size, seq_len, -1) 
        poses = poses.view(batch_size, seq_len, -1) 
        beta = torch.mean(betas, dim=1, keepdim=True).expand(-1, seq_len, -1)
        return beta, poses

def get_model_seq_smpl(dim_z, encoder, cfg_hrnet):
    encoder_pretrained = get_encoder(encoder, cfg_hrnet)
    model = PoseSeqNetXtreme3(
        encoder=encoder_pretrained,
        shape_pose_decoder=PoseDecoder(dim_z, dim_z, dim_z),
        dim_z=dim_z
    )
    return model

class PoseSeqNetXtreme2(torch.nn.Module):
    def __init__(self, encoder, decoder, shape_pose_encoder, dim_z=128):
        super(PoseSeqNetXtreme2, self).__init__()
        self.encoder = encoder
        self.dim_z = dim_z
        self.linear = torch.nn.Linear(1000, self.dim_z)
        self.shape_pose_encoder = shape_pose_encoder
        self.decoder = decoder
        
    def forward(self, images):
        batch_size = images.shape[0]
        img_size = images.shape[-2:]
        seq_len = images.shape[1]
        images = images.view(batch_size*seq_len, 3,img_size[0],img_size[1])
        features = self.linear(torch.nn.functional.relu(self.encoder(images)))
        betas, poses = self.shape_pose_encoder(features)
        betas = betas.view(batch_size, seq_len, -1) 
        poses = poses.view(batch_size, seq_len, -1) 
        betas = torch.mean(betas, dim=1, keepdim=True).expand(-1, seq_len, -1)
        z = torch.cat((betas, poses), dim=-1)
        verts_sub2, verts_sub, verts_full = self.decoder(z)
        return betas, poses, verts_sub2.reshape(-1, 431, 3), verts_sub.reshape(-1, 1723, 3), verts_full.reshape(-1, 6890,3)

def get_model_seq2(dim_z, dim_z_pose , dim_z_shape,  encoder, cfg_hrnet):
    encoder_pretrained = get_encoder(encoder, cfg_hrnet)
    model = PoseSeqNetXtreme2(
        encoder=encoder_pretrained,
        shape_pose_encoder=PoseDecoder(dim_z, dim_z, dim_z),
        decoder=PoseSeqDecoder(dim_z_pose+dim_z_shape),
        dim_z=dim_z
    )
    return model

class PoseSeqNetXtreme(torch.nn.Module):
    def __init__(self, encoder, decoder, dim_z_shape=10, dim_z_pose=72):
        super(PoseSeqNetXtreme, self).__init__()
        self.encoder = encoder
        self.dim_z_shape = dim_z_shape
        self.dim_z_pose = dim_z_pose
        self.linear_shape = torch.nn.Linear(1000, self.dim_z_shape)
        self.linear_pose = torch.nn.Linear(1000, self.dim_z_pose)
        self.decoder = decoder
        
    def forward(self, images):
        batch_size = images.shape[0]
        img_size = images.shape[-2:]
        seq_len = images.shape[1]
        images = images.view(batch_size*seq_len, 3,img_size[0],img_size[1])
        features = torch.nn.functional.relu(self.encoder(images)).view(batch_size, seq_len, -1) 
        betas = self.linear_shape(features) 
        poses = self.linear_pose(features)
        betas = torch.mean(betas, dim=1, keepdim=True).expand(-1, seq_len, -1)
        z = torch.cat((betas, poses), dim=-1)
        verts_sub2, verts_sub, verts_full = self.decoder(z)
        return betas, poses, verts_sub2.reshape(-1, 431, 3), verts_sub.reshape(-1, 1723, 3), verts_full.reshape(-1, 6890,3)

class PoseSeqDecoder(torch.nn.Module):
    def __init__(self, dim_sub_space):
        super(PoseSeqDecoder, self).__init__()
        self.upsampling1 = torch.nn.Linear(dim_sub_space, 431*3) #82*3 -> 431*3
        self.upsampling2 = torch.nn.Linear(431*3, 1723*3) #431*3 -> 1723*3
        self.upsampling3 = torch.nn.Linear(1723*3, 6890*3) #1723*3 -> 6890*3
    def forward(self, z):
        verts_sub2 = self.upsampling1(z) 
        verts_sub = self.upsampling2(verts_sub2) 
        verts_full = self.upsampling3(verts_sub)
        return verts_sub2, verts_sub, verts_full

def get_model_seq(dim_z_pose, dim_z_shape, encoder, cfg_hrnet):
    encoder_pretrained = get_encoder(encoder, cfg_hrnet)
    model = PoseSeqNetXtreme(
        encoder=encoder_pretrained,
        decoder=PoseSeqDecoder(dim_z_pose+dim_z_shape),
        dim_z_shape=dim_z_shape,
        dim_z_pose=dim_z_pose,
    )
    return model

class SMPLParamRegressor(torch.nn.Module):
    def __init__(self):
        super(SMPLParamRegressor, self).__init__()
        # 1723 is the number of vertices in the subsampled SMPL mesh
        self.layers = torch.nn.Sequential(FCBlock(1723 * 3, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    torch.nn.Linear(1024, 24 * 3 * 3 + 10))
    def forward(self, x):
        """Forward pass.
        Input:
            x: size = (B, 1723*6)
        Returns:
            SMPL pose parameters as rotation matrices: size = (B,24,3,3)
            SMPL shape parameters: size = (B,10)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        rotmat = x[:, :24*3*3].view(-1, 24, 3, 3).contiguous()
        betas = x[:, 24*3*3:].contiguous()
        rotmat = rotmat.view(-1, 3, 3).contiguous()
        U, _ , V = torch.linalg.svd(rotmat)
        rotmat = torch.matmul(U, V.transpose(1,2))
<<<<<<< HEAD
        '''det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        with torch.no_grad():
            for i in range(rotmat.shape[0]):
                det[i] = torch.det(rotmat[i])
        rotmat = rotmat * det'''
=======
>>>>>>> 6034b46770d009d779e3730833381d7dd23abe1c
        rotmat = rotmat.view(batch_size, 24, 3, 3)
        return rotmat, betas


class FCBlock(torch.nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=torch.nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [torch.nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(torch.nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = torch.nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.fc_block(x)

class FCResBlock(torch.nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=torch.nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = torch.nn.Sequential(torch.nn.Linear(in_size, out_size),
                                      torch.nn.BatchNorm1d(out_size),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Linear(out_size, out_size),
                                      torch.nn.BatchNorm1d(out_size))
        
    def forward(self, x):
        return F.relu(x + self.fc_block(x))
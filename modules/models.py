import torch
from torch._C import dtype
import torchvision.models as models
from hrnet_model_imgnet.models.cls_hrnet import get_cls_net


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
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1], layers[1:])]
        )
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
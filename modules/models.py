import torch
import torchvision.models as models
from hrnet_model_imgnet.models.cls_hrnet import get_cls_net

resnet_dict = { "resnet18": "models.resnet18(pretrained = True)",
                "resnet34": "models.resnet34(pretrained = True)", 
                "resnet50": "models.resnet50(pretrained = True)", 
                "resnet101": "models.resnet101(pretrained = True)",
                "resnet152": "models.resnet152(pretrained = True)", }

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
        return eval(resnet_dict[encoder])
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
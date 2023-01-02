import os
from models.fields import RenderingNetwork, SDFNetwork
from torch import nn
import torch


class PointLightNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("light", nn.Parameter(torch.tensor(5.0)))

    def forward(self):
        return self.light

    def set_light(self, light):
        self.light.data.fill_(light)

    def get_light(self):
        return self.light.data.clone().detach()


def init_sdf_network_dict():
    sdf_network = SDFNetwork(
        d_in=3,
        d_out=257,
        d_hidden=256,
        n_layers=8,
        skip_in=[4, ],
        multires=6,
        bias=0.5,
        scale=1.0,
        geometric_init=True,
        weight_norm=True,
    ).cuda()
    return sdf_network


def init_rendering_network_dict():
    color_network_dict = {
        "color_network": RenderingNetwork(
            d_in=9,
            d_out=3,
            d_feature=256,
            d_hidden=256,
            n_layers=4,
            multires_view=4,
            mode="idr",
            squeeze_out=True,
        ).cuda(),
        "diffuse_albedo_network": RenderingNetwork(
            d_in=9,
            d_out=3,
            d_feature=256,
            d_hidden=256,
            n_layers=8,
            multires=10,
            multires_view=4,
            mode="idr",
            squeeze_out=True,
            skip_in=(4,),
        ).cuda(),
        "material_network": RenderingNetwork(
            d_in=3,
            d_out=4,
            d_feature=256,
            d_hidden=256,
            n_layers=4,
            multires=6,
            multires_view=-1,
            mode="points_only",
            squeeze_out=False,
            output_bias=0.1,
            output_scale=0.1,
        ).cuda(),
        "specular_albedo_network": RenderingNetwork(
            d_in=6,
            d_out=3,
            d_feature=256,
            d_hidden=256,
            n_layers=4,
            multires=6,
            multires_view=-1,
            mode="no_view_dir",
            squeeze_out=False,
            output_bias=0.4,
            output_scale=0.1,
        ).cuda(),
        "specular_roughness_network": RenderingNetwork(
            d_in=6,
            d_out=1,
            d_feature=256,
            d_hidden=256,
            n_layers=4,
            multires=6,
            multires_view=-1,
            mode="no_view_dir",
            squeeze_out=False,
            output_bias=0.1,
            output_scale=0.1,
        ).cuda(),
        "point_light_network": PointLightNetwork().cuda(),
    }
    return color_network_dict
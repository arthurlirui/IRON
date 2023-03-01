import os
from models.fields import RenderingNetwork, SDFNetwork
from torch import nn
import torch
from models.renderer_ggx import GGXColocatedRenderer

from models.renderer_ggx import RoughPlasticCoLocRenderer
from models.renderer_ggx import SmoothDielectricRenderer
from models.renderer_ggx import RoughConductorCoLocRenderer
from models.renderer_ggx import SmoothConductorCoLocRenderer
from models.renderer_ggx import CoLocRenderer, CompositeRenderer

from models.rendering_func import get_materials_comp, get_materials_multi, get_materials


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
        skip_in=[4],
        multires=6,
        bias=0.5,
        scale=1.0,
        geometric_init=True,
        weight_norm=True,
    ).cuda()
    return sdf_network


def init_rendering_network_dict(renderer_name='comp'):
    if renderer_name == 'ggx':
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
            # "diffuse_albedo_network": RenderingNetwork(
            #     d_in=9,
            #     d_out=3,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=8,
            #     multires=10,
            #     multires_view=4,
            #     mode="idr",
            #     squeeze_out=True,
            #     skip_in=(4,),
            # ).cuda(),
            "diffuse_albedo_network": RenderingNetwork(
                d_in=9,
                d_out=3,
                d_feature=256,
                d_hidden=256,
                n_layers=4,
                multires_view=4,
                mode="idr",
                squeeze_out=True,
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
    elif renderer_name == 'multi':
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
    elif renderer_name == 'comp2':
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
            "metallic_network": RenderingNetwork(
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
            "dielectric_network": RenderingNetwork(
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
            "metallic_eta_network": RenderingNetwork(
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
            "metallic_k_network": RenderingNetwork(
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
            "dielectric_eta_network": RenderingNetwork(
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
            ).cuda()
        }

    elif renderer_name == 'comp':
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
                n_layers=4,
                #multires=10,
                multires_view=4,
                mode="idr",
                squeeze_out=True,
                #skip_in=(4,),
            ).cuda(),

            # "diffuse_albedo_network": RenderingNetwork(
            #     d_in=6,
            #     d_out=3,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=4,
            #     multires=6,
            #     multires_view=-1,
            #     mode="no_view_dir",
            #     squeeze_out=False,
            #     output_bias=0.4,
            #     output_scale=0.1,
            #     skip_in=(4,),
            # ).cuda(),

            # "material_network": RenderingNetwork(
            #     d_in=3,
            #     d_out=4,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=4,
            #     multires=6,
            #     multires_view=-1,
            #     mode="points_only",
            #     squeeze_out=False,
            #     output_bias=0.1,
            #     output_scale=0.1,
            # ).cuda(),
            "specular_albedo_network": RenderingNetwork(
                d_in=9,
                d_out=3,
                d_feature=256,
                d_hidden=256,
                n_layers=4,
                #multires=10,
                multires_view=4,
                mode="idr",
                squeeze_out=True,
                skip_in=(4,),
            ).cuda(),

            # "specular_albedo_network": RenderingNetwork(
            #     d_in=6,
            #     d_out=3,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=4,
            #     multires=6,
            #     multires_view=-1,
            #     mode="no_view_dir",
            #     squeeze_out=False,
            #     output_bias=0.4,
            #     output_scale=0.1,
            #     skip_in=(4,),
            # ).cuda(),
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
            "metallic_eta_network": RenderingNetwork(
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
            "metallic_k_network": RenderingNetwork(
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
            # "env_light_network": RenderingNetwork(
            #     d_in=6,
            #     d_out=1,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=4,
            #     multires=6,
            #     multires_view=-1,
            #     mode="no_view_dir",
            #     squeeze_out=False,
            #     output_bias=0.1,
            #     output_scale=0.1,
            # ).cuda(),
            # "clearcoat_network": RenderingNetwork(
            #     d_in=6,
            #     d_out=1,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=4,
            #     multires=6,
            #     multires_view=-1,
            #     mode="no_view_dir",
            #     squeeze_out=False,
            #     output_bias=0.1,
            #     output_scale=0.1,
            # ).cuda(),
            "metallic_network": RenderingNetwork(
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
            "dielectric_network": RenderingNetwork(
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
            # "spec_tint_network": RenderingNetwork(
            #     d_in=6,
            #     d_out=1,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=4,
            #     multires=6,
            #     multires_view=-1,
            #     mode="no_view_dir",
            #     squeeze_out=False,
            #     output_bias=0.1,
            #     output_scale=0.1,
            # ).cuda(),
            # "anisotropic_network": RenderingNetwork(
            #     d_in=6,
            #     d_out=1,
            #     d_feature=256,
            #     d_hidden=256,
            #     n_layers=4,
            #     multires=6,
            #     multires_view=-1,
            #     mode="no_view_dir",
            #     squeeze_out=False,
            #     output_bias=0.1,
            #     output_scale=0.1,
            # ).cuda(),
            "dielectric_eta_network": RenderingNetwork(
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
        }
    return color_network_dict


def choose_optmizer(renderer_name='comp', network_name_list=[], network_dict={}):
    if renderer_name == 'ggx':
        color_optimizer_dict = {
            "color_network": torch.optim.Adam(network_dict["color_network"].parameters(), lr=1e-4),
            "diffuse_albedo_network": torch.optim.Adam(network_dict["diffuse_albedo_network"].parameters(), lr=1e-4),
            "specular_albedo_network": torch.optim.Adam(network_dict["specular_albedo_network"].parameters(),lr=1e-4),
            "specular_roughness_network": torch.optim.Adam(network_dict["specular_roughness_network"].parameters(), lr=1e-4),
            "point_light_network": torch.optim.Adam(network_dict["point_light_network"].parameters(), lr=1e-2),
        }
        return color_optimizer_dict
    if renderer_name == 'multi':
        color_optimizer_dict = {
            "color_network": torch.optim.Adam(network_dict["color_network"].parameters(), lr=1e-4),
            "diffuse_albedo_network": torch.optim.Adam(network_dict["diffuse_albedo_network"].parameters(), lr=1e-4),
            "specular_albedo_network": torch.optim.Adam(network_dict["specular_albedo_network"].parameters(), lr=1e-4),
            "specular_roughness_network": torch.optim.Adam(network_dict["specular_roughness_network"].parameters(), lr=1e-4),
            "material_network": torch.optim.Adam(network_dict["material_network"].parameters(), lr=1e-4),
            "point_light_network": torch.optim.Adam(network_dict["point_light_network"].parameters(), lr=1e-2),
        }
        return color_optimizer_dict
    if renderer_name == 'comp' or renderer_name == 'comp2':
        color_optimizer_dict = {
            "color_network": torch.optim.Adam(network_dict["color_network"].parameters(), lr=1e-4),
            "diffuse_albedo_network": torch.optim.Adam(network_dict["diffuse_albedo_network"].parameters(), lr=1e-4),
            "specular_albedo_network": torch.optim.Adam(network_dict["specular_albedo_network"].parameters(), lr=1e-4),
            "specular_roughness_network": torch.optim.Adam(network_dict["specular_roughness_network"].parameters(), lr=1e-4),
            "point_light_network": torch.optim.Adam(network_dict["point_light_network"].parameters(), lr=1e-4),
            "metallic_network": torch.optim.Adam(network_dict["metallic_network"].parameters(), lr=1e-4),
            "dielectric_network": torch.optim.Adam(network_dict["dielectric_network"].parameters(), lr=1e-4),
            "metallic_eta_network": torch.optim.Adam(network_dict["metallic_eta_network"].parameters(), lr=1e-4),
            "metallic_k_network": torch.optim.Adam(network_dict["metallic_k_network"].parameters(), lr=1e-4),
            "dielectric_eta_network": torch.optim.Adam(network_dict["dielectric_eta_network"].parameters(), lr=1e-4),
        }
        new_color_optimizer_dict = {}
        if len(network_name_list) > 0:
            for network_name in network_name_list:
                new_color_optimizer_dict[network_name] = color_optimizer_dict[network_name]
            color_optimizer_dict = new_color_optimizer_dict
        return color_optimizer_dict


def choose_renderer(renderer_name='comp'):
    if renderer_name == 'ggx':
        renderer = GGXColocatedRenderer(use_cuda=True)
    if renderer_name == 'multi':
        rough_plastic_renderer = RoughPlasticCoLocRenderer(use_cuda=True)
        dielectric_renderer = SmoothDielectricRenderer(use_cuda=True)
        ior_path = './resource/ior'
        conductor_renderer = SmoothConductorCoLocRenderer(ior_path=ior_path, use_cuda=True)
        rough_conductor_renderer = RoughConductorCoLocRenderer(ior_path=ior_path, use_cuda=True)

        renderer = CoLocRenderer(rough_plastic=rough_plastic_renderer,
                                 dielectric=dielectric_renderer,
                                 smooth_conductor=conductor_renderer,
                                 conductor=rough_conductor_renderer, use_cuda=True)
    if renderer_name == 'comp' or renderer_name == 'comp2':
        renderer = CompositeRenderer(use_cuda=True)
    return renderer


def render_fn_comp(interior_mask, color_network_dict, ray_o, ray_d, points, normals, features):
    # rendering by using composite models
    dots_sh = list(interior_mask.shape)
    rgb = torch.zeros(dots_sh + [3], dtype=torch.float32, device=interior_mask.device)
    diffuse_rgb, specular_rgb = rgb.clone(), rgb.clone()
    diffuse_albedo, specular_albedo = rgb.clone(), rgb.clone()
    metallic_rgb = rgb.clone()
    dielectric_rgb = rgb.clone()

    specular_roughness = rgb[..., 0:1].clone()
    if True:
        clearcoat = specular_roughness.clone()
        metallic = specular_roughness.clone()
        spec_tint = specular_roughness.clone()

    material_vector = torch.zeros(dots_sh + [4], dtype=torch.float32, device=interior_mask.device)
    normals_pad = rgb.clone()

    if interior_mask.any():
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
        #outputs = get_materials_exp(color_network_dict, points, normals, features)
        #interior_diffuse_albedo, interior_specular_albedo, interior_specular_roughness, interior_material_vector = outputs
        params = get_parameter_map(network_dict=color_network_dict, points=points, normals=normals, features=features)
        results = comp_renderer(
            color_network_dict["point_light_network"](),
            (points - ray_o).norm(dim=-1, keepdim=True),
            normals,
            -ray_d,
            params=params
        )

        rgb[interior_mask] = results["rgb"]
        diffuse_rgb[interior_mask] = results["diffuse_rgb"]
        specular_rgb[interior_mask] = results["specular_rgb"]
        metallic_rgb[interior_mask] = results["metallic_rgb"]
        dielectric_rgb[interior_mask] = results["dielectric_rgb"]

        diffuse_albedo[interior_mask] = params['diffuse_albedo']
        specular_albedo[interior_mask] = params['specular_albedo']
        specular_roughness[interior_mask] = params['specular_roughness']
        normals_pad[interior_mask] = normals

        clearcoat[interior_mask] = params['clearcoat']
        metallic[interior_mask] = params['metallic']
        spec_tint[interior_mask] = params['spec_tint']

    return {
        "color": rgb,
        "diffuse_color": diffuse_rgb,
        "specular_color": specular_rgb,
        "diffuse_albedo": diffuse_albedo,
        "specular_albedo": specular_albedo,
        "specular_roughness": specular_roughness,
        "normal": normals_pad,
        "clearcoat": clearcoat,
        "metallic": metallic_rgb,
        "dielectric": dielectric_rgb,
        "spec_tint": spec_tint
    }


def render_fn_exp(interior_mask, color_network_dict, ray_o, ray_d, points, normals, features):
    dots_sh = list(interior_mask.shape)
    rgb = torch.zeros(dots_sh + [3], dtype=torch.float32, device=interior_mask.device)
    diffuse_rgb = rgb.clone()
    specular_rgb = rgb.clone()
    diffuse_albedo = rgb.clone()
    specular_albedo = rgb.clone()
    specular_roughness = rgb[..., 0].clone()
    material_vector = torch.zeros(dots_sh + [4], dtype=torch.float32, device=interior_mask.device)
    normals_pad = rgb.clone()

    if interior_mask.any():
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
        params = get_materials_exp(color_network_dict, points, normals, features)
        #interior_diffuse_albedo, interior_specular_albedo, interior_specular_roughness, interior_material_vector = outputs

        results = full_renderer(
            color_network_dict["point_light_network"](),
            (points - ray_o).norm(dim=-1, keepdim=True),
            normals,
            -ray_d,
            params=params
        )

        rgb[interior_mask] = results["rgb"]
        diffuse_rgb[interior_mask] = results["diffuse_rgb"]
        specular_rgb[interior_mask] = results["specular_rgb"]
        material_vector[interior_mask] = results["material_map"]
        diffuse_albedo[interior_mask] = params["diffuse_albedo"]
        specular_albedo[interior_mask] = params["specular_albedo"]
        specular_roughness[interior_mask] = params['specular_roughness'].squeeze(-1)
        normals_pad[interior_mask] = normals

    return {
        "color": rgb,
        "material_vector": material_vector,
        "diffuse_color": diffuse_rgb,
        "specular_color": specular_rgb,
        "diffuse_albedo": diffuse_albedo,
        "specular_albedo": specular_albedo,
        "specular_roughness": specular_roughness,
        "normal": normals_pad,
    }


def init_outputs(renderer_name='comp', shapes=[], device='cuda:0'):
    if renderer_name == 'multi':
        outputs = {}
        outputs_list1d = ['specular_roughness']
        outputs_list3d = ['rgb', 'diffuse_rgb', 'specular_rgb', 'diffuse_albedo', 'specular_albedo', 'normal']
        outputs_list4d = ['material_vector']
        for key in outputs_list1d:
            outputs[key] = torch.zeros(shapes+[1], dtype=torch.float32, device=device)
        for key in outputs_list3d:
            outputs[key] = torch.zeros(shapes+[3], dtype=torch.float32, device=device)
        for key in outputs_list4d:
            outputs[key] = torch.zeros(shapes+[4], dtype=torch.float32, device=device)
        return outputs
    if renderer_name == 'comp':
        outputs = {}
        outputs_list1d = ['specular_roughness']
        outputs_list3d = ['rgb', 'diffuse_rgb', 'specular_rgb', 'diffuse_albedo', 'specular_albedo', 'normal',
                          'metallic_rgb', 'dielectric_rgb']
        outputs_list4d = []
        for key in outputs_list1d:
            outputs[key] = torch.zeros(shapes + [1], dtype=torch.float32, device=device)
        for key in outputs_list3d:
            outputs[key] = torch.zeros(shapes + [3], dtype=torch.float32, device=device)
        for key in outputs_list4d:
            outputs[key] = torch.zeros(shapes + [4], dtype=torch.float32, device=device)
        return outputs
    if renderer_name == 'ggx':
        outputs = {}
        outputs_list1d = ['specular_roughness']
        outputs_list3d = ['rgb', 'diffuse_rgb', 'specular_rgb', 'diffuse_albedo', 'specular_albedo', 'normal']
        outputs_list4d = []
        for key in outputs_list1d:
            outputs[key] = torch.zeros(shapes + [1], dtype=torch.float32, device=device)
        for key in outputs_list3d:
            outputs[key] = torch.zeros(shapes + [3], dtype=torch.float32, device=device)
        for key in outputs_list4d:
            outputs[key] = torch.zeros(shapes + [4], dtype=torch.float32, device=device)
        return outputs


def choose_renderer_func(renderer_name='comp', renderer=None, device='cuda:0'):
    renderer = choose_renderer(renderer_name)
    if renderer_name == 'multi':
        get_material_fn = get_materials_multi
    elif renderer_name == 'comp':
        get_material_fn = get_materials_comp
    elif renderer_name == 'ggx':
        get_material_fn = get_materials
    else:
        get_material_fn = get_materials

    def render_fn_comp(interior_mask, color_network_dict, ray_o, ray_d, points, normals, features):
        # rendering by using composite models
        dots_sh = list(interior_mask.shape)
        rgb = torch.zeros(dots_sh + [3], dtype=torch.float32, device=interior_mask.device)
        diffuse_rgb, specular_rgb = rgb.clone(), rgb.clone()
        diffuse_albedo, specular_albedo = rgb.clone(), rgb.clone()
        metallic_rgb = rgb.clone()
        dielectric_rgb = rgb.clone()

        specular_roughness = rgb[..., 0:1].clone()
        if True:
            clearcoat = specular_roughness.clone()
            metallic = specular_roughness.clone()
            spec_tint = specular_roughness.clone()

        #material_vector = torch.zeros(dots_sh + [4], dtype=torch.float32, device=interior_mask.device)
        normals_pad = rgb.clone()

        if interior_mask.any():
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
            # outputs = get_materials_exp(color_network_dict, points, normals, features)
            # interior_diffuse_albedo, interior_specular_albedo, interior_specular_roughness, interior_material_vector = outputs
            params = get_material_fn(network_dict=color_network_dict, points=points, normals=normals,
                                       features=features)
            results = renderer(
                color_network_dict["point_light_network"](),
                (points - ray_o).norm(dim=-1, keepdim=True),
                normals,
                -ray_d,
                params=params
            )

            rgb[interior_mask] = results["rgb"]
            diffuse_rgb[interior_mask] = results["diffuse_rgb"]
            specular_rgb[interior_mask] = results["specular_rgb"]
            metallic_rgb[interior_mask] = results["metallic_rgb"]
            dielectric_rgb[interior_mask] = results["dielectric_rgb"]

            diffuse_albedo[interior_mask] = params['diffuse_albedo']
            specular_albedo[interior_mask] = params['specular_albedo']
            specular_roughness[interior_mask] = params['specular_roughness']
            normals_pad[interior_mask] = normals

            clearcoat[interior_mask] = params['clearcoat']
            metallic[interior_mask] = params['metallic']
            spec_tint[interior_mask] = params['spec_tint']

        return {
            "color": rgb,
            "diffuse_color": diffuse_rgb,
            "specular_color": specular_rgb,
            "diffuse_albedo": diffuse_albedo,
            "specular_albedo": specular_albedo,
            "specular_roughness": specular_roughness,
            "normal": normals_pad,
            "clearcoat": clearcoat,
            "metallic": metallic_rgb,
            "dielectric": dielectric_rgb,
            "spec_tint": spec_tint
        }

    def render_fn(interior_mask, network_dict, ray_o, ray_d, points, normals, features):
        dots_sh = list(interior_mask.shape)
        outputs = init_outputs(renderer_name=renderer_name, shapes=dots_sh, device=device)
        if interior_mask.any():
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
            params = get_material_fn(network_dict, points, normals, features)
            results = renderer(
                network_dict["point_light_network"](),
                (points - ray_o).norm(dim=-1, keepdim=True),
                normals,
                -ray_d,
                params=params)

            for key in results:
                if key in outputs:
                    outputs[key][interior_mask] = results[key]
            for key in params:
                if key in outputs:
                    outputs[key][interior_mask] = params[key]
            outputs['normal'][interior_mask] = normals
            return outputs
    return render_fn



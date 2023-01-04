import os
import torch

###### rendering functions
def get_materials(color_network_dict, points, normals, features, is_metal=False):
    diffuse_albedo = color_network_dict["diffuse_albedo_network"](points, normals, -normals, features).abs()[
        ..., [2, 1, 0]
    ]
    specular_albedo = color_network_dict["specular_albedo_network"](points, normals, None, features).abs()
    if not is_metal:
        specular_albedo = torch.mean(specular_albedo, dim=-1, keepdim=True).expand_as(specular_albedo)
    specular_roughness = color_network_dict["specular_roughness_network"](points, normals, None, features).abs() + 0.01
    return diffuse_albedo, specular_albedo, specular_roughness


def get_parameter_map(network_dict, points, normals, features):
    res = {}
    diffuse_albedo = network_dict["diffuse_albedo_network"](points, normals, -normals, features).abs()[..., [2, 1, 0]]
    specular_albedo = network_dict["specular_albedo_network"](points, normals, None, features).abs()
    clearcoat = network_dict["clearcoat"](points, normals, None, features).abs()
    metallic = network_dict["metallic"]()(points, normals, None, features).abs()
    spec_tint = network_dict["spec_tint"]()(points, normals, None, features).abs()
    specular_roughness = network_dict["specular_roughness"]()(points, normals, None, features).abs()
    material_vector = network_dict["material_network"](points, None, None, features).abs()
    res['diffuse_albedo'] = diffuse_albedo
    res['specular_albedo'] = specular_albedo
    res['clearcoat'] = clearcoat
    res['metallic'] = metallic
    res['spec_tint'] = spec_tint
    res['specular_roughness'] = specular_roughness
    res['material_vector'] = material_vector
    return res


def get_materials_exp(color_network_dict, points, normals, features, is_metal=False):
    diffuse_albedo = color_network_dict["diffuse_albedo_network"](points, normals, -normals, features).abs()[
        ..., [2, 1, 0]
    ]
    specular_albedo = color_network_dict["specular_albedo_network"](points, normals, None, features).abs()
    #if not is_metal:
    #    specular_albedo = torch.mean(specular_albedo, dim=-1, keepdim=True).expand_as(specular_albedo)
    specular_roughness = color_network_dict["specular_roughness_network"](points, normals, None, features).abs() + 0.01
    material_vector = color_network_dict["material_network"](points, None, None, features).abs()
    return diffuse_albedo, specular_albedo, specular_roughness, material_vector


def render_fn(interior_mask, color_network_dict, ray_o, ray_d, points, normals, features, ggx_renderer):
    dots_sh = list(interior_mask.shape)
    rgb = torch.zeros(dots_sh + [3], dtype=torch.float32, device=interior_mask.device)
    diffuse_rgb = rgb.clone()
    specular_rgb = rgb.clone()
    diffuse_albedo = rgb.clone()
    specular_albedo = rgb.clone()
    specular_roughness = rgb[..., 0].clone()
    material_tensor = torch.zeros(dots_sh + [4], dtype=torch.float32, device=interior_mask.device)
    normals_pad = rgb.clone()

    dielectric_specular = rgb.clone()
    conduct_specular = rgb.clone()

    if interior_mask.any():
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
        # interior_diffuse_albedo, interior_specular_albedo, interior_specular_roughness = get_materials(
        #     color_network_dict, points, normals, features
        # )
        outputs = get_materials_exp(color_network_dict, points, normals, features)
        interior_diffuse_albedo, interior_specular_albedo, interior_specular_roughness, material_vector = outputs
        results = ggx_renderer(
            color_network_dict["point_light_network"](),
            (points - ray_o).norm(dim=-1, keepdim=True),
            normals,
            -ray_d,
            interior_diffuse_albedo,
            interior_specular_albedo,
            interior_specular_roughness,
        )

        # results_dielectric = dielectric_renderer(
        #     color_network_dict["point_light_network"](),
        #     (points - ray_o).norm(dim=-1, keepdim=True),
        #     normals,
        #     -ray_d,
        #     interior_diffuse_albedo,
        #     interior_specular_albedo,
        #     interior_specular_roughness,
        # )
        #
        # results_conduct = conduct_renderer(
        #     color_network_dict["point_light_network"](),
        #     (points - ray_o).norm(dim=-1, keepdim=True),
        #     normals,
        #     -ray_d,
        #     interior_diffuse_albedo,
        #     interior_specular_albedo,
        #     interior_specular_roughness,
        # )
        #
        # dielectric_specular[interior_mask] = results_dielectric["specular_rgb"]
        # conduct_specular[interior_mask] = results_conduct["specular_rgb"]

        rgb[interior_mask] = results["rgb"]
        diffuse_rgb[interior_mask] = results["diffuse_rgb"]
        specular_rgb[interior_mask] = results["specular_rgb"]
        diffuse_albedo[interior_mask] = interior_diffuse_albedo
        specular_albedo[interior_mask] = interior_specular_albedo
        specular_roughness[interior_mask] = interior_specular_roughness.squeeze(-1)
        normals_pad[interior_mask] = normals

    return {
        "color": rgb,
        "dielectric_specular": dielectric_specular,
        "conduct_specular": conduct_specular,
        "diffuse_color": diffuse_rgb,
        "specular_color": specular_rgb,
        "diffuse_albedo": diffuse_albedo,
        "specular_albedo": specular_albedo,
        "specular_roughness": specular_roughness,
        "normal": normals_pad,
    }



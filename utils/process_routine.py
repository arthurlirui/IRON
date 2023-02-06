import os
import tqdm
from models.raytracer import RayTracer, Camera, render_camera
from icecream import ic
from models.helper import gamma_correction, inv_gamma_correction
import numpy as np
import torch
from models.dataset import to8b


def render_all(out_dir, cameras, raytracer, image_fpaths, sdf_network, color_network_dict, render_fn, gamma_pred, imwriter):
    render_out_dir = os.path.join(out_dir, f"render_{os.path.basename(args.data_dir)}_{start_step}")
    os.makedirs(render_out_dir, exist_ok=True)
    ic(f"Rendering images to: {render_out_dir}")
    n_cams = len(cameras)
    for i in tqdm.tqdm(range(n_cams)):
        cam, impath = cameras[i], image_fpaths[i]
        results = render_camera(
            cam,
            sdf_network,
            raytracer,
            color_network_dict,
            render_fn=render_fn,
            fill_holes=True,
            handle_edges=True,
            is_training=False,
        )
        if gamma_pred:
            results["color"] = gamma_correction(results["color"], gamma=2.2)
            results["diffuse_color"] = gamma_correction(results["diffuse_color"], gamma=2.2)
            #results["color"] = torch.pow(results["color"] + 1e-6, 1.0 / 2.2)
            #results["diffuse_color"] = torch.pow(results["diffuse_color"] + 1e-6, 1.0 / 2.2)
            results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)
        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()
        color_im = results["color"]
        timgname = os.path.basename(impath).split('.')[0]
        #imageio.imwrite(os.path.join(render_out_dir, timgname + '.jpg'), to8b(color_im))
        imwriter(os.path.join(render_out_dir, timgname + '.jpg'), to8b(color_im))

        normal = results["normal"]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
        normal_im = (normal + 1.0) / 2.0
        #imageio.imwrite(os.path.join(render_out_dir, timgname + '_normal.jpg'), to8b(normal_im))
        imwriter(os.path.join(render_out_dir, timgname + '_normal.jpg'), to8b(normal_im))
        diff_im = results["diffuse_color"]
        #imageio.imwrite(os.path.join(render_out_dir, timgname + '_diff.jpg'), to8b(diff_im))
        imwriter(os.path.join(render_out_dir, timgname + '_diff.jpg'), to8b(diff_im))
        specular_im = results["specular_color"]
        #imageio.imwrite(os.path.join(render_out_dir, timgname + '_specular.jpg'), to8b(specular_im))
        imwriter(os.path.join(render_out_dir, timgname + '_specular.jpg'), to8b(specular_im))


def train():
    pass
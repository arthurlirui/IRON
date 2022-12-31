import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import imageio
imageio.plugins.freeimage.download()
from torch.utils.tensorboard import SummaryWriter
import configargparse
from icecream import ic
import glob
import traceback

from models.network_conf import init_sdf_network_dict, init_rendering_network_dict
from models.raytracer import RayTracer, Camera, render_camera
from models.renderer_ggx import GGXColocatedRenderer

from models.renderer_ggx import RoughPlasticCoLocRenderer
from models.renderer_ggx import SmoothDielectricRenderer
from models.renderer_ggx import RoughConductorCoLocRenderer
from models.renderer_ggx import SmoothConductorCoLocRenderer
from models.renderer_ggx import CoLocRenderer

from models.image_losses import PyramidL2Loss, ssim_loss_fn
from models.export_mesh import export_mesh, export_mesh_no_translation
from models.export_materials import export_materials
from models.rendering_func import get_materials, get_materials_exp

import kornia
from models.dataset import image_reader, image_writer, exr_writer, exr_reader
from models.dataset import to8b, load_dataset_NIRRGB, load_datadir

from models.helper import gamma_correction, inv_gamma_correction
from models.dataset import image_writer, image_reader

#from torchmetrics.functional import image_gradients
###### arguments
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="input data directory")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--folder_name", type=str, default="image", help="dataset image folder")
    parser.add_argument("--neus_ckpt_fpath", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--num_iters", type=int, default=100001, help="number of iterations")
    parser.add_argument("--patch_size", type=int, default=128, help="width and height of the rendered patches")
    parser.add_argument("--eik_weight", type=float, default=0.1, help="weight for eikonal loss")
    parser.add_argument("--ssim_weight", type=float, default=1.0, help="weight for ssim loss")
    parser.add_argument("--roughrange_weight", type=float, default=0.1, help="weight for roughness range loss")

    parser.add_argument("--plot_image_name", type=str, default=None, help="image to plot during training")
    parser.add_argument("--no_edgesample", action="store_true", help="whether to disable edge sampling")
    parser.add_argument(
        "--inv_gamma_gt", action="store_true", help="whether to inverse gamma correct the ground-truth photos"
    )
    parser.add_argument("--gamma_pred", action="store_true", help="whether to gamma correct the predictions")
    parser.add_argument(
        "--is_metal",
        action="store_true",
        help="whether the object of interest is made of metals or the scene contains metals",
    )
    parser.add_argument("--init_light_scale", type=float, default=8.0, help="scaling parameters for light")
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="whether to export meshes and uv textures",
    )
    parser.add_argument(
        "--render_all",
        action="store_true",
        help="whether to render the input image set",
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser


parser = config_parser()
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
ic(args)

###### back up arguments and code scripts
os.makedirs(args.out_dir, exist_ok=True)
parser.write_config_file(args, [os.path.join(args.out_dir, "args.txt")])

raytracer = RayTracer()
sdf_network = init_sdf_network_dict()
color_network_dict = init_rendering_network_dict()

sdf_optimizer = torch.optim.Adam(sdf_network.parameters(), lr=1e-5)
color_optimizer_dict = {
    "color_network": torch.optim.Adam(color_network_dict["color_network"].parameters(), lr=1e-4),
    "diffuse_albedo_network": torch.optim.Adam(color_network_dict["diffuse_albedo_network"].parameters(), lr=1e-4),
    "specular_albedo_network": torch.optim.Adam(color_network_dict["specular_albedo_network"].parameters(), lr=1e-4),
    "specular_roughness_network": torch.optim.Adam(color_network_dict["specular_roughness_network"].parameters(), lr=1e-4),
    "material_network": torch.optim.Adam(color_network_dict["material_network"].parameters(), lr=1e-2),
    "point_light_network": torch.optim.Adam(color_network_dict["point_light_network"].parameters(), lr=1e-2),
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
        outputs = get_materials_exp(color_network_dict, points, normals, features)
        interior_diffuse_albedo, interior_specular_albedo, interior_specular_roughness, interior_material_vector = outputs

        results = full_renderer(
            color_network_dict["point_light_network"](),
            (points - ray_o).norm(dim=-1, keepdim=True),
            normals,
            -ray_d,
            interior_diffuse_albedo,
            interior_specular_albedo,
            interior_specular_roughness,
            interior_material_vector
        )

        rgb[interior_mask] = results["rgb"]
        diffuse_rgb[interior_mask] = results["diffuse_rgb"]
        specular_rgb[interior_mask] = results["specular_rgb"]
        material_vector[interior_mask] = results["material_map"]
        diffuse_albedo[interior_mask] = interior_diffuse_albedo
        specular_albedo[interior_mask] = interior_specular_albedo
        specular_roughness[interior_mask] = interior_specular_roughness.squeeze(-1)
        normals_pad[interior_mask] = normals

    return {
        "color": rgb,
        "material_map": material_vector,
        "diffuse_color": diffuse_rgb,
        "specular_color": specular_rgb,
        "diffuse_albedo": diffuse_albedo,
        "specular_albedo": specular_albedo,
        "specular_roughness": specular_roughness,
        "normal": normals_pad,
    }

###### loss specifications
ggx_renderer = GGXColocatedRenderer(use_cuda=True)
rough_plastic_renderer = RoughPlasticCoLocRenderer(use_cuda=True)
dielectric_renderer = SmoothDielectricRenderer(use_cuda=True)
#thin_dielectric_renderer = ThinDielectricRenderer(use_cuda=True)
ior_path = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/mitsuba-data/ior'
conductor_renderer = SmoothConductorCoLocRenderer(ior_path=ior_path, use_cuda=True)
rought_conductor_renderer = RoughConductorCoLocRenderer(ior_path=ior_path, use_cuda=True)

full_renderer = CoLocRenderer(rough_plastic=rough_plastic_renderer,
                              dielectric=dielectric_renderer,
                              smooth_conductor=conductor_renderer,
                              conductor=rought_conductor_renderer, use_cuda=True)

pyramidl2_loss_fn = PyramidL2Loss(use_cuda=True)

#image_fpaths, gt_images, Ks, W2Cs = load_datadir(args.data_dir, args.folder_name)
#RGB_fpaths, RGB_gt_images, RGB_Ks, RGB_W2Cs = load_datadir(args.data_dir, folder_name='rgb')
#image_fpaths, gt_images, Ks, W2Cs = load_datadir(args.data_dir, folder_name='nir')
image_fpaths, gt_images, Ks, W2Cs = load_dataset_NIRRGB(args.data_dir, folder_name='nir')
#image_fpaths, gt_images, Ks, W2Cs = load_datadir(args.data_dir, args.folder_name)
cameras = [
    Camera(W=gt_images[i].shape[1], H=gt_images[i].shape[0], K=Ks[i].cuda(), W2C=W2Cs[i].cuda())
    for i in range(gt_images.shape[0])
]
ic(len(image_fpaths), gt_images.shape, Ks.shape, W2Cs.shape, len(cameras))

###### initialization using neus
ic(args.neus_ckpt_fpath)
if os.path.isfile(args.neus_ckpt_fpath):
    ic(f"Loading from neus checkpoint: {args.neus_ckpt_fpath}")
    ckpt = torch.load(args.neus_ckpt_fpath, map_location=torch.device("cuda"))
    try:
        sdf_network.load_state_dict(ckpt["sdf_network_fine"])
        color_network_dict["diffuse_albedo_network"].load_state_dict(ckpt["color_network_fine"])
    except:
        traceback.print_exc()
        # ic("Failed to initialize diffuse_albedo_network from checkpoint: ", args.neus_ckpt_fpath)
dist = np.median([torch.norm(cameras[i].get_camera_origin()).item() for i in range(len(cameras))])
init_light = args.init_light_scale * dist * dist
color_network_dict["point_light_network"].set_light(init_light)

#### load pretrained checkpoints
start_step = -1
ckpt_fpaths = glob.glob(os.path.join(args.out_dir, "ckpt_*.pth"))
if len(ckpt_fpaths) > 0:
    path2step = lambda x: int(os.path.basename(x)[len("ckpt_"): -4])
    ckpt_fpaths = sorted(ckpt_fpaths, key=path2step)
    ckpt_fpath = ckpt_fpaths[-1]
    start_step = path2step(ckpt_fpath)
    ic("Reloading from checkpoint: ", ckpt_fpath)
    ckpt = torch.load(ckpt_fpath, map_location=torch.device("cuda"))
    sdf_network.load_state_dict(ckpt["sdf_network"])
    for x in list(color_network_dict.keys()):
        #print(x)
        color_network_dict[x].load_state_dict(ckpt[x])
    # logim_names = [os.path.basename(x) for x in glob.glob(os.path.join(args.out_dir, "logim_*.png"))]
    # start_step = sorted([int(x[len("logim_") : -4]) for x in logim_names])[-1]
ic(dist, color_network_dict["point_light_network"].light.data)
ic(start_step)


###### export mesh and materials
blender_fpath = "./blender-3.1.0-linux-x64/blender"
if not os.path.isfile(blender_fpath):
    os.system(
        "wget https://mirror.clarkson.edu/blender/release/Blender3.1/blender-3.1.0-linux-x64.tar.xz && \
             tar -xvf blender-3.1.0-linux-x64.tar.xz"
    )


def export_mesh_and_materials(export_out_dir, sdf_network, color_network_dict):
    ic(f"Exporting mesh and materials to: {export_out_dir}")
    sdf_fn = lambda x: sdf_network(x)[..., 0]
    ic("Exporting mesh and uv...")
    with torch.no_grad():
        if True:
            mesh_name = 'mesh.obj'
            export_mesh(sdf_fn, os.path.join(export_out_dir, mesh_name))
            os.system(
                f"{blender_fpath} --background --python models/export_uv.py {os.path.join(export_out_dir, mesh_name)} {os.path.join(export_out_dir, mesh_name)}"
            )
        if False:
            mesh_name = 'mesh_no_translation.obj'
            export_mesh_no_translation(sdf_fn, os.path.join(export_out_dir, mesh_name))
            os.system(f"{blender_fpath} --background --python models/export_uv.py {os.path.join(export_out_dir, mesh_name)} {os.path.join(export_out_dir, mesh_name)}")

    class MaterialPredictor(nn.Module):
        def __init__(self, sdf_network, color_network_dict):
            super().__init__()
            self.sdf_network = sdf_network
            self.color_network_dict = color_network_dict

        def forward(self, points):
            _, features, normals = self.sdf_network.get_all(points, is_training=False)
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
            diffuse_albedo, specular_albedo, specular_roughness = get_materials(
                color_network_dict, points, normals, features
            )
            return diffuse_albedo, specular_albedo, specular_roughness

    ic("Exporting materials...")
    material_predictor = MaterialPredictor(sdf_network, color_network_dict)
    with torch.no_grad():
        export_materials(os.path.join(export_out_dir, mesh_name), material_predictor, export_out_dir)

    ic(f"Exported mesh and materials to: {export_out_dir}")


if args.export_all:
    export_out_dir = os.path.join(args.out_dir, f"mesh_and_materials_{start_step}")
    os.makedirs(export_out_dir, exist_ok=True)
    export_mesh_and_materials(export_out_dir, sdf_network, color_network_dict)
    exit(0)


###### render all images
if args.render_all:
    render_out_dir = os.path.join(args.out_dir, f"render_{os.path.basename(args.data_dir)}_{start_step}")
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
            render_fn_exp,
            fill_holes=True,
            handle_edges=True,
            is_training=False,
        )
        if args.gamma_pred:
            results["color"] = gamma_correction(results["color"], g=2.2)
            results["diffuse_color"] = gamma_correction(results["diffuse_color"], g=2.2)
            #results["color"] = torch.pow(results["color"] + 1e-6, 1.0 / 2.2)
            #results["diffuse_color"] = torch.pow(results["diffuse_color"] + 1e-6, 1.0 / 2.2)
            results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)
        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()
        color_im = results["color"]
        timgname = os.path.basename(impath).split('.')[0]
        imageio.imwrite(os.path.join(render_out_dir, timgname + '.jpg'), to8b(color_im))

        if True:
            normal = results["normal"]
            normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
            normal_im = (normal + 1.0) / 2.0
            imageio.imwrite(os.path.join(render_out_dir, timgname + '_normal.jpg'), to8b(normal_im))
            diff_im = results["diffuse_color"]
            imageio.imwrite(os.path.join(render_out_dir, timgname + '_diff.jpg'), to8b(diff_im))
            specular_im = results["specular_color"]
            imageio.imwrite(os.path.join(render_out_dir, timgname + '_specular.jpg'), to8b(specular_im))
    exit(0)

###### training
fill_holes = False
handle_edges = not args.no_edgesample
is_training = True
if args.inv_gamma_gt:
    ic("linearizing ground-truth images using inverse gamma correction")
    gt_images = torch.pow(gt_images, 2.2)

ic(fill_holes, handle_edges, is_training, args.inv_gamma_gt)
writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))

global_step = args.num_iters

for global_step in tqdm.tqdm(range(start_step + 1, args.num_iters)):
    sdf_optimizer.zero_grad()
    for x in color_optimizer_dict.keys():
        color_optimizer_dict[x].zero_grad()

    idx = np.random.randint(0, gt_images.shape[0])
    camera_crop, gt_color_crop = cameras[idx].crop_region(trgt_W=args.patch_size, trgt_H=args.patch_size, image=gt_images[idx])

    results = render_camera(
        camera_crop,
        sdf_network,
        raytracer,
        color_network_dict,
        render_fn_exp,
        fill_holes=fill_holes,
        handle_edges=handle_edges,
        is_training=is_training,
    )
    if args.gamma_pred:
        #results["color"] = torch.pow(results["color"] + 1e-6, 1.0 / 2.2)
        #results["diffuse_color"] = torch.pow(results["diffuse_color"] + 1e-6, 1.0 / 2.2)
        results["color"] = gamma_correction(results["color"])
        results["diffuse_color"] = gamma_correction(results["diffuse_color"])
        results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)

        w = (results['distance'] / (init_light * torch.sum(results['normal'] * results['ray_d'], dim=-1, keepdim=False) + 1e-6))
        f = results["color"] * torch.stack([w, w, w], dim=-1)

    mask = results["convergent_mask"]
    if handle_edges:
        mask = mask | results["edge_mask"]

    img_loss = torch.Tensor([0.0]).cuda()
    img_l2_loss = torch.Tensor([0.0]).cuda()
    img_ssim_loss = torch.Tensor([0.0]).cuda()
    roughrange_loss = torch.Tensor([0.0]).cuda()
    material_type_loss = torch.Tensor([0.0]).cuda()

    eik_points = torch.empty(camera_crop.H * camera_crop.W // 2, 3).cuda().float().uniform_(-1.0, 1.0)
    eik_grad = sdf_network.gradient(eik_points).view(-1, 3)
    eik_cnt = eik_grad.shape[0]
    eik_loss = ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

    #dy, dx = image_gradients(f)

    #sparse_loss = dy.norm(1, dim=-1)+dx.norm(1, dim=-1)
    out_gradient = kornia.filters.spatial_gradient(f.permute(2, 0, 1).unsqueeze(0), mode='sobel', order=1, normalized=True)
    #print(out_gradient.shape, f.shape)
    sparse_loss = out_gradient.norm(1)/torch.numel(out_gradient)
    #print(sparse_loss)
    if mask.any():
        pred_img = results["color"].permute(2, 0, 1).unsqueeze(0)
        gt_img = gt_color_crop.permute(2, 0, 1).unsqueeze(0).to(pred_img.device, dtype=pred_img.dtype)
        #print(pred_img.shape, gt_img.shape)
        pred_img = pred_img[:, :3, :, :]
        gt_img = gt_img[:, :3, :, :]
        img_l2_loss = pyramidl2_loss_fn(pred_img, gt_img)
        img_ssim_loss = args.ssim_weight * ssim_loss_fn(pred_img, gt_img, mask.unsqueeze(0).unsqueeze(0))
        img_loss = img_l2_loss + img_ssim_loss

        eik_grad = results["normal"][mask]
        eik_cnt += eik_grad.shape[0]
        eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()
        if "edge_pos_neg_normal" in results:
            eik_grad = results["edge_pos_neg_normal"]
            eik_cnt += eik_grad.shape[0]
            eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

        roughness = results["specular_roughness"][mask]
        roughness = roughness[roughness > 0.5]
        if roughness.numel() > 0:
            roughrange_loss = (roughness - 0.5).mean() * args.roughrange_weight

        # constraint for material map
        material_type_loss = torch.norm(torch.sum(torch.abs(results["material_map"]), dim=-1) - 1, p=2)
    eik_loss = eik_loss / eik_cnt * args.eik_weight

    loss = img_loss + eik_loss + roughrange_loss + 0.1 * sparse_loss + material_type_loss

    loss.backward()
    sdf_optimizer.step()
    for x in color_optimizer_dict.keys():
        color_optimizer_dict[x].step()

    if global_step % 50 == 0:
        writer.add_scalar("loss/loss", loss, global_step)
        writer.add_scalar("loss/img_loss", img_loss, global_step)
        writer.add_scalar("loss/img_l2_loss", img_l2_loss, global_step)
        writer.add_scalar("loss/img_ssim_loss", img_ssim_loss, global_step)
        writer.add_scalar("loss/eik_loss", eik_loss, global_step)
        writer.add_scalar("loss/roughrange_loss", roughrange_loss, global_step)
        writer.add_scalar("loss/sparse_loss", sparse_loss, global_step)
        writer.add_scalar("loss/material_type_loss", material_type_loss, global_step)
        writer.add_scalar("light", color_network_dict["point_light_network"].get_light())

    if global_step % 1000 == 0:
        torch.save(
            dict(
                [
                    ("sdf_network", sdf_network.state_dict()),
                ]
                + [(x, color_network_dict[x].state_dict()) for x in color_network_dict.keys()]
            ),
            os.path.join(args.out_dir, f"ckpt_{global_step}.pth"),
        )

    if global_step % 500 == 0:
        ic(
            args.out_dir,
            global_step,
            loss.item(),
            img_loss.item(),
            img_l2_loss.item(),
            img_ssim_loss.item(),
            eik_loss.item(),
            roughrange_loss.item(),
            sparse_loss.item(),
            color_network_dict["point_light_network"].get_light().item(),
        )

        for x in list(results.keys()):
            del results[x]

        idx = 0
        if args.plot_image_name is not None:
            while idx < len(image_fpaths):
                if args.plot_image_name in image_fpaths[idx]:
                    break
                idx += 1

        camera_resize, gt_color_resize = cameras[idx].resize(factor=0.25, image=gt_images[idx])
        results = render_camera(
            camera_resize,
            sdf_network,
            raytracer,
            color_network_dict,
            render_fn_exp,
            fill_holes=fill_holes,
            handle_edges=handle_edges,
            is_training=False,
        )
        if args.gamma_pred:
            results["color"] = torch.pow(results["color"] + 1e-6, 1.0 / 2.2)
            results["diffuse_color"] = torch.pow(results["diffuse_color"] + 1e-6, 1.0 / 2.2)
            results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)

            results["material_map"] = torch.clamp(results["material_map"], min=0.0, max=10.0)

            w = (results['distance'] / (torch.sum(results['normal'] * results['ray_d'], dim=-1, keepdim=False) + 1))
            f = results["color"] * torch.stack([w, w, w], dim=-1)

        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()

        gt_color_im = gt_color_resize.detach().cpu().numpy()
        nir_sparse = f.detach().cpu().numpy()
        color_im = results["color"]
        diffuse_color_im = results["diffuse_color"]
        specular_color_im = results["specular_color"]
        normal = results["normal"]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
        normal_im = (normal + 1.0) / 2.0
        edge_mask_im = np.tile(results["edge_mask"][:, :, np.newaxis], (1, 1, 3))
        diffuse_albedo_im = results["diffuse_albedo"]
        specular_albedo_im = results["specular_albedo"]
        specular_roughness_im = np.tile(results["specular_roughness"][:, :, np.newaxis], (1, 1, 3))

        rough_plastic_map = results["material_map"][..., 0]
        dielectric_map = results["material_map"][..., 1]
        rough_conductor_map = results["material_map"][..., 2]
        smooth_conductor_map = results["material_map"][..., 3]

        if args.inv_gamma_gt:
            gt_color_im = np.power(gt_color_im + 1e-6, 1.0 / 2.2)
            color_im = np.power(color_im + 1e-6, 1.0 / 2.2)
            diffuse_color_im = np.power(diffuse_color_im + 1e-6, 1.0 / 2.2)
            specular_color_im = color_im - diffuse_color_im
            nir_sparse = np.power(nir_sparse + 1e-6, 1.0 / 2.2)

            rough_plastic_map = np.power(rough_plastic_map + 1e-6, 1.0/2.2)
            dielectric_map = np.power(dielectric_map + 1e-6, 1.0/2.2)
            rough_conductor_map = np.power(rough_conductor_map + 1e-6, 1.0/2.2)
            smooth_conductor_map = np.power(smooth_conductor_map + 1e-6, 1.0/2.2)

        material_map3 = np.stack([rough_plastic_map, dielectric_map, rough_conductor_map], axis=-1)

        gt_color_im = gt_color_im[..., :3]
        normal_im = normal_im[..., :3]
        #row1 = np.concatenate([gt_color_im, normal_im, edge_mask_im], axis=1)
        #row2 = np.concatenate([color_im, diffuse_color_im, specular_color_im], axis=1)

        #row2 = np.concatenate([color_im, diffuse_color_im, material_map3], axis=1)

        #row3 = np.concatenate([diffuse_albedo_im, specular_albedo_im, specular_roughness_im], axis=1)

        #row4 = np.concatenate([rough_plastic_map, dielectric_map, rough_conductor_map], axis=1)
        #row4 = np.stack([row4, row4, row4], axis=-1)

        #im = np.concatenate((row1, row2, row3, row4), axis=0)
        from models.helper import concatenate_result
        img_list = [gt_color_im, normal_im, edge_mask_im,
                    color_im, diffuse_color_im, material_map3,
                    rough_plastic_map, dielectric_map, rough_conductor_map]
        im = concatenate_result(image_list=img_list, imarray_length=3)
        imageio.imwrite(os.path.join(args.out_dir, f"logim_{global_step}.png"), to8b(im))


###### export mesh and materials
export_out_dir = os.path.join(args.out_dir, f"mesh_and_materials_{global_step}")
os.makedirs(export_out_dir, exist_ok=True)
export_mesh_and_materials(export_out_dir, sdf_network, color_network_dict)

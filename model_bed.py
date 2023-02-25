import os
import torch
from models.image_losses import PyramidL2Loss, ssim_loss_fn
from models.dataset import load_dataset_general
from models.raytracer import RayTracer, Camera, render_camera
from icecream import ic
from utils.ckpt_loader import load_neus_checkpoint, load_pretrained_checkpoint
from models.network_conf import init_sdf_network_dict, init_rendering_network_dict, choose_optmizer, choose_renderer
import numpy as np
from models.helper import gamma_correction, inv_gamma_correction
from torch.utils.tensorboard import SummaryWriter
from models.helper import concatenate_result
from models.dataset import image_writer, image_reader, to8b
from models.export_mesh import export_mesh, export_mesh_no_translation
from utils.ckpt_loader import load_ckpt, load_neus_checkpoint, load_pretrained_checkpoint, download_blender
from utils.process_routine import render_all
import tqdm
import configargparse
from models.network_conf import choose_optmizer, choose_renderer, choose_renderer_func
import kornia


class ModelBed:
    def __init__(self, args, use_cuda=True):
        self.args = args
        self.use_cuda = use_cuda

        self.renderer_name = 'comp'
        self.device = 'cuda:0'
        self.raytracer = RayTracer()
        #self.set_render_fn(render_fn=render_fn)
        sdf_network = init_sdf_network_dict()
        color_network_dict = init_rendering_network_dict(renderer_name=self.renderer_name)
        sdf_optimizer = torch.optim.Adam(sdf_network.parameters(), lr=1e-6)
        color_optimizer_dict = choose_optmizer(renderer_name=self.renderer_name, network_dict=color_network_dict)
        #renderer = choose_renderer(renderer_name=self.renderer_name)
        log_dir = os.path.join(self.args.out_dir, "logs")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.imreader = image_reader('opencv')
        self.imwriter = image_writer('opencv')

        self.sdf_network = sdf_network
        self.color_network_dict = color_network_dict
        self.sdf_optimizer = sdf_optimizer
        self.color_optimizer_dict = color_optimizer_dict

        image_fpaths, gt_images, Ks, W2Cs = self.load_dataset(data_dir=self.args.data_dir,
                                                              folder_name=self.args.folder_name,
                                                              cam_dict_name=self.args.cam_file_name,
                                                              use_mask=self.args.use_mask,
                                                              use_trans=self.args.use_trans)
        self.image_fpaths = image_fpaths
        self.gt_images = gt_images
        self.Ks = Ks
        self.W2Cs = W2Cs
        self.cameras = []
        for i in range(self.gt_images.shape[0]):
            ci = Camera(W=self.gt_images[i].shape[1],
                        H=self.gt_images[i].shape[0],
                        K=self.Ks[i].cuda(),
                        W2C=self.W2Cs[i].cuda())
            self.cameras.append(ci)
        ic(len(image_fpaths), gt_images.shape, Ks.shape, W2Cs.shape, len(self.cameras))

        # load neus models as sdf initialization
        ic(args.neus_ckpt_fpath)
        load_neus_checkpoint(neus_ckpt_fpath=self.args.neus_ckpt_fpath,
                             sdf_network=self.sdf_network,
                             color_network_dict=self.color_network_dict,
                             load_diffuse_albedo=True)
        # init init lighting
        dist = np.median([torch.norm(self.cameras[i].get_camera_origin()).item() for i in range(len(self.cameras))])
        init_light = args.init_light_scale * dist * dist
        self.color_network_dict["point_light_network"].set_light(init_light)

        # load pretrained checkpoints
        start_step = load_pretrained_checkpoint(ckpt_dir=self.args.out_dir,
                                                sdf_network=self.sdf_network,
                                                color_network_dict=self.color_network_dict)
        self.start_step = start_step
        self.pyramidl2_loss_fn = PyramidL2Loss(use_cuda=self.use_cuda)
        self.ssim_loss_fn = ssim_loss_fn

        # download blender
        self.blender_fpath = "./blender-3.1.0-linux-x64/blender"
        download_blender()

        # choose renderer
        self.renderer = choose_renderer(renderer_name=self.renderer_name)

        self.fill_holes = False
        self.handle_edges = not self.args.no_edgesample
        self.num_iters = self.args.num_iters

    def load_dataset(self, data_dir, folder_name, cam_dict_name, use_mask=True, use_trans=False):
        image_fpaths, gt_images, Ks, W2Cs = load_dataset_general(data_dir=data_dir,
                                                                 folder_name=folder_name,
                                                                 file_name=cam_dict_name,
                                                                 use_mask=use_mask, use_trans=use_trans)
        return image_fpaths, gt_images, Ks, W2Cs

    def set_render_fn(self, render_fn=None):
        self.render_fn = render_fn

    def get_material_comp(self, points, normals, features):
        res = {}
        # configure (1)
        #diffuse_albedo = self.color_network_dict["diffuse_albedo_network"](points, normals, -normals, features).abs()
        #specular_albedo = self.color_network_dict["specular_albedo_network"](points, normals, None, features).abs()
        # configure (2)
        #diffuse_albedo = self.color_network_dict["diffuse_albedo_network"](points, normals, None, features).abs()
        #specular_albedo = self.color_network_dict["specular_albedo_network"](points, normals, -normals, features).abs()
        #specular_albedo = self.color_network_dict["specular_albedo_network"](points, normals, None, features).abs()

        diffuse_albedo = self.color_network_dict["diffuse_albedo_network"](points, normals, -normals, features).abs()
        specular_albedo = self.color_network_dict["specular_albedo_network"](points, normals, -normals, features).abs()

        metallic = self.color_network_dict["metallic_network"](points, normals, None, features).abs()
        specular_roughness = self.color_network_dict["specular_roughness_network"](points, normals, None, features).abs()
        dielectric = self.color_network_dict["dielectric_network"](points, normals, None, features).abs()
        metallic_eta = self.color_network_dict["metallic_eta_network"](points, normals, None, features).abs()
        metallic_k = self.color_network_dict["metallic_k_network"](points, normals, None, features).abs()
        dielectric_eta = self.color_network_dict["dielectric_eta_network"](points, normals, None, features).abs()

        res['diffuse_albedo'] = diffuse_albedo
        res['specular_albedo'] = specular_albedo
        res['metallic'] = metallic
        res['dielectric'] = dielectric
        res['specular_roughness'] = specular_roughness
        res['metallic_eta'] = metallic_eta
        res['metallic_k'] = metallic_k
        res['dielectric_eta'] = dielectric_eta
        return res

    def render_fn(self, interior_mask, color_network_dict, ray_o, ray_d, points, normals, features):
        dots_sh = list(interior_mask.shape)
        rgb = torch.zeros(dots_sh + [3], dtype=torch.float32, device=interior_mask.device)

        # specturm buffer
        diffuse_rgb, specular_rgb = rgb.clone(), rgb.clone()
        diffuse_albedo, specular_albedo = rgb.clone(), rgb.clone()
        metallic_rgb = rgb.clone()
        dielectric_rgb = rgb.clone()

        # rendering model parameters
        specular_roughness = rgb[..., 0:1].clone()
        metallic_eta = rgb[..., 0:1].clone()
        metallic_k = rgb[..., 0:1].clone()
        dielectric_eta = rgb[..., 0:1].clone()

        metallic = torch.zeros(dots_sh + [1], dtype=torch.float32, device=interior_mask.device)
        dielectric = torch.zeros(dots_sh + [1], dtype=torch.float32, device=interior_mask.device)
        normals_pad = rgb.clone()
        costheta = rgb[..., 0:1].clone()

        if interior_mask.any():
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
            ray_d_norm = ray_d / (ray_d.norm(dim=-1, keepdim=True) + 1e-10)
            costheta_t = torch.sum(-1*ray_d_norm*normals, dim=-1, keepdim=True)
            params = self.get_material_comp(points=points, normals=normals, features=features)
            results = self.renderer(
                color_network_dict["point_light_network"](),
                (points - ray_o).norm(dim=-1, keepdim=True),
                normals,
                -ray_d,
                params=params)

            rgb[interior_mask] = results["rgb"]
            diffuse_rgb[interior_mask] = results["diffuse_rgb"]
            specular_rgb[interior_mask] = results["specular_rgb"]
            metallic_rgb[interior_mask] = results["metallic_rgb"]
            dielectric_rgb[interior_mask] = results["dielectric_rgb"]

            diffuse_albedo[interior_mask] = params['diffuse_albedo']
            specular_albedo[interior_mask] = params['specular_albedo']

            specular_roughness[interior_mask] = params['specular_roughness']
            metallic_eta[interior_mask] = params['metallic_eta']
            metallic_k[interior_mask] = params['metallic_k']
            dielectric_eta[interior_mask] = params['dielectric_eta']

            costheta[interior_mask] = costheta_t
            normals_pad[interior_mask] = normals
            metallic[interior_mask] = params['metallic']
            dielectric[interior_mask] = params['dielectric']
        return {
            "color": rgb,
            "diffuse_color": diffuse_rgb,
            "specular_color": specular_rgb,
            "diffuse_albedo": diffuse_albedo,
            "specular_albedo": specular_albedo,
            "specular_roughness": specular_roughness,
            "metallic_eta": metallic_eta,
            "metallic_k": metallic_k,
            "dielectric_eta": dielectric_eta,
            "normal": normals_pad,
            "metallic_rgb": metallic_rgb,
            "metallic": metallic,
            "dielectric_rgb": dielectric_rgb,
            "dielectric": dielectric,
            "costheta": costheta,
        }

    def set_raytracer(self, raytracer=None):
        self.raytracer = raytracer

    def set_optimizer_dict(self):
        pass

    def pyramid_loss_fn(self):
        return PyramidL2Loss(use_cuda=self.use_cuda)

    def render_all(self):
        file_name = f"render_{os.path.basename(self.args.data_dir)}_{self.start_step}"
        render_out_dir = os.path.join(self.args.out_dir, file_name)
        os.makedirs(render_out_dir, exist_ok=True)
        ic(f"Rendering images to: {render_out_dir}")
        n_cams = len(self.cameras)
        for i in tqdm.tqdm(range(n_cams)):
            cam, impath = self.cameras[i], self.image_fpaths[i]
            results = render_camera(
                cam,
                self.sdf_network,
                self.raytracer,
                self.color_network_dict,
                render_fn=self.render_fn,
                fill_holes=True,
                handle_edges=True,
                is_training=False)
            if self.args.gamma_pred:
                results["color"] = gamma_correction(results["color"], gamma=2.2)
                results["diffuse_color"] = gamma_correction(results["diffuse_color"], gamma=2.2)
                results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)
            for x in list(results.keys()):
                results[x] = results[x].detach().cpu().numpy()
            color_im = results["color"]
            timgname = os.path.basename(impath).split('.')[0]
            self.imwriter(os.path.join(render_out_dir, timgname + '.jpg'), to8b(color_im))
            normal = results["normal"]
            normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
            normal_im = (normal + 1.0) / 2.0
            self.imwriter(os.path.join(render_out_dir, timgname + '_normal.jpg'), to8b(normal_im))
            diff_im = results["diffuse_color"]
            self.imwriter(os.path.join(render_out_dir, timgname + '_diff.jpg'), to8b(diff_im))
            specular_im = results["specular_color"]
            self.imwriter(os.path.join(render_out_dir, timgname + '_specular.jpg'), to8b(specular_im))

    def export_mesh_and_materials(self, export_out_dir, use_no_translation=False):
        ic(f"Exporting mesh and materials to: {export_out_dir}")
        sdf_fn = lambda x: self.sdf_network(x)[..., 0]
        ic("Exporting mesh and uv...")
        with torch.no_grad():
            if not use_no_translation:
                mesh_name = 'mesh.obj'
                export_mesh(sdf_fn, os.path.join(export_out_dir, mesh_name))
                os.system(
                    f"{self.blender_fpath} --background --python models/export_uv.py {os.path.join(export_out_dir, mesh_name)} {os.path.join(export_out_dir, mesh_name)}"
                )
            if use_no_translation:
                mesh_name = 'mesh_no_translation.obj'
                export_mesh_no_translation(sdf_fn, os.path.join(export_out_dir, mesh_name))
                os.system(
                    f"{self.blender_fpath} --background --python models/export_uv.py {os.path.join(export_out_dir, mesh_name)} {os.path.join(export_out_dir, mesh_name)}")

    def export_all(self, global_step=-1):
        #### export mesh and materials
        export_out_dir = os.path.join(self.args.out_dir, f"mesh_and_materials_{global_step}")
        os.makedirs(export_out_dir, exist_ok=True)
        self.export_mesh_and_materials(export_out_dir, self.sdf_network, self.color_network_dict)

    def write_loss_all(self, loss_dict={}, global_step=-1):
        self.writer.add_scalar("loss/loss", loss_dict['loss'], global_step)
        self.writer.add_scalar("loss/img_loss", loss_dict['img_loss'], global_step)
        self.writer.add_scalar("loss/img_l2_loss", loss_dict['img_l2_loss'], global_step)
        self.writer.add_scalar("loss/img_ssim_loss", loss_dict['img_ssim_loss'], global_step)
        self.writer.add_scalar("loss/eik_loss", loss_dict['eik_loss'], global_step)
        self.writer.add_scalar("loss/roughrange_loss", loss_dict['roughrange_loss'], global_step)
        self.writer.add_scalar("loss/metallicness_loss", loss_dict['metallicness_loss'], global_step)
        self.writer.add_scalar("loss/dielectricness_loss", loss_dict['dielectricness_loss'], global_step)
        self.writer.add_scalar("light", self.color_network_dict["point_light_network"].get_light())

    def save_checkpoint(self, global_step=-1):
        para_list = [("sdf_network", self.sdf_network.state_dict())]
        para_list.extend([(x, self.color_network_dict[x].state_dict()) for x in self.color_network_dict.keys()])
        save_dict = dict(para_list)
        torch.save(save_dict, os.path.join(self.args.out_dir, f"ckpt_{global_step}.pth"))

    def validate_image(self, resize_ratio=0.25, global_step=-1, idx=-1):
        if idx == -1:
            idx = np.random.randint(0, len(self.cameras), 1)[0]
        if self.args.plot_image_name is not None:
            while idx < len(self.image_fpaths):
                if self.args.plot_image_name in self.image_fpaths[idx]:
                    break
                idx += 1

        camera_resize, gt_color_resize = self.cameras[idx].resize(factor=resize_ratio, image=self.gt_images[idx])
        results = render_camera(
            camera_resize,
            self.sdf_network,
            self.raytracer,
            self.color_network_dict,
            render_fn=self.render_fn,
            fill_holes=self.fill_holes,
            handle_edges=self.handle_edges,
            is_training=False)
        #if self.args.gamma_pred:
        #    results["color"] = gamma_correction(results["color"], 1.0 / 2.2)
        #    results["diffuse_color"] = gamma_correction(results["diffuse_color"], 1.0 / 2.2)
        #    results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)

        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()

        gt_color_im = gt_color_resize.detach().cpu().numpy()
        color_im = results["color"]
        diffuse_color_im = results["diffuse_color"]
        specular_color_im = results["specular_color"]
        normal = results["normal"]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
        normal_im = (normal + 1.0) / 2.0
        if 'edge_mask' in results:
            edge_mask_im = np.tile(results["edge_mask"][:, :, np.newaxis], (1, 1, 3))
        else:
            edge_mask_im = np.zeros_like(results["diffuse_albedo"])
        diffuse_albedo_im = results["diffuse_albedo"]
        specular_albedo_im = results["specular_albedo"]
        specular_roughness_im = np.tile(results["specular_roughness"][:, :, np.newaxis], (1, 1, 3))
        depth = np.log(np.tile(results["depth"][:, :, np.newaxis] + 1e-6, (1, 1, 3)))
        costheta = np.tile(results["costheta"][:, :, np.newaxis], (1, 1, 3))
        costheta_mask = costheta > 0.9

        metallic_rgb = results['metallic_rgb']
        dielectric_rgb = results['dielectric_rgb']
        metal_eta = results['metallic_eta']
        metal_k = results['metallic_k']
        dielectric_eta = results['dielectric_eta']

        if self.args.gamma_pred:
            gt_color_im = gamma_correction(gt_color_im + 1e-6, 2.2)
            color_im = gamma_correction(color_im + 1e-6, 2.2)
            diffuse_color_im = gamma_correction(diffuse_color_im + 1e-6, 2.2)
            specular_color_im = gamma_correction(specular_color_im + 1e-6, 2.2)
            metallic_rgb = gamma_correction(metallic_rgb + 1e-6, 2.2)
            diffuse_albedo_im = gamma_correction(diffuse_albedo_im + 1e-6, 2.2)
            specular_albedo_im = gamma_correction(specular_albedo_im + 1e-6, 2.2)
            #maxv, minv = np.max(metallic_rgb[:]), np.min(metallic_rgb[:])
            dielectric_rgb = gamma_correction(dielectric_rgb + 1e-6, 2.2)

            #depth = gamma_correction(depth + 1e-6, 1.0 / 2.2)
            #depth = (depth - np.min(depth[:])) / (np.max(depth[:]) - np.min(depth[:]))

            metal_eta = gamma_correction(metal_eta + 1e-6, 2.2)
            metal_eta = (metal_eta - np.min(metal_eta[:])) / (np.max(metal_eta[:]) - np.min(metal_eta[:]))
            metal_eta = metal_eta
            metal_k = gamma_correction(metal_k + 1e-6, 2.2)
            metal_eta = (metal_eta - np.min(metal_eta[:])) / (np.max(metal_eta[:]) - np.min(metal_eta[:]))
            dielectric_eta = gamma_correction(dielectric_eta + 1e-6, 2.2)

        gt_color_im = gt_color_im[..., :3]
        normal_im = normal_im[..., :3]

        img_list = [gt_color_im, color_im, normal_im, edge_mask_im,
                    diffuse_color_im, specular_color_im, diffuse_albedo_im, specular_albedo_im,
                    depth, metallic_rgb, dielectric_rgb, specular_roughness_im,
                    costheta, costheta_mask]
                    #metal_eta, dielectric_eta, costheta]
        im = concatenate_result(image_list=img_list, imarray_length=4)
        file_name = f"logim_{global_step}_{os.path.basename(self.image_fpaths[idx])}"
        self.imwriter(os.path.join(self.args.out_dir, file_name), to8b(im))

    def train(self):
        fill_holes = self.fill_holes
        handle_edges = not self.args.no_edgesample
        is_training = True
        if self.args.inv_gamma_gt:
            ic("linearizing ground-truth images using inverse gamma correction")
            self.gt_images = inv_gamma_correction(self.gt_images, gamma=2.2)

        ic(fill_holes, handle_edges, is_training, self.args.inv_gamma_gt)

        global_step = self.args.num_iters
        start_step = self.start_step
        num_iter = self.num_iters
        num_gt_images = self.gt_images.shape[0]

        for global_step in tqdm.tqdm(range(start_step + 1, num_iter)):
            self.sdf_optimizer.zero_grad()
            for x in self.color_optimizer_dict.keys():
                self.color_optimizer_dict[x].zero_grad()

            idx = np.random.randint(0, num_gt_images)
            camera_crop, gt_color_crop = self.cameras[idx].crop_region(trgt_W=self.args.patch_size,
                                                                       trgt_H=self.args.patch_size,
                                                                       image=self.gt_images[idx])

            results = render_camera(camera_crop,
                                    self.sdf_network,
                                    self.raytracer,
                                    self.color_network_dict,
                                    render_fn=self.render_fn,
                                    fill_holes=self.fill_holes,
                                    handle_edges=self.handle_edges,
                                    is_training=is_training)
            if self.args.gamma_pred:
                try:
                    results["color"] = gamma_correction(results["color"])
                    results["diffuse_color"] = gamma_correction(results["diffuse_color"])
                    results["specular_color"] = gamma_correction(results["specular_color"])
                except:
                    print(results.keys())

            mask = results["convergent_mask"]
            if handle_edges:
                mask = mask | results["edge_mask"]

            img_loss = torch.Tensor([0.0]).cuda()
            img_l2_loss = torch.Tensor([0.0]).cuda()
            img_ssim_loss = torch.Tensor([0.0]).cuda()
            roughrange_loss = torch.Tensor([0.0]).cuda()
            metallicness_loss = torch.Tensor([0.0]).cuda()
            dielectricness_loss = torch.Tensor([0.0]).cuda()

            eik_points = torch.empty(camera_crop.H * camera_crop.W // 2, 3).cuda().float().uniform_(-1.0, 1.0)
            eik_grad = self.sdf_network.gradient(eik_points).view(-1, 3)
            eik_cnt = eik_grad.shape[0]
            eik_loss = ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

            if mask.any():
                pred_img = results["color"].permute(2, 0, 1).unsqueeze(0)
                gt_img = gt_color_crop.permute(2, 0, 1).unsqueeze(0).to(pred_img.device, dtype=pred_img.dtype)
                # print(pred_img.shape, gt_img.shape)
                pred_img = pred_img[:, :3, :, :]
                gt_img = gt_img[:, :3, :, :]

                # calculate image loss
                img_l2_loss = self.pyramidl2_loss_fn(pred_img, gt_img)
                img_ssim_loss = self.ssim_loss_fn(pred_img, gt_img, mask.unsqueeze(0).unsqueeze(0))
                img_loss = img_l2_loss + img_ssim_loss * self.args.ssim_weight

                # calculate eik loss
                eik_grad = results["normal"][mask]
                eik_cnt += eik_grad.shape[0]
                eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

                if "edge_pos_neg_normal" in results:
                    eik_grad = results["edge_pos_neg_normal"]
                    eik_cnt += eik_grad.shape[0]
                    eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

                # calculate roughness range loss
                roughness = results["specular_roughness"][mask]
                roughness_value = 0.5
                roughness = roughness[roughness > roughness_value]
                if roughness.numel() > 0:
                    roughrange_loss = (roughness - roughness_value).mean() * self.args.roughrange_weight

                # calculate metallic eta k loss
                if 'metallic_eta' in results:
                    metal_eta = results["metallic_eta"][mask]
                    metal_k = results["metallic_k"][mask]
                    # metal_eta_value, metal_k_value = 0.198125, 5.631250
                    metal_eta_value, metal_k_value = 1.0, 10.0
                    metal_eta = metal_eta[metal_eta > metal_eta_value]
                    metal_k = metal_k[metal_k > metal_k_value]
                    metallicness_loss = (torch.abs(metal_eta - metal_eta_value)).mean() * self.args.metal_eta_weight
                    metallicness_loss += (torch.abs(metal_k - metal_k_value)).mean() * self.args.metal_k_weight

                if 'dielectric_eta' in results:
                    dielectric_eta = results['dielectric_eta'][mask]
                    dielectricness_loss = (torch.abs(dielectric_eta - 1.5)).mean() * self.args.dielectric_eta_weight

            eik_loss = eik_loss / eik_cnt * self.args.eik_weight
            loss = img_loss + eik_loss + roughrange_loss + metallicness_loss + dielectricness_loss
            loss.backward()

            self.sdf_optimizer.step()
            for x in self.color_optimizer_dict.keys():
                self.color_optimizer_dict[x].step()

            if global_step % 500 == 0:
                loss_dict = {'loss': loss, 'img_loss': img_loss, 'img_l2_loss': img_l2_loss,
                             'img_ssim_loss': img_ssim_loss,
                             'eik_loss': eik_loss, 'roughrange_loss': roughrange_loss,
                             'metallicness_loss': metallicness_loss, 'dielectricness_loss': dielectricness_loss}
                self.write_loss_all(loss_dict=loss_dict, global_step=global_step)

            if global_step % 1000 == 0:
                self.save_checkpoint(global_step=global_step)

            if global_step % 100 == 0:
                ic(
                    self.args.out_dir,
                    global_step,
                    loss.item(),
                    img_loss.item(),
                    img_l2_loss.item(),
                    img_ssim_loss.item(),
                    eik_loss.item(),
                    roughrange_loss.item(),
                    metallicness_loss.item(),
                    dielectricness_loss.item(),
                    self.color_network_dict["point_light_network"].get_light().item(),
                )

                for x in list(results.keys()):
                    del results[x]
                self.validate_image(resize_ratio=0.25, global_step=global_step)

    def component_switch(self, network_list=[]):
        for nn in network_list:
            if nn in self.color_network_dict:
                self.color_network_dict[nn].requires_grad_(requires_grad=True)
            else:
                self.color_network_dict[nn].requires_grad_(requires_grad=False)

    def train_comp(self, network_list=['diffuse_albedo_network'], opt_sdf=True, num_iter=10000):
        fill_holes = self.fill_holes
        handle_edges = not self.args.no_edgesample
        is_training = True
        if opt_sdf:
            self.sdf_network.requires_grad_(requires_grad=True)
        else:
            self.sdf_network.requires_grad_(requires_grad=False)
        self.component_switch(network_list=network_list)

        if self.args.inv_gamma_gt:
            ic("linearizing ground-truth images using inverse gamma correction")
            self.gt_images = inv_gamma_correction(self.gt_images, gamma=2.2)

        ic(fill_holes, handle_edges, is_training, self.args.inv_gamma_gt)

        global_step = self.args.num_iters
        start_step = self.start_step
        #start_step = 0
        #num_iter = self.num_iters
        num_gt_images = self.gt_images.shape[0]

        # change optimize components
        #self.color_optimizer_dict = choose_optmizer(renderer_name='comp',
        #                                            network_name_list=network_list,
        #                                            network_dict=self.color_network_dict)

        for global_step in tqdm.tqdm(range(start_step, num_iter)):
            self.sdf_optimizer.zero_grad()
            for x in self.color_optimizer_dict.keys():
                self.color_optimizer_dict[x].zero_grad()

            idx = np.random.randint(0, num_gt_images)
            camera_crop, gt_color_crop = self.cameras[idx].crop_region(trgt_W=self.args.patch_size,
                                                                       trgt_H=self.args.patch_size,
                                                                       image=self.gt_images[idx])

            results = render_camera(camera_crop,
                                    self.sdf_network,
                                    self.raytracer,
                                    self.color_network_dict,
                                    render_fn=self.render_fn,
                                    fill_holes=self.fill_holes,
                                    handle_edges=self.handle_edges,
                                    is_training=is_training)
            if self.args.gamma_pred:
                try:
                    results["color"] = gamma_correction(results["color"])
                    results["diffuse_color"] = gamma_correction(results["diffuse_color"])
                    results["specular_color"] = gamma_correction(results["specular_color"])
                except:
                    print(results.keys())

            #results["color"] = results["diffuse_color"]

            mask = results["convergent_mask"]
            costheta_mask = (results["costheta"]) > 0.9
            if handle_edges:
                mask = mask | results["edge_mask"]

            img_loss = torch.Tensor([0.0]).cuda()
            albedo_loss = torch.Tensor([0.0]).cuda()
            img_l2_loss = torch.Tensor([0.0]).cuda()
            img_ssim_loss = torch.Tensor([0.0]).cuda()
            roughrange_loss = torch.Tensor([0.0]).cuda()
            metallicness_loss = torch.Tensor([0.0]).cuda()
            dielectricness_loss = torch.Tensor([0.0]).cuda()

            eik_points = torch.empty(camera_crop.H * camera_crop.W // 2, 3).cuda().float().uniform_(-1.0, 1.0)
            eik_grad = self.sdf_network.gradient(eik_points).view(-1, 3)
            eik_cnt = eik_grad.shape[0]
            eik_loss = ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

            if mask.any():
                pred_img = results["color"].permute(2, 0, 1).unsqueeze(0)
                gt_img = gt_color_crop.permute(2, 0, 1).unsqueeze(0).to(pred_img.device, dtype=pred_img.dtype)
                pred_diff_img = results["diffuse_color"].permute(2, 0, 1).unsqueeze(0)
                pred_spec_img = results["specular_color"].permute(2, 0, 1).unsqueeze(0)
                pred_diff_albedo = results["diffuse_albedo"].permute(2, 0, 1).unsqueeze(0)
                pred_spec_albedo = results["specular_albedo"].permute(2, 0, 1).unsqueeze(0)

                # print(pred_img.shape, gt_img.shape)
                pred_img = pred_img[:, :3, :, :]
                gt_img = gt_img[:, :3, :, :]

                #pred_diff_img = pred_diff_img[:, :3, :, :]
                #pred_spec_img = pred_spec_img[:, :3, :, :]
                #gt_img = gt_img[:, :3, :, :]

                # calculate image loss
                #pred_diff_img[..., costheta_mask] = 0
                #gt_diff_img = gt_img.clone()
                #gt_diff_img[..., costheta_mask] = 0

                #img_l2_loss = self.pyramidl2_loss_fn(pred_diff_img, gt_diff_img)
                #img_l2_loss += self.pyramidl2_loss_fn(pred_img, gt_img)
                # 1. end2end image loss
                img_l2_loss = self.pyramidl2_loss_fn(pred_img, gt_img)
                img_ssim_loss = self.ssim_loss_fn(pred_img, gt_img, mask.unsqueeze(0).unsqueeze(0))
                img_loss = img_l2_loss + img_ssim_loss * self.args.ssim_weight

                # 2. loss for albedo, L1 smooth
                if False:
                    grad_diff = kornia.filters.laplacian(pred_diff_albedo, 3)
                    albedo_loss = torch.sum(kornia.losses.total_variation(grad_diff, reduction='sum'))
                    grad_spec = kornia.filters.laplacian(pred_spec_albedo, 3)
                    albedo_loss += torch.sum(kornia.losses.total_variation(grad_spec, reduction='sum'))
                    albedo_loss *= 0.01

                # calculate eik loss
                eik_grad = results["normal"][mask]
                eik_cnt += eik_grad.shape[0]
                eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

                if "edge_pos_neg_normal" in results:
                    eik_grad = results["edge_pos_neg_normal"]
                    eik_cnt += eik_grad.shape[0]
                    eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

                # calculate roughness range loss
                roughness = results["specular_roughness"][mask]
                roughness_value = 0.5
                roughness = roughness[roughness > roughness_value]
                if roughness.numel() > 0:
                    roughrange_loss = (roughness - roughness_value).mean() * self.args.roughrange_weight

                # calculate metallic eta k loss
                if False:
                    if 'metallic_eta' in results:
                        metal_eta = results["metallic_eta"][mask]
                        metal_k = results["metallic_k"][mask]
                        # metal_eta_value, metal_k_value = 0.198125, 5.631250
                        metal_eta_value, metal_k_value = 1.0, 10.0
                        metal_eta = metal_eta[metal_eta > metal_eta_value]
                        metal_k = metal_k[metal_k > metal_k_value]
                        metallicness_loss = (torch.abs(metal_eta - metal_eta_value)).mean() * self.args.metal_eta_weight
                        metallicness_loss += (torch.abs(metal_k - metal_k_value)).mean() * self.args.metal_k_weight

                    if 'dielectric_eta' in results:
                        dielectric_eta = results['dielectric_eta'][mask]
                        dielectricness_loss = (torch.abs(dielectric_eta - 1.5)).mean() * self.args.dielectric_eta_weight

            eik_loss = eik_loss / eik_cnt * self.args.eik_weight
            loss = img_loss + albedo_loss + eik_loss + roughrange_loss + metallicness_loss + dielectricness_loss
            loss.backward()

            self.sdf_optimizer.step()
            for x in self.color_optimizer_dict.keys():
                self.color_optimizer_dict[x].step()

            if global_step % 500 == 0:
                loss_dict = {'loss': loss, 'img_loss': img_loss, 'img_l2_loss': img_l2_loss,
                             'img_ssim_loss': img_ssim_loss,
                             'eik_loss': eik_loss, 'roughrange_loss': roughrange_loss,
                             'metallicness_loss': metallicness_loss, 'dielectricness_loss': dielectricness_loss}
                self.write_loss_all(loss_dict=loss_dict, global_step=global_step)

            if global_step % 1000 == 0:
                self.save_checkpoint(global_step=global_step)

            if global_step % 100 == 0:
                ic(
                    self.args.out_dir,
                    global_step,
                    loss.item(),
                    img_loss.item(),
                    img_l2_loss.item(),
                    img_ssim_loss.item(),
                    albedo_loss.item(),
                    eik_loss.item(),
                    roughrange_loss.item(),
                    metallicness_loss.item(),
                    dielectricness_loss.item(),
                    self.color_network_dict["point_light_network"].get_light().item(),
                )

                for x in list(results.keys()):
                    del results[x]
                if False:
                    for idx in range(len(self.cameras)):
                        self.validate_image(resize_ratio=0.5, global_step=global_step, idx=idx)
                else:
                    self.validate_image(resize_ratio=0.25, global_step=global_step, idx=12)


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="input data directory")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--nir_dir", type=str, default=None, help="output directory")
    parser.add_argument("--folder_name", type=str, default="image", help="dataset image folder")
    parser.add_argument("--neus_ckpt_fpath", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--num_iters", type=int, default=100001, help="number of iterations")
    parser.add_argument("--patch_size", type=int, default=128, help="width and height of the rendered patches")
    parser.add_argument("--eik_weight", type=float, default=0.1, help="weight for eikonal loss")
    parser.add_argument("--ssim_weight", type=float, default=1.0, help="weight for ssim loss")
    parser.add_argument("--roughrange_weight", type=float, default=0.1, help="weight for roughness range loss")
    parser.add_argument("--metal_eta_weight", type=float, default=0.1, help="weight for metal eta loss")
    parser.add_argument("--metal_k_weight", type=float, default=0.1, help="weight for metal k loss")
    parser.add_argument("--dielectric_eta_weight", type=float, default=0.1, help="weight for dielectric eta loss")

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

    parser.add_argument(
        "--train_rgb",
        action="store_true",
        help="whether to render the input image set RGB",
    )
    parser.add_argument(
        "--train_nir",
        action="store_true",
        help="whether to render the input image set NIR",
    )
    parser.add_argument("--cam_file_name", type=str, default='cam_dict_norm.json', help="cam dict file name")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_mask",
                        action="store_true",
                        help="whether to use mask")
    parser.add_argument("--use_trans",
                        action="store_true",
                        help="whether to use translation")
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    ic(args)

    ###### back up arguments and code scripts
    os.makedirs(args.out_dir, exist_ok=True)
    parser.write_config_file(args, [os.path.join(args.out_dir, "args.txt")])
    testbed = ModelBed(args=args, use_cuda=True)
    if args.train_rgb:
        network_list = ['color_network',
                        'diffuse_albedo_network',
                        'specular_albedo_network',
                        'specular_roughness_network',
                        #'metallic_eta_network', 'metallic_k_network', 'dielectric_eta_network']
                        'point_light_network']

        testbed.train_comp(network_list=network_list, opt_sdf=True, num_iter=100000)
    if args.render_all:
        testbed.render_all()

    # if args.render_all:
    #     testbed.render_all()
    #
    # if args.export_all:
    #     testbed.export_all()


if __name__ == '__main__':
    main()
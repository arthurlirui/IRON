import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset, DatasetNIRRGB
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, NeRFdual
from models.renderer import NeuSRenderer


class Runner:
    def __init__(self, conf_path, mode="train",
                 case="CASE_NAME",
                 nir_case="NIR_NAME",
                 rgb_case="RGB_NAME", is_continue=False):
        self.device = torch.device("cuda")

        # Configuration
        self.conf_path = conf_path
        conf_text = None
        with open(self.conf_path) as f:
            #f = open(self.conf_path)
            conf_text = f.read()
            conf_text = conf_text.replace("CASE_NAME", case)
        #f.close()
        print(conf_text)
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf["dataset.data_dir"] = self.conf["dataset.data_dir"].replace("CASE_NAME", case)
        self.conf['dataset']['rgb_dir'] = self.conf['dataset']['rgb_dir'].replace("CASE_NAME", rgb_case)
        self.conf['dataset']['nir_dir'] = self.conf['dataset']['nir_dir'].replace("CASE_NAME", nir_case)
        #self.conf["dataset.rgb_dir"] = self.conf['dataset']['rgb_dir']
        #self.conf["dataset.nir_dir"] = self.conf['dataset']['nir_dir']
        self.base_exp_dir = self.conf["general.base_exp_dir"]
        self.rgb_exp_dir = self.conf["general.rgb_exp_dir"]
        os.makedirs(self.rgb_exp_dir, exist_ok=True)
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = DatasetNIRRGB(self.conf["dataset"])
        # if os.path.exists(self.conf['dataset']['rgb_dir']):
        #     #self.rgb_dataset = DatasetNIRRGB(self.conf["dataset"], dataset_type='rgb')
        #     self.rgb_dataset = DatasetNIRRGB(self.conf["dataset"], dataset_type='rgb')
        # if os.path.exists(self.conf['dataset']['nir_dir']):
        #     self.nir_dataset = DatasetNIRRGB(self.conf["dataset"], dataset_type='nir')

        #self.dataset_nir = DatasetNIRRGB(self.conf["dataset"], dataset_type='nir')
        #self.dataset_rgb = DatasetNIRRGB(self.conf["dataset"], dataset_type='rgb')
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int("train.end_iter")
        self.save_freq = self.conf.get_int("train.save_freq")

        self.RGB_end_iter = self.end_iter
        self.NIR_end_iter = 2*self.end_iter

        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.val_mesh_freq = self.conf.get_int("train.val_mesh_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        self.validate_resolution_level = self.conf.get_int("train.validate_resolution_level")
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.learning_rate_alpha = self.conf.get_float("train.learning_rate_alpha")
        self.use_white_bkgd = self.conf.get_bool("train.use_white_bkgd")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.anneal_end = self.conf.get_float("train.anneal_end", default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float("train.igr_weight")
        self.mask_weight = self.conf.get_float("train.mask_weight")
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.nir_nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf["model.sdf_network"]).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf["model.variance_network"]).to(self.device)
        if self.dataset.enable_RGB:
            self.color_network = RenderingNetwork(**self.conf["model.rendering_network"]).to(self.device)
        if self.dataset.enable_NIR:
            self.nir_network = RenderingNetwork(**self.conf["model.nir_network"]).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        if self.dataset.enable_RGB:
            params_to_train += list(self.color_network.parameters())
        if self.dataset.enable_NIR:
            params_to_train += list(self.nir_nerf_outside.parameters())
            params_to_train += list(self.nir_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        if self.dataset.enable_RGB:
            self.renderer = NeuSRenderer(
                self.nerf_outside,
                self.sdf_network,
                self.deviation_network,
                self.color_network,
                **self.conf["model.neus_renderer"]
            )

        if self.dataset.enable_NIR:
            self.renderer_nir = NeuSRenderer(
                self.nir_nerf_outside,
                self.sdf_network,
                self.deviation_network,
                self.nir_network,
                **self.conf["model.neus_renderer"]
            )

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if self.dataset.enable_NIR:
                model_list_raw = os.listdir(os.path.join(self.rgb_exp_dir, "checkpoints"))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == "pth" and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                if len(model_list) > 0:
                    latest_rgb_model_name = model_list[-1]
                else:
                    latest_rgb_model_name = None

                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, "checkpoints"))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == "pth" and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                if len(model_list) > 0:
                    latest_nir_model_name = model_list[-1]
                else:
                    latest_nir_model_name = None
                logging.info("Find checkpoint: rgb{} nir{}".format(latest_rgb_model_name, latest_nir_model_name))
                #if latest_rgb_model_name is not None and latest_nir_model_name is not None:
                self.load_checkpoint_NIR(rgb_checkpoint_name=latest_rgb_model_name,
                                         nir_checkpoint_name=latest_nir_model_name)

            if self.dataset.enable_RGB:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, "checkpoints"))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == "pth" and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]
                if latest_model_name is not None:
                    logging.info("Find checkpoint: {}".format(latest_model_name))
                    self.load_checkpoint(latest_model_name)

        # if latest_model_name is not None:
        #     logging.info("Find checkpoint: {}".format(latest_model_name))
        #     if self.dataset.enable_NIR:
        #         self.load_checkpoint_NIR(rgb_checkpoint_name=rgb_models_name,
        #                                  nir_checkpoint_name=latest_model_name)
        #     if self.dataset.enable_RGB:
        #         self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == "train":
            self.file_backup()

    def train_NIR(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, "logs"))
        self.update_learning_rate()
        res_step = self.NIR_end_iter - self.iter_step
        image_perm = torch.randperm(self.dataset.n_NIR)

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)],
                                                   self.batch_size, data_type='nir')

            rays_o, rays_d, true_rgb, mask = (data[:, 0:3], data[:, 3:6], data[:, 6:9], data[:, 9:10])
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            background_nir = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])
                background_nir = torch.zeros([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer_nir.render(
                rays_o,
                rays_d,
                near,
                far,
                background_rgb=background_nir,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )

            color_fine = render_out["color_fine"]
            s_val = render_out["s_val"]
            cdf_fine = render_out["cdf_fine"]
            gradient_error = render_out["gradient_error"]
            weight_max = render_out["weight_max"]
            weight_sum = render_out["weight_sum"]
            pred_rgb = color_fine[..., :3]

            # Loss
            color_error = (pred_rgb - true_rgb) * mask

            # previous
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum

            # Arthur
            # color_fine_loss = F.l2_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum
            # color_fine_loss = F.mse_loss(color_fine * mask, true_rgb * mask, reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((pred_rgb - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + eikonal_loss * self.igr_weight + mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar("Loss/loss", loss, self.iter_step)
            self.writer.add_scalar("Loss/color_loss", color_fine_loss, self.iter_step)
            self.writer.add_scalar("Loss/eikonal_loss", eikonal_loss, self.iter_step)
            self.writer.add_scalar("Statistics/s_val", s_val.mean(), self.iter_step)
            self.writer.add_scalar("Statistics/cdf", (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/weight_max", (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/psnr", psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print("iter:{:8>d} loss = {} lr={}".format(self.iter_step, loss, self.optimizer.param_groups[0]["lr"]))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image(data_type='nir')

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = torch.randperm(self.dataset.n_NIR)

    def train_RGB(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, "logs"))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = torch.randperm(self.dataset.n_RGB)

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size, data_type='rgb')

            rays_o, rays_d, true_rgb, mask = (data[:, 0:3], data[:, 3:6], data[:, 6:9], data[:, 9:10])
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d, scale=0.2)

            background_rgb = None
            #background_nir = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])
                #background_nir = torch.zeros([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(
                rays_o,
                rays_d,
                near,
                far,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )

            color_fine = render_out["color_fine"]
            s_val = render_out["s_val"]
            cdf_fine = render_out["cdf_fine"]
            gradient_error = render_out["gradient_error"]
            weight_max = render_out["weight_max"]
            weight_sum = render_out["weight_sum"]
            pred_rgb = color_fine[..., :3]

            # Loss
            color_error = (pred_rgb - true_rgb) * mask

            # previous
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum

            # Arthur
            # color_fine_loss = F.l2_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum
            # color_fine_loss = F.mse_loss(color_fine * mask, true_rgb * mask, reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((pred_rgb - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + eikonal_loss * self.igr_weight + mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar("Loss/loss", loss, self.iter_step)
            self.writer.add_scalar("Loss/color_loss", color_fine_loss, self.iter_step)
            self.writer.add_scalar("Loss/eikonal_loss", eikonal_loss, self.iter_step)
            self.writer.add_scalar("Statistics/s_val", s_val.mean(), self.iter_step)
            self.writer.add_scalar("Statistics/cdf", (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/weight_max", (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/psnr", psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print("iter:{:8>d} loss = {} lr={}".format(self.iter_step, loss, self.optimizer.param_groups[0]["lr"]))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image(data_type='rgb')

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = torch.randperm(self.dataset.n_RGB)

    def train_NIRRGB(self, data_type='rgb'):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, "logs"))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        res_step = 0
        if data_type == 'rgb':
            res_step = self.RGB_end_iter - self.iter_step
            image_perm = torch.randperm(self.dataset.n_RGB)
        elif data_type == 'nir':
            res_step = self.NIR_end_iter - self.iter_step
            image_perm = torch.randperm(self.dataset.n_NIR)
        else:
            image_perm = torch.randperm(self.dataset.n_RGB)

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)],
                                                   self.batch_size,
                                                   data_type=data_type)

            rays_o, rays_d, true_rgb, mask = (data[:, :3], data[:, 3:6], data[:, 6:9], data[:, 9:10])
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            if data_type == 'rgb':
                render_out = self.renderer.render(
                    rays_o,
                    rays_d,
                    near,
                    far,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                )
            elif data_type == 'nir':
                render_out = self.renderer_nir.render(
                    rays_o,
                    rays_d,
                    near,
                    far,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                )

            color_fine = render_out["color_fine"]
            s_val = render_out["s_val"]
            cdf_fine = render_out["cdf_fine"]
            gradient_error = render_out["gradient_error"]
            weight_max = render_out["weight_max"]
            weight_sum = render_out["weight_sum"]
            pred_rgb = color_fine[..., :3]
            #pred_nir = color_fine[..., 3:6]
            # Loss
            color_error = (pred_rgb - true_rgb) * mask
            # if data_type == 'rgb':
            #     color_error = (pred_rgb - true_rgb) * mask
            # elif data_type == 'nir':
            #     color_error = (pred_nir - true_rgb) * mask
            # else:
            #     pass

            # previous
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum

            # Arthur
            # color_fine_loss = F.l2_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum
            # color_fine_loss = F.mse_loss(color_fine * mask, true_rgb * mask, reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((pred_rgb - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            # if data_type == 'rgb':
            #     psnr = 20.0 * torch.log10(1.0 / (((pred_rgb - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            # elif data_type == 'nir':
            #     psnr = 20.0 * torch.log10(1.0 / (((pred_nir - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            # else:
            #     pass
            #psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + eikonal_loss * self.igr_weight + mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar("Loss/loss", loss, self.iter_step)
            self.writer.add_scalar("Loss/color_loss", color_fine_loss, self.iter_step)
            self.writer.add_scalar("Loss/eikonal_loss", eikonal_loss, self.iter_step)
            self.writer.add_scalar("Statistics/s_val", s_val.mean(), self.iter_step)
            self.writer.add_scalar("Statistics/cdf", (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/weight_max", (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/psnr", psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print("iter:{:8>d} loss = {} lr={}".format(self.iter_step, loss, self.optimizer.param_groups[0]["lr"]))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint(data_type=data_type)

            if self.iter_step % self.val_freq == 0:
                #self.validate_image(data_type=data_type)
                self.validate_image(data_type=data_type)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                if data_type == 'rgb':
                    image_perm = torch.randperm(self.dataset.n_RGB)
                elif data_type == 'nir':
                    image_perm = torch.randperm(self.dataset.n_NIR)
                else:
                    image_perm = torch.randperm(self.dataset.n_RGB)

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, "logs"))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = (
                data[:, :3],
                data[:, 3:6],
                data[:, 6:9],
                data[:, 9:10],
            )
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(
                rays_o,
                rays_d,
                near,
                far,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )

            color_fine = render_out["color_fine"]
            s_val = render_out["s_val"]
            cdf_fine = render_out["cdf_fine"]
            gradient_error = render_out["gradient_error"]
            weight_max = render_out["weight_max"]
            weight_sum = render_out["weight_sum"]

            # Loss
            color_error = (color_fine - true_rgb) * mask

            # previous
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum

            # Arthur
            #color_fine_loss = F.l2_loss(color_error, torch.zeros_like(color_error), reduction="sum") / mask_sum
            #color_fine_loss = F.mse_loss(color_fine * mask, true_rgb * mask, reduction='sum') / mask_sum

            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + eikonal_loss * self.igr_weight + mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar("Loss/loss", loss, self.iter_step)
            self.writer.add_scalar("Loss/color_loss", color_fine_loss, self.iter_step)
            self.writer.add_scalar("Loss/eikonal_loss", eikonal_loss, self.iter_step)
            self.writer.add_scalar("Statistics/s_val", s_val.mean(), self.iter_step)
            self.writer.add_scalar("Statistics/cdf", (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/weight_max", (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar("Statistics/psnr", psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print("iter:{:8>d} loss = {} lr={}".format(self.iter_step, loss, self.optimizer.param_groups[0]["lr"]))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if True or self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf["general.recording"]
        os.makedirs(os.path.join(self.base_exp_dir, "recording"), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, "recording", dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == ".py":
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, "recording", "config.conf"))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
        )
        self.nerf_outside.load_state_dict(checkpoint["nerf"])
        self.sdf_network.load_state_dict(checkpoint["sdf_network_fine"])
        self.deviation_network.load_state_dict(checkpoint["variance_network_fine"])
        self.color_network.load_state_dict(checkpoint["color_network_fine"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_step = checkpoint["iter_step"]

        logging.info("End")

    def load_checkpoint_NIR(self, rgb_checkpoint_name=None, nir_checkpoint_name=None):
        print('loading sdf from RGB part')
        if rgb_checkpoint_name is not None:
            checkpoint = torch.load(os.path.join(self.rgb_exp_dir, "checkpoints", rgb_checkpoint_name),
                                    map_location=self.device)
            self.sdf_network.load_state_dict(checkpoint["sdf_network_fine"])

        if nir_checkpoint_name is not None:
            checkpoint = torch.load(os.path.join(self.base_exp_dir, "checkpoints", nir_checkpoint_name),
                                    map_location=self.device)
            print('loading other network from NIR models')
            self.nir_nerf_outside.load_state_dict(checkpoint["nerf"])
            #self.nir_nerf_outside.load_state_dict

            self.deviation_network.load_state_dict(checkpoint["variance_network_fine"])
            self.nir_network.load_state_dict(checkpoint["color_network_fine"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.iter_step = checkpoint["iter_step"]

            logging.info("End")

    def save_checkpoint(self, data_type='rgb'):
        if data_type == 'nir':
            checkpoint = {
                "nerf": self.nir_nerf_outside.state_dict(),
                "sdf_network_fine": self.sdf_network.state_dict(),
                "variance_network_fine": self.deviation_network.state_dict(),
                "color_network_fine": self.nir_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iter_step": self.iter_step,
            }
        elif data_type == 'rgb':
            checkpoint = {
                "nerf": self.nerf_outside.state_dict(),
                "sdf_network_fine": self.sdf_network.state_dict(),
                "variance_network_fine": self.deviation_network.state_dict(),
                "color_network_fine": self.color_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iter_step": self.iter_step,
            }
        else:
            print('data_type:', data_type)

        os.makedirs(os.path.join(self.base_exp_dir, "checkpoints"), exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(
                self.base_exp_dir,
                "checkpoints",
                "ckpt_{:0>6d}.pth".format(self.iter_step),
            ),
        )

    def validate_image(self, idx=-1, resolution_level=-1, data_type='rgb'):
        if idx < 0:
            if data_type == 'rgb':
                idx = np.random.randint(self.dataset.n_RGB)
            elif data_type == 'nir':
                idx = np.random.randint(self.dataset.n_NIR)
            else:
                pass

        print("Validate: iter: {}, camera: {}".format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level, data_type=data_type)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch, self.dataset.scale)
            #background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            if data_type == 'nir':
                background_rgb = torch.zeros([1, 3]) if self.use_white_bkgd else None
                render_out = self.renderer_nir.render(
                    rays_o_batch,
                    rays_d_batch,
                    near,
                    far,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    background_rgb=background_rgb,
                )
                n_samples = self.renderer_nir.n_samples + self.renderer_nir.n_importance
            elif data_type == 'rgb':
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
                render_out = self.renderer.render(
                    rays_o_batch,
                    rays_d_batch,
                    near,
                    far,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    background_rgb=background_rgb,
                )
                n_samples = self.renderer.n_samples + self.renderer.n_importance

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible("color_fine"):
                out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())
            if feasible("gradients") and feasible("weights"):
                #n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out["gradients"] * render_out["weights"][:, :n_samples, None]
                if feasible("inside_sphere"):
                    normals = normals * render_out["inside_sphere"][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        RGB_fine = None
        NIR_fine = None
        if len(out_rgb_fine) > 0:
            #img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 4, -1]) * 256).clip(0, 255)
            #RGB_fine = img_fine[:, :, 0:3, :]
            #NIR_fine = img_fine[:, :, 3:4, :]
            #img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            if data_type == 'rgb':
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            if data_type == 'nir':
                NIR_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
                #img_fine = np.concatenate([NIR_fine, NIR_fine, NIR_fine], axis=2)
                img_fine = NIR_fine

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            if data_type == 'rgb':
                rot = np.linalg.inv(self.dataset.pose_RGB[idx, :3, :3].detach().cpu().numpy())
            elif data_type == 'nir':
                rot = np.linalg.inv(self.dataset.pose_NIR[idx, :3, :3].detach().cpu().numpy())
            else:
                pass
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1]) * 128 + 128).clip(
                0, 255
            )

        os.makedirs(os.path.join(self.base_exp_dir, "validations_fine"), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, "normals"), exist_ok=True)

        #print(normal_img.shape, img_fine.shape)
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                if False:
                    print(img_fine[..., i].shape)
                    print(self.dataset.image_at(idx, resolution_level=resolution_level).shape)
                img_path = os.path.join(
                    self.base_exp_dir,
                    "validations_fine",
                    "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx))
                imggt = self.dataset.image_at(idx, resolution_level=resolution_level, data_type=data_type)
                if len(imggt.shape) == 2:
                    imggt = np.stack([imggt, imggt, imggt], axis=-1)
                elif imggt.shape[-1] == 1:
                    imggt = np.concatenate([imggt, imggt, imggt], axis=-1)
                img = np.concatenate([img_fine[..., i], imggt], axis=0).astype('uint8')
                #print(img.shape)
                self.dataset.image_writer(img_path, img)
                # cv.imwrite(
                #     os.path.join(
                #         self.base_exp_dir,
                #         "validations_fine",
                #         "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx),
                #     ),
                #
                #     np.concatenate(
                #         [
                #             img_fine[..., i],
                #             self.dataset.image_at(idx, resolution_level=resolution_level)[:, :, :3],
                #         ]
                #     ),
                # )
            if len(out_normal_fine) > 0:
                normal_path = os.path.join(self.base_exp_dir,
                                           "normals",
                                           "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx))
                self.dataset.image_writer(normal_path, normal_img[..., i].astype('uint8'))
                # cv.imwrite(
                #     os.path.join(
                #         self.base_exp_dir,
                #         "normals",
                #         "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx),
                #     ),
                #     normal_img[..., i],
                # )

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        if self.dataset.enable_RGB:
            vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max,
                                                                 resolution=resolution, threshold=threshold)
            os.makedirs(os.path.join(self.base_exp_dir, "meshes"), exist_ok=True)
            if world_space:
                vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, "meshes", "{:0>8d}_rgb.ply".format(self.iter_step)))
            logging.info("End")

        if self.dataset.enable_NIR:
            vertices, triangles = self.renderer_nir.extract_geometry(bound_min, bound_max,
                                                                     resolution=resolution, threshold=threshold)
            os.makedirs(os.path.join(self.base_exp_dir, "meshes"), exist_ok=True)

            if world_space:
                vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, "meshes", "{:0>8d}_nir.ply".format(self.iter_step)))

            logging.info("End")

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(
                self.render_novel_image(
                    img_idx_0,
                    img_idx_1,
                    np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                    resolution_level=4,
                )
            )
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(self.base_exp_dir, "render")
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(
                video_dir,
                "{:0>8d}_{}_{}.mp4".format(self.iter_step, img_idx_0, img_idx_1),
            ),
            fourcc,
            30,
            (w, h),
        )

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == "__main__":
    print("Hello Arthur")

    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/base.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcube_threshold", type=float, default=0.0)
    parser.add_argument("--is_continue", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case", type=str, default="")
    parser.add_argument("--nir_case", type=str, default="apple_nir")
    parser.add_argument("--rgb_case", type=str, default="apple_rgb")

    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.nir_case, args.rgb_case, args.is_continue)

    if args.mode == "train":
        #runner.train()
        conf_filename = os.path.basename(args.conf)
        if conf_filename == 'nirrgb.conf':
            #runner.train_RGB()
            runner.train_NIR()
        elif os.path.basename(args.conf) == 'rgb.conf' or os.path.basename(args.conf) == 'rgb_mask.conf':
            runner.train_NIRRGB(data_type='rgb')
        elif os.path.basename(args.conf) == 'nir.conf' or os.path.basename(args.conf) == 'nir_mask.conf':
            runner.train_NIRRGB(data_type='nir')
        elif os.path.basename(args.conf) == 'flash_rgb_real.conf':
            runner.train_NIRRGB(data_type='rgb')
        elif os.path.basename(args.conf) == 'nirrgb_mask.conf':
            runner.train_NIRRGB(data_type='rgb')
        else:
            runner.train_NIRRGB(data_type='rgb')
    elif args.mode == "validate_mesh":
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith("interpolate"):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split("_")
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)

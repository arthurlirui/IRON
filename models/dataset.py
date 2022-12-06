import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import traceback
import imageio
import cv2
import pyexr

import json


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def image_reader(reader_name='imageio'):
    if reader_name == 'imageio':
        def image_imageio(im_name):
            return imageio.v3.imread(im_name)[:, :, :3]
        return image_imageio
    if reader_name == 'opencv':
        def image_opencv(im_name):
            return cv2.imread(im_name)[:, :, :3][:, :, ::-1]
        return image_opencv


def exr_reader(reader_name='pyexr'):
    if reader_name == 'pyexr':
        def pyexr_reader(im_name):
            img = np.power(pyexr.open(im_name).get()[:, :, :3], 1.0 / 2.2)
            #img = img[:, :, ::-1]
            return img
        return pyexr_reader


def exr_writer(writer_name='pyexr'):
    if writer_name == 'pyexr':
        def pyexr_writer(outpath, img):
            return pyexr.write(outpath, img)
        return pyexr_writer


def normal_writer(writer_name='pyexr'):
    if writer_name == 'pyexr':
        def pyexr_writer(outpath, normal):
            return pyexr.write(outpath, normal, channel_names=['X', 'Y', 'Z'])
        return pyexr_writer


def image_writer(writer_name='imageio'):
    if writer_name == 'imageio':
        def writer_imageio(outpath, img):
            return imageio.v3.imwrite(outpath, img)
        return writer_imageio
    if writer_name == 'opencv':
        def writer_opencv(outpath, img):
            return cv2.imwrite(outpath, img)
        return writer_opencv


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print("Load data: Begin")
        self.device = torch.device("cuda")
        self.conf = conf

        self.data_dir = conf.get_string("data_dir")
        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        self.camera_outside_sphere = conf.get_bool("camera_outside_sphere", default=True)

        #self.image_reader = conf.get_string("image_reader", default='imageio')
        #self.image_writer = conf.get_string("image_reader", default='imageio')
        #self.image_reader_exr = conf.get_string("image_reader_exr", default='pyexr')

        # initial image reader: use same image reader for any place
        self.image_reader = image_reader(reader_name='imageio')
        self.image_writer = image_writer(writer_name='imageio')
        # if self.image_reader == 'imageio':
        #     def image_imageio(im_name):
        #         return imageio.v3.imread(im_name)[:, :, :3]
        #
        #     self.image_reader = lambda im_name: image_imageio(im_name)
        # elif self.image_reader == 'opencv':
        #     def image_opencv(im_name):
        #         return cv2.imread(im_name)[:, :, :3][:, :, ::-1]
        #
        #     self.image_reader = lambda im_name: image_opencv(im_name)
        # else:
        #     self.image_reader = None
        self.exr_reader = exr_reader(reader_name='pyexr')
        self.exr_writer = exr_writer(writer_name='pyexr')
        self.normal_writer = exr_writer(writer_name='pyexr')

        # if self.image_writer == 'imageio':
        #     def writer_imageio(outpath, img):
        #         return imageio.v3.imwrite(outpath, img)
        #
        #     self.image_writer = lambda outpath, img: writer_imageio(outpath, img)
        # elif self.image_writer == 'opencv':
        #     def writer_opencv(outpath, img):
        #         return cv2.imwrite(outpath, img)
        #
        #     self.image_writer = lambda outpath, img: writer_opencv(outpath, img)
        # else:
        #     self.image_writer = None

        # self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)  # not used
        # self.near = conf.get_float("near")
        # self.far = conf.get_float("far")

        # import json

        # camera_dict = json.load(open(os.path.join(self.data_dir, "cam_dict_norm.json")))
        # if os.path.exists(os.path.join(self.data_dir, "cam_dict.json")):
        if False:
            camera_dict = json.load(open(os.path.join(self.data_dir, "cam_dict.json")))
        else:
            camera_dict = json.load(open(os.path.join(self.data_dir, "cam_dict_norm.json")))
        for x in list(camera_dict.keys()):
            x = x[:-4] + ".png"
            camera_dict[x]["K"] = np.array(camera_dict[x]["K"]).reshape((4, 4))
            camera_dict[x]["W2C"] = np.array(camera_dict[x]["W2C"]).reshape((4, 4))

        self.camera_dict = camera_dict

        folder_name = self.conf['folder_name'] if 'folder_name' in self.conf else 'image'
        try:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, f"{folder_name}/*.png")))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([self.image_reader(im_name) for im_name in self.images_lis]) / 255.0
            # print('min max:', np.min(self.images_np[:]), np.max(self.images_np[:]))
        except:
            # traceback.print_exc()
            print("Loading png images failed; try loading exr images")
            if False:
                import pyexr
                self.images_lis = sorted(glob(os.path.join(self.data_dir, f"{folder_name}/*.exr")))
                self.n_images = len(self.images_lis)
                # self.images_np = np.clip(
                #     np.power(np.stack([pyexr.open(im_name).get()[:, :, ::-1] for im_name in self.images_lis]),
                #              1.0 / 2.2),
                #     0.0,
                #     1.0,
                # )
                self.images_np = np.clip(np.stack([self.exr_reader(im_name) for im_name in self.images_lis]), 0.0, 1.0)
            else:
                #import imageio
                self.images_lis = sorted(glob(os.path.join(self.data_dir, f"{folder_name}/*.exr")))
                self.n_images = len(self.images_lis)
                self.images_np = np.stack([self.exr_reader(im_name) for im_name in self.images_lis])

        no_mask = True
        if no_mask:
            print("Not using masks")
            self.masks_lis = None
            self.masks_np = np.ones_like(self.images_np)
        else:
            try:
                self.masks_lis = sorted(glob(os.path.join(self.data_dir, "mask/*.png")))
                self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
            except:
                # traceback.print_exc()

                print("Loading mask images failed; try not using masks")
                self.masks_lis = None
                self.masks_np = np.ones_like(self.images_np)

        self.images_np = self.images_np[..., :3]
        self.masks_np = self.masks_np[..., :3]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        self.world_mats_np = []
        for x in self.images_lis:
            x = os.path.basename(x)[:-4] + ".png"
            K = self.camera_dict[x]["K"].astype(np.float32)
            W2C = self.camera_dict[x]["W2C"].astype(np.float32)
            C2W = np.linalg.inv(self.camera_dict[x]["W2C"]).astype(np.float32)
            self.intrinsics_all.append(torch.from_numpy(K))
            self.pose_all.append(torch.from_numpy(C2W))
            self.world_mats_np.append(W2C)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        print("image shape, mask shape: ", self.images.shape, self.masks.shape)
        print("image pixel range: ", self.images.min().item(), self.images.max().item())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.eye(4).astype(np.float32)
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print("Load data: End")

    def load_IRON_dict(self, filename='cam_dict_norm.json'):
        camera_dict = json.load(open(os.path.join(self.data_dir, filename)))
        for x in list(camera_dict.keys()):
            x = x[:-4] + ".png"
            camera_dict[x]["K"] = np.array(camera_dict[x]["K"]).reshape((4, 4))
            camera_dict[x]["W2C"] = np.array(camera_dict[x]["W2C"]).reshape((4, 4))
        return camera_dict

    def load_TCNN_dict(self, filename='transforms.json'):
        camera_dict_raw = json.load(open(os.path.join(self.data_dir, filename)))
        fx = camera_dict_raw['fl_x']
        fy = camera_dict_raw['fl_y']
        cx = camera_dict_raw['cx']
        cy = camera_dict_raw['cy']
        K = np.array([[fx, 0, cx, 0], [0, fy, cx, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        camera_dict = {}
        for frame in camera_dict_raw['frames']:
            file_path = frame['file_path']
            x = os.path.basename(file_path)
            transform_matrix = frame['transform_matrix']
            C2W = transform_matrix
            camera_dict[x]['K'] = K
            camera_dict[x]['C2W'] = C2W
        return camera_dict

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        # rays_v = rays_v * -1
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a

        if True:
            near = mid - 1.0
            far = mid + 1.0
            # print(near, far)

        if False:
            near = mid - 1.0
            far = mid + 1.0
        if False:
            near = mid - 1.5
            far = mid + 0.5
        if False:
            near = self.near
            far = self.far
        if False:
            near = 0.3 * torch.ones_like(mid)
            far = 1.5 * torch.ones_like(mid)
        if False:
            print(mid - 1.0, mid + 1.0)
            near = torch.maximum(mid - 1.0, 0.05 * torch.ones_like(mid))
            far = torch.minimum(mid + 1.0, 2.0 * torch.ones_like(mid))
        return near, far

    def image_at(self, idx, resolution_level):
        if self.images_lis[idx].endswith(".exr"):
            img = np.clip(self.exr_reader(self.images_lis[idx])*256, 0, 255)
            # if False:
            #     import pyexr
            #     img = np.power(pyexr.open(self.images_lis[idx]).get()[:, :, ::-1], 1.0 / 2.2) * 255.0
            # else:
            #     img = imageio.v3.imread(self.images_lis[idx])
        if self.images_lis[idx].endswith(".png"):
            img = self.image_reader(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255).astype(np.uint8)

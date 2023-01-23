import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import traceback
import imageio
import cv2
import pyexr
import json
from models.normalize_cam_dict import get_tf_cams, normalize_cam_dict, get_tf_cams_list


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
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
            return cv2.imwrite(outpath, img[:, :, ::-1])
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
        self.image_reader = image_reader(reader_name='opencv')
        self.image_writer = image_writer(writer_name='opencv')
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
            self.images_lis = sorted(glob.glob(os.path.join(self.data_dir, f"{folder_name}/*.png")))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([self.image_reader(im_name) for im_name in self.images_lis]) / 255.0
            # print('min max:', np.min(self.images_np[:]), np.max(self.images_np[:]))
        except:
            # traceback.print_exc()
            print("Loading png images failed; try loading exr images")
            if False:
                import pyexr
                self.images_lis = sorted(glob.glob(os.path.join(self.data_dir, f"{folder_name}/*.exr")))
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
                self.images_lis = sorted(glob.glob(os.path.join(self.data_dir, f"{folder_name}/*.exr")))
                self.n_images = len(self.images_lis)
                self.images_np = np.stack([self.exr_reader(im_name) for im_name in self.images_lis])

        no_mask = True
        if no_mask:
            print("Not using masks")
            self.masks_lis = None
            self.masks_np = np.ones_like(self.images_np)
        else:
            try:
                self.masks_lis = sorted(glob.glob(os.path.join(self.data_dir, "mask/*.png")))
                self.masks_np = np.stack([cv2.imread(im_name) for im_name in self.masks_lis]) / 255.0
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
        return (cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255).astype(np.uint8)


class DatasetNIRRGB:
    def __init__(self, conf):
        super(DatasetNIRRGB, self).__init__()
        print("Loading NIR RGB data: Begin")
        self.device = torch.device("cuda")
        self.conf = conf

        self.data_rgb_dir = conf.get_string('rgb_dir')
        self.data_nir_dir = conf.get_string('nir_dir')
        self.data_dir = conf.get_string("data_dir")
        self.file_type = conf.get_string("file_type")

        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        self.camera_outside_sphere = conf.get_bool("camera_outside_sphere", default=True)

        # initial image reader: use same image reader for any place
        self.image_reader = image_reader(reader_name='opencv')
        self.image_writer = image_writer(writer_name='opencv')
        self.exr_reader = exr_reader(reader_name='pyexr')
        self.exr_writer = exr_writer(writer_name='pyexr')
        self.normal_writer = exr_writer(writer_name='pyexr')

        # load NIR and RGB dataset
        #cam_dict_list = []
        # if False:
        #     if os.path.exists(os.path.join(self.data_rgb_dir, "cam_dict.json")):
        #         rgb_camera_dict = json.load(open(os.path.join(self.data_rgb_dir, "cam_dict.json")))
        #         self.rgb_camera_dict = rgb_camera_dict
        #         cam_dict_list.append(rgb_camera_dict)
        #     if os.path.exists(os.path.join(self.data_nir_dir, "cam_dict.json")):
        #         nir_camera_dict = json.load(open(os.path.join(self.data_nir_dir, "cam_dict.json")))
        #         self.nir_camera_dict = nir_camera_dict
        #         cam_dict_list.append(nir_camera_dict)
        if True:
            self.camera_dict = None
            cam_dict_filename = 'cam_dict_norm.json'
            if os.path.exists(os.path.join(self.data_dir, cam_dict_filename)):
                with open(os.path.join(self.data_dir, cam_dict_filename)) as f:
                    camera_dict = json.load(fp=f)
                    #cam_dict_list.extend(camera_dict)
                    self.camera_dict = camera_dict
                # check non-exist files and remove empty keys
        use_trans = False
        self.scale = 1.0
        if use_trans:
            target_radius = 1.0
            translate, scale = get_tf_cams(self.camera_dict, target_radius=target_radius)
            self.translate = translate
            self.scale = scale
            print('scale:', scale, 'translate:', translate)

        for x in list(self.camera_dict.keys()):
            if os.path.exists(os.path.join(self.data_rgb_dir, x)) or os.path.exists(os.path.join(self.data_nir_dir, x)):
                self.camera_dict[x]["K"] = np.array(self.camera_dict[x]["K"]).reshape((4, 4))
                self.camera_dict[x]["W2C"] = np.array(self.camera_dict[x]["W2C"]).reshape((4, 4))
            else:
                self.camera_dict.pop(x, None)

        #self.camera_dict = camera_dict_list

        for img_name in self.camera_dict:
            W2C = np.array(self.camera_dict[img_name]['W2C']).reshape((4, 4))
            if use_trans:
                W2C = self.transform_pose(W2C, translate, scale)
            assert (np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
            self.camera_dict[img_name]['W2C'] = W2C

        print(self.camera_dict.keys())

        self.enable_RGB = conf['enable_rgb']
        self.enable_NIR = conf['enable_nir']
        if os.path.exists(self.data_rgb_dir):
            try:
                #self.RGB_list = sorted(glob.glob(os.path.join(self.data_rgb_dir, f'*.{self.file_type}')))
                self.RGB_list = sorted(glob.glob(os.path.join(self.data_rgb_dir, f'*.*')))
                self.RGB_list = [f for f in self.RGB_list if os.path.basename(f) in self.camera_dict]
                self.n_RGB = len(self.RGB_list)
                self.RGB_np = np.stack([self.image_reader(im_name) for im_name in self.RGB_list]) / 255.0
            except:
                print("Loading png images failed; try loading exr images")
                self.RGB_list = sorted(glob.glob(os.path.join(self.data_rgb_dir, '*.exr')))
                self.n_RGB = len(self.RGB_list)
                self.RGB_np = np.stack([self.exr_reader(im_name) for im_name in self.RGB_list])
            #self.enable_RGB = True
        else:
            print("No existing RGB dataset")
            self.enable_RGB = False
            self.RGB_list = []
            self.n_RGB = 0
            self.RGB_np = None

        # load NIR images
        if os.path.exists(self.data_nir_dir):
            try:
                #self.NIR_list = sorted(glob.glob(os.path.join(self.data_nir_dir, f'*.{self.file_type}')))
                self.NIR_list = sorted(glob.glob(os.path.join(self.data_nir_dir, f'*.*')))
                self.NIR_list = [f for f in self.NIR_list if os.path.basename(f) in self.camera_dict]
                self.n_NIR = len(self.NIR_list)
                self.NIR_np = np.stack([self.image_reader(im_name) for im_name in self.NIR_list]) / 255.0
            except:
                print("Loading png images failed; try loading exr images")
                self.NIR_list = sorted(glob.glob(os.path.join(self.data_nir_dir, '*.exr')))
                self.n_NIR = len(self.NIR_list)
                self.NIR_np = np.stack([self.exr_reader(im_name) for im_name in self.NIR_list])
            if self.NIR_np.shape[-1] == 1:
                self.NIR_np = np.concatenate([self.NIR_np, self.NIR_np, self.NIR_np], axis=-1)
            #self.enable_NIR = True
        else:
            print("No existing NIR dataset")
            self.enable_NIR = False
            self.NIR_list = []
            self.n_NIR = 0
            self.NIR_np = None

        no_mask = False
        if no_mask:
            print("Not using masks")
            self.masks_lis = None
            if self.enable_RGB:
                self.masks_RGB_np = np.ones_like(self.RGB_np)
            if self.enable_NIR:
                self.masks_NIR_np = np.ones_like(self.NIR_np)
        else:
            try:
                if self.enable_RGB:
                    self.masks_RGB_lis = sorted(glob.glob(os.path.join(self.data_dir, 'rgb', 'masks', f'*.{self.file_type}')))
                    self.masks_RGB_np = np.stack([self.image_reader(im_name) for im_name in self.masks_RGB_lis]) / 255.0
                if self.enable_NIR:
                    self.masks_NIR_lis = sorted(glob.glob(os.path.join(self.data_dir, 'nir', 'masks', f'*.{self.file_type}')))
                    self.masks_NIR_np = np.stack([self.image_reader(im_name) for im_name in self.masks_NIR_lis]) / 255.0
                #self.masks_np = np.stack([cv2.imread(im_name) for im_name in self.masks_lis]) / 255.0
            except:
                # traceback.print_exc()
                print("Loading mask images failed; try not using masks")
                self.masks_lis = None
                self.masks_np = np.ones_like(self.RGB_np)
                self.masks_RGB_np = np.ones_like(self.RGB_np)
                self.masks_NIR_np = np.ones_like(self.NIR_np)

        if self.enable_RGB:
            self.RGB_np = self.RGB_np[..., :3]
            self.masks_RGB_np = self.masks_RGB_np[..., :3]

        if self.enable_NIR:
            self.NIR_np = self.NIR_np[..., :3]
            self.masks_NIR_np = self.masks_NIR_np[..., :3]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.n_RGB+self.n_NIR)]

        if self.enable_RGB:
            self.intrinsics_RGB = []
            self.pose_RGB = []
            self.world_mats_RGB_np = []
            for x in self.RGB_list:
                x = os.path.basename(x)
                K = self.camera_dict[x]["K"].astype(np.float32)
                W2C = self.camera_dict[x]["W2C"].astype(np.float32)
                C2W = np.linalg.inv(self.camera_dict[x]["W2C"]).astype(np.float32)
                self.intrinsics_RGB.append(torch.from_numpy(K))
                self.pose_RGB.append(torch.from_numpy(C2W))
                self.world_mats_RGB_np.append(W2C)
            self.RGB_images = torch.from_numpy(self.RGB_np.astype(np.float32)).cpu()  # [n_RGB, H, W, 3]
            self.RGB_masks = torch.from_numpy(self.masks_RGB_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]

        if self.enable_NIR:
            self.intrinsics_NIR = []
            self.pose_NIR = []
            self.world_mats_NIR_np = []
            for x in self.NIR_list:
                x = os.path.basename(x)
                K = self.camera_dict[x]["K"].astype(np.float32)
                W2C = self.camera_dict[x]["W2C"].astype(np.float32)
                C2W = np.linalg.inv(self.camera_dict[x]["W2C"]).astype(np.float32)
                self.intrinsics_NIR.append(torch.from_numpy(K))
                self.pose_NIR.append(torch.from_numpy(C2W))
                self.world_mats_NIR_np.append(W2C)
            self.NIR_images = torch.from_numpy(self.NIR_np.astype(np.float32)).cpu()  # [n_NIR, H, W, 3]
            self.NIR_masks = torch.from_numpy(self.masks_NIR_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]

        if False:
            print("image shape, mask shape: ", self.RGB_images.shape, self.NIR_images.shape,
                  self.RGB_masks.shape, self.NIR_masks.shape)
            print("image pixel range: ", self.RGB_images.min().item(), self.RGB_images.max().item(),
                  self.NIR_images.min().item(), self.NIR_images.max().item())

        if self.enable_RGB:
            self.intrinsics_RGB = torch.stack(self.intrinsics_RGB).to(self.device)  # [n_RGB, 4, 4]
            self.intrinsics_RGB_inv = torch.inverse(self.intrinsics_RGB)  # [n_RGB, 4, 4]
            self.pose_RGB = torch.stack(self.pose_RGB).to(self.device)  # [n_RGB, 4, 4]
            self.focal = self.intrinsics_RGB[0][0, 0]

        if self.enable_NIR:
            self.intrinsics_NIR = torch.stack(self.intrinsics_NIR).to(self.device)  # [n_NIR, 4, 4]
            self.intrinsics_NIR_inv = torch.inverse(self.intrinsics_NIR)  # [n_NIR, 4, 4]
            self.pose_NIR = torch.stack(self.pose_NIR).to(self.device)  # [n_NIR, 4, 4]
            self.focal = self.intrinsics_NIR[0][0, 0]

        if self.enable_RGB:
            self.H, self.W = self.RGB_images.shape[1], self.RGB_images.shape[2]

        if self.enable_NIR:
            self.H, self.W = self.NIR_images.shape[1], self.NIR_images.shape[2]

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

    def transform_pose(self, W2C, translate, scale):
        return transform_pose(W2C, translate, scale)
        # C2W = np.linalg.inv(W2C)
        # cam_center = C2W[:3, 3]
        # cam_center = (cam_center + translate) * scale
        # C2W[:3, 3] = cam_center
        # return np.linalg.inv(C2W)

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

    def gen_rays_at_nir(self, img_idx, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_NIR_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_NIR[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_NIR[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at_rgb(self, img_idx, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_RGB_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_RGB[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_RGB[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at(self, img_idx, resolution_level=1, data_type='rgb'):
        """
        Generate rays at world space from one camera.
        """
        # l = resolution_level
        # tx = torch.linspace(0, self.W - 1, self.W // l)
        # ty = torch.linspace(0, self.H - 1, self.H // l)
        # pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        # p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        # rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        # return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
        if data_type == 'rgb' and self.enable_RGB:
            return self.gen_rays_at_rgb(img_idx, resolution_level=resolution_level)
        elif data_type == 'nir' and self.enable_NIR:
            return self.gen_rays_at_nir(img_idx, resolution_level=resolution_level)
        else:
            return None

    def gen_random_rays_at_RGB(self, img_idx, batch_size):
        """
            Generate random rays at world space from one camera for RGB dataset
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.RGB_images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.RGB_masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_RGB_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_RGB[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        # rays_v = rays_v * -1
        rays_o = self.pose_RGB[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

    def gen_random_rays_at_NIR(self, img_idx, batch_size):
        """
            Generate random rays at world space from one camera for NIR dataset
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.NIR_images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.NIR_masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_NIR_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_NIR[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        # rays_v = rays_v * -1
        rays_o = self.pose_NIR[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

    def gen_random_rays_at(self, img_idx, batch_size, data_type='rgb'):
        """
        Generate random rays at world space from one camera.
        """
        # pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        # pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        # color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        # mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        # p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        # rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        # # rays_v = rays_v * -1
        # rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        # return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10
        if data_type == 'rgb':
            return self.gen_random_rays_at_RGB(img_idx=img_idx, batch_size=batch_size)
        elif data_type == 'nir':
            return self.gen_random_rays_at_NIR(img_idx=img_idx, batch_size=batch_size)
        else:
            return None

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1, data_type='rgb'):
        """
        Interpolate pose between two cameras.
        """
        if data_type == 'rgb':
            intrinsics_all_inv = self.intrinsics_RGB_inv
            pose_all = self.pose_RGB
        elif data_type == 'nir':
            intrinsics_all_inv = self.intrinsics_NIR_inv
            pose_all = self.pose_NIR
        else:
            pass
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = pose_all[idx_0, :3, 3] * (1.0 - ratio) + pose_all[idx_1, :3, 3] * ratio
        pose_0 = pose_all[idx_0].detach().cpu().numpy()
        pose_1 = pose_all[idx_1].detach().cpu().numpy()
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

    def near_far_from_sphere(self, rays_o, rays_d, scale=1.0):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        halfdist = 1.0 / (scale + 0.0001)
        near = mid - 1.0 * halfdist
        far = mid + 1.0 * halfdist
        return near, far

    def image_at(self, idx, resolution_level, data_type='rgb'):
        if data_type == 'rgb':
            if self.RGB_list[idx].endswith(".exr"):
                img = np.clip(self.exr_reader(self.RGB_list[idx])*256, 0, 255)
            if self.RGB_list[idx].endswith(".png"):
                img = self.image_reader(self.RGB_list[idx])
            if self.RGB_list[idx].endswith(".jpg"):
                img = self.image_reader(self.RGB_list[idx])
        elif data_type == 'nir':
            if self.NIR_list[idx].endswith(".exr"):
                img = np.clip(self.exr_reader(self.NIR_list[idx])*256, 0, 255)
            if self.NIR_list[idx].endswith(".png"):
                img = self.image_reader(self.NIR_list[idx])
            if self.NIR_list[idx].endswith(".jpg"):
                img = self.image_reader(self.NIR_list[idx])
        else:
            print('not file find')
        return (cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255).astype(np.uint8)


###### load dataset
def to8b(x):
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


def load_dataset_NIRRGB_alignRGB(datadir, folder_name='images', file_name='cam_dict.json', use_mask=True):
    parpath = os.path.dirname(datadir)
    rgbpath = os.path.join(parpath, 'rgb')
    nirpath = os.path.join(parpath, 'nir')
    imglist = sorted(glob.glob(os.path.join(datadir, folder_name, '*.*')))

    if use_mask:
        maskpath = os.path.join(datadir, 'masks')
        if os.path.exists(maskpath):
            masklist = sorted(glob.glob(os.path.join(maskpath, '*.*')))
    with open(os.path.join(parpath, file_name)) as f:
        cam_dict = json.load(f)

    use_trans = False
    if use_trans:
        target_radius = 1.0
        with open(os.path.join(parpath, file_name)) as f:
            rgb_cam_dict = json.load(f)
        with open(os.path.join(parpath, file_name)) as f:
            nir_cam_dict = json.load(f)
        translate, scale = get_tf_cams(rgb_cam_dict, target_radius=target_radius)

    image_fpaths = []
    gt_images = []
    mask_images = []
    Ks = []
    W2Cs = []
    #imgtype = 'png'
    imreader = None
    imwriter = None

    if len(imglist) > 0:
        x = imglist[0]
        if x.endswith('png') or x.endswith('jpg'):
            imreader = image_reader('opencv')
            imwriter = image_writer('opencv')
        if x.endswith('exr'):
            imreader = exr_reader('pyexr')
            imwriter = exr_writer('pyexr')
    if len(masklist) > 0:
        x = masklist[0]
        if x.endswith('png') or x.endswith('jpg'):
            maskreader = image_reader('opencv')
            maskwriter = image_writer('opencv')
        if x.endswith('exr'):
            maskreader = exr_reader('pyexr')
            maskwriter = exr_writer('pyexr')

    # load file from folder image
    for i, x in enumerate(imglist):
        filename = os.path.basename(x)
        if filename.endswith('png') or filename.endswith('jpg'):
            im = imreader(x)/255.0
        if filename.endswith('exr'):
            im = imreader(x)
        if im.shape[-1] == 1:
            im = np.concatenate([im, im, im], axis=-1)

        fpath = x

        if use_mask:
            maski = maskreader(masklist[i])/255.0
            h, w, d = im.shape
            if d == 1:
                im[maski[:, :, 0] < 0.1] = 0
            else:
                im[maski < 0.1] = 0

        if not filename in cam_dict:
            continue
        K = np.array(cam_dict[filename]["K"]).reshape((4, 4)).astype(np.float32)
        W2C = np.array(cam_dict[filename]["W2C"]).reshape((4, 4)).astype(np.float32)

        if use_trans:
            W2C = transform_pose(W2C, translate, scale)

        image_fpaths.append(fpath)
        gt_images.append(torch.from_numpy(im))
        Ks.append(torch.from_numpy(K))
        W2Cs.append(torch.from_numpy(W2C))
    gt_images = torch.stack(gt_images, dim=0)
    Ks = torch.stack(Ks, dim=0)
    W2Cs = torch.stack(W2Cs, dim=0)
    return image_fpaths, gt_images, Ks, W2Cs


def load_dataset_NIRRGB(datadir, folder_name='rgb', file_name='cam_dict_norm.json'):
    imglist = glob.glob(os.path.join(datadir, folder_name, '*.*'))
    with open(os.path.join(datadir, file_name)) as f:
        cam_dict = json.load(f)

    use_trans = False
    if use_trans:
        target_radius = 1.0
        translate, scale = get_tf_cams(cam_dict, target_radius=target_radius)

    # cam_dict = json.load(open(os.path.join(datadir, "cam_dict_norm.json")))
    # imgnames = list(cam_dict.keys())
    # try:
    #     imgnames = sorted(imgnames, key=lambda x: int(x[:-4]))
    # except:
    #     imgnames = sorted(imgnames)

    image_fpaths = []
    gt_images = []
    Ks = []
    W2Cs = []
    #imgtype = 'png'
    imreader = None
    imwriter = None

    if len(imglist) > 0:
        x = imglist[0]
        if x.endswith('png') or x.endswith('jpg'):
            imreader = image_reader('imageio')
            imwriter = image_writer('imageio')
        if x.endswith('exr'):
            imreader = exr_reader('pyexr')
            imwriter = image_writer('pyexr')

    # load file from folder image
    for x in imglist:
        # if x.endswith('png') or x.endswith('jpg'):
        #    fpath = os.path.join(datadir, 'image', x)
        # if False:
        #     if x[-4:] == imgtype:
        #         fpath = os.path.join(datadir, imgtype, x)
        #     else:
        #         fpath = os.path.join(datadir, imgtype, x[:-4] + '.' + imgtype)
        #     assert fpath[-4:] in [".jpg", ".png"], "must use ldr images as inputs"
        #     im = imageio.v3.imread(fpath).astype(np.float32) / 255.0
        filename = os.path.basename(x)
        if filename.endswith('png') or filename.endswith('jpg'):
            im = imreader(x)/255.0
        if filename.endswith('exr'):
            im = imreader(x)
        fpath = x


        # if True:
        #     filename = x.split('.')[0]
        #     if os.path.exists(os.path.join(datadir, folder_name, filename + '.png')):
        #         fpath = os.path.join(datadir, folder_name, filename + '.png')
        #         im = imageio.v3.imread(fpath).astype(np.float32) / 255.0
        #     elif os.path.exists(os.path.join(datadir, folder_name, filename + '.jpg')):
        #         fpath = os.path.join(datadir, folder_name, filename + '.jpg')
        #         im = imageio.v3.imread(fpath).astype(np.float32) / 255.0
        #     elif os.path.exists(os.path.join(datadir, folder_name, filename + '.exr')):
        #         fpath = os.path.join(datadir, folder_name, filename + '.exr')
        #         im = imageio.v3.imread(fpath)
        #         im = np.clip(np.power(im, 1.0 / 2.2), 0, 1)  # gamma correction
        #     else:
        #         assert fpath[-4:] in [".jpg", ".png", ".exr"], "must use ldr images as inputs"

        if not filename in cam_dict:
            continue
        K = np.array(cam_dict[filename]["K"]).reshape((4, 4)).astype(np.float32)
        W2C = np.array(cam_dict[filename]["W2C"]).reshape((4, 4)).astype(np.float32)

        if use_trans:
            W2C = transform_pose(W2C, translate, scale)

        image_fpaths.append(fpath)
        gt_images.append(torch.from_numpy(im))
        Ks.append(torch.from_numpy(K))
        W2Cs.append(torch.from_numpy(W2C))
    gt_images = torch.stack(gt_images, dim=0)
    Ks = torch.stack(Ks, dim=0)
    W2Cs = torch.stack(W2Cs, dim=0)
    return image_fpaths, gt_images, Ks, W2Cs


def transform_pose(W2C, translate, scale):
    C2W = np.linalg.inv(W2C)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return np.linalg.inv(C2W)


def load_datadir(datadir, folder_name='image'):
    cam_dict = json.load(open(os.path.join(datadir, "cam_dict_norm.json")))
    imgnames = list(cam_dict.keys())
    try:
        imgnames = sorted(imgnames, key=lambda x: int(x[:-4]))
    except:
        imgnames = sorted(imgnames)

    image_fpaths = []
    gt_images = []
    Ks = []
    W2Cs = []
    imgtype = 'png'
    # load file from folder image
    for x in imgnames:
        #if x.endswith('png') or x.endswith('jpg'):
        #    fpath = os.path.join(datadir, 'image', x)
        if False:
            if x[-4:] == imgtype:
                fpath = os.path.join(datadir, imgtype, x)
            else:
                fpath = os.path.join(datadir, imgtype, x[:-4]+'.'+imgtype)
            assert fpath[-4:] in [".jpg", ".png"], "must use ldr images as inputs"
            im = imageio.v3.imread(fpath).astype(np.float32) / 255.0
        if True:
            filename = x.split('.')[0]
            if os.path.exists(os.path.join(datadir, folder_name, filename+'.png')):
                fpath = os.path.join(datadir, folder_name, filename+'.png')
                im = imageio.v3.imread(fpath).astype(np.float32) / 255.0
            elif os.path.exists(os.path.join(datadir, folder_name, filename+'.jpg')):
                fpath = os.path.join(datadir, folder_name, filename + '.jpg')
                im = imageio.v3.imread(fpath).astype(np.float32) / 255.0
            elif os.path.exists(os.path.join(datadir, folder_name, filename+'.exr')):
                fpath = os.path.join(datadir, folder_name, filename + '.exr')
                im = imageio.v3.imread(fpath)
                im = np.clip(np.power(im, 1.0/2.2), 0, 1)    # gamma correction
            else:
                assert fpath[-4:] in [".jpg", ".png", ".exr"], "must use ldr images as inputs"

        K = np.array(cam_dict[x]["K"]).reshape((4, 4)).astype(np.float32)
        W2C = np.array(cam_dict[x]["W2C"]).reshape((4, 4)).astype(np.float32)

        image_fpaths.append(fpath)
        gt_images.append(torch.from_numpy(im))
        Ks.append(torch.from_numpy(K))
        W2Cs.append(torch.from_numpy(W2C))
    gt_images = torch.stack(gt_images, dim=0)
    Ks = torch.stack(Ks, dim=0)
    W2Cs = torch.stack(W2Cs, dim=0)
    return image_fpaths, gt_images, Ks, W2Cs


if __name__ == '__main__':
    from pyhocon import ConfigFactory
    conf_path = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/confs/nirrgb.conf'
    case = 'apple_all'
    rgb_case = 'apple_all'
    nir_case = 'apple_all'
    with open(conf_path) as f:
        conf_text = f.read()
        conf_text = conf_text.replace("CASE_NAME", case)
        #f.close()
    conf = ConfigFactory.parse_string(conf_text)
    conf["dataset.data_dir"] = conf["dataset.data_dir"].replace("CASE_NAME", case)
    conf['dataset']['rgb_dir'] = conf['dataset']['rgb_dir'].replace("RGB_NAME", rgb_case)
    conf['dataset']['nir_dir'] = conf['dataset']['nir_dir'].replace("NIR_NAME", nir_case)
    conf["dataset.rgb_dir"] = conf['dataset']['rgb_dir']
    conf["dataset.nir_dir"] = conf['dataset']['nir_dir']
    base_exp_dir = conf["general.base_exp_dir"]
    os.makedirs(base_exp_dir, exist_ok=True)
    dataset = DatasetNIRRGB(conf["dataset"], dataset_type='nir')
    #print(dataset)
    print(dataset.NIR_images.shape, dataset.RGB_images.shape)
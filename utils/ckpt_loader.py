import glob
import os
import torch
from icecream import ic
import traceback


def load_ckpt(ckpt_path):
    ckpt_fpaths = glob.glob(os.path.join(ckpt_path, "ckpt_*.pth"))
    ckpt = None
    start_step = 0
    if len(ckpt_fpaths) > 0:
        path2step = lambda x: int(os.path.basename(x)[len("ckpt_"): -4])
        ckpt_fpaths = sorted(ckpt_fpaths, key=path2step)
        ckpt_fpath = ckpt_fpaths[-1]
        start_step = path2step(ckpt_fpath)
        ic("Reloading from checkpoint: ", ckpt_fpath)
        ckpt = torch.load(ckpt_fpath, map_location=torch.device("cuda"))
    return ckpt, start_step


def load_pretrained_checkpoint(ckpt_dir, sdf_network, color_network_dict, is_train_rgb=False, is_train_nir=False):
    start_step = -1
    ckpt_fpaths = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pth"))
    if len(ckpt_fpaths) > 0:
        ckpt, start_step = load_ckpt(ckpt_path=ckpt_dir)
        sdf_network.load_state_dict(ckpt["sdf_network"])

        if is_train_rgb:
            network_enable = {'color_network': True, 'diffuse_albedo_network': True, 'specular_albedo_network': True,
                              'point_light_network': True,
                              'metallic_network': False, 'dielectric_network': False,
                              'specular_roughness_network': False,
                              'metallic_eta_network': False, 'metallic_k_network': False,
                              'dielectric_eta_network': False}

        if is_train_nir:
            network_enable = {'color_network': True, 'diffuse_albedo_network': True, 'specular_albedo_network': True,
                              'point_light_network': True,
                              'metallic_network': False, 'dielectric_network': False,
                              'specular_roughness_network': True,
                              'metallic_eta_network': True, 'metallic_k_network': True, 'dielectric_eta_network': True}
        for x in list(color_network_dict.keys()):
            if x in ckpt:
                color_network_dict[x].load_state_dict(ckpt[x])
    return start_step

            # if x in ckpt:
            #     if args.train_nir:
            #         if not network_enable[x]:
            #             color_network_dict[x].load_state_dict(ckpt_nir[x])
            #     else:
            #         color_network_dict[x].load_state_dict(ckpt[x])
            #
            #     if not network_enable[x]:
            #         print(x)
            #         for para in color_network_dict[x].parameters():
            #             para.requires_grad = False


def load_neus_checkpoint(neus_ckpt_fpath, sdf_network, color_network_dict, load_diffuse_albedo=True):
    #if os.path.isfile(args.neus_ckpt_fpath):
    if os.path.isfile(neus_ckpt_fpath):
        ic(f"Loading from neus checkpoint: {neus_ckpt_fpath}")
        ckpt = torch.load(neus_ckpt_fpath, map_location=torch.device("cuda"))
        try:
            sdf_network.load_state_dict(ckpt["sdf_network_fine"])
            # load sdf, train diffuse albedo and specular albedo for RGB images
            if load_diffuse_albedo:
                color_network_dict["diffuse_albedo_network"].load_state_dict(ckpt["color_network_fine"])
            # color_network_dict["specular_albedo_network"].load_state_dict(ckpt["color_network_fine"])
        except:
            traceback.print_exc()


def download_blender():
    blender_fpath = "./blender-3.1.0-linux-x64/blender"
    if not os.path.isfile(blender_fpath):
        cmd_str = "wget https://mirror.clarkson.edu/blender/release/Blender3.1/blender-3.1.0-linux-x64.tar.xz"
        cmd_str += "&& tar -xvf blender-3.1.0-linux-x64.tar.xz"
        # os.system("wget https://mirror.clarkson.edu/blender/release/Blender3.1/blender-3.1.0-linux-x64.tar.xz && tar -xvf blender-3.1.0-linux-x64.tar.xz")
        os.system(cmd_str)
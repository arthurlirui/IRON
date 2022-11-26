import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

import json
import tinycudann as tcnn

class TCNNSDF(nn.Module):
    def __init__(
        self,
        conf_path, n_inputs_dims=3, n_outputs_dims=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super(TCNNSDF, self).__init__()
        with open(conf_path) as f:
            config = json.load(f)

        self.sdf_encoding = tcnn.Encoding(n_inputs_dims, config["encoding"])
        self.sdf_network = tcnn.Network(self.sdf_encoding.n_output_dims, n_outputs_dims, config["network"])
        self.model = torch.nn.Sequential(self.sdf_encoding, self.sdf_network)

    def forward(self, inputs):
        return self.model(inputs)

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients

    def get_all(self, x, is_training=True):
        with torch.enable_grad():
            x.requires_grad_(True)
            tmp = self.forward(x)
            y, feature = tmp[..., :1], tmp[..., 1:]

            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=is_training,
                retain_graph=is_training,
                only_inputs=True,
            )[0]
        if not is_training:
            return y.detach(), feature.detach(), gradients.detach()
        return y, feature, gradients


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients

    def get_all(self, x, is_training=True):
        with torch.enable_grad():
            x.requires_grad_(True)
            tmp = self.forward(x)
            y, feature = tmp[..., :1], tmp[..., 1:]

            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=is_training,
                retain_graph=is_training,
                only_inputs=True,
            )[0]
        if not is_training:
            return y.detach(), feature.detach(), gradients.detach()
        return y, feature, gradients


class TCNNRendering(nn.Module):
    def __init__(self, conf_path, n_inputs_dims=3):
        super(TCNNRendering, self).__init__()
        with open(conf_path, 'r') as f:
            config = json.load(f)

        self.pts_encoding = tcnn.Encoding(n_inputs_dims, config["pts_encoding"])
        self.dir_encoding = tcnn.Encoding(n_inputs_dims, config["dir_encoding"])
        self.network = tcnn.Network(self.pts_encoding.n_output_dims, config["network"])

    def forward(self, points, normals, view_dirs, feature_vectors):
        pass


class TCNNNeRF(nn.Module):
    def __init__(self, conf_path, n_input_dims=3):
        super(TCNNNeRF, self).__init__()
        with open(conf_path) as f:
            config = json.load(f)
        self.pos_encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.pos_network = tcnn.Network(self.pos_encoding.n_output_dims, 1, config["network"])
        self.pos_model = torch.nn.Sequential(self.pos_network)

        self.dir_encoding = tcnn.Encoding(n_input_dims, config["dir_encoding"])

        rgb_network_input_dims = self.pos_encoding.n_output_dims+self.dir_encoding.n_output_dims
        self.rgb_network = tcnn.Network(rgb_network_input_dims, 3, config["rgb_network"])
        self.rgb_model = torch.nn.Sequential(self.rgb_network)
        #self.density_network = torch.nn.Sequential(self.pos_encoding, self.pos_network)
        #self.cmodel = torch.nn.ModuleList([rgb_network_input_dims, ])
        #self.model = tcnn.NetworkWithInputEncoding(n_input_dims, n_output_dims, config["encoding"], config["network"])

    def parameters(self):
        tt = []
        tt += list(self.pos_model.parameters())
        tt += list(self.rgb_model.parameters())
        return tt

    def forward(self, pos_inputs, dir_inputs):
        pos_feat = self.pos_encoding(pos_inputs)
        dir_feat = self.dir_encoding(dir_inputs)

        density = self.pos_model(pos_feat)
        rgb = self.rgb_model(torch.cat([pos_feat, dir_feat], dim=-1))
        return rgb, density
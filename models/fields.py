import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

import json
import tinycudann as tcnn


class tinySDFNetwork():
    __author__ = 'Arthur'
    def __init__(self):
        pass

    def forward(self, inputs):
        pass


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




# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(
        self,
        d_feature,
        mode,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        weight_norm=True,
        multires=0,
        multires_view=0,
        squeeze_out=True,
        squeeze_out_scale=1.0,
        output_bias=0.0,
        output_scale=1.0,
        skip_in=(),
    ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.output_bias = output_bias
        self.output_scale = output_scale
        self.squeeze_out_scale = squeeze_out_scale

    def forward(self, points, normals, view_dirs, feature_vectors):

        if self.embed_fn is not None:
            points = self.embed_fn(points)

        if self.embedview_fn is not None and self.mode != "no_view_dir":
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == "idr":
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "no_view_dir":
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == "no_normal":
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.output_scale * (x + self.output_bias)
        if self.squeeze_out:
            x = self.squeeze_out_scale * torch.sigmoid(x)

        return x


# Implementation for tiny-cuda-nn
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
        return list(self.rgb_model.parameters())+list(self.pos_model.parameters())

    def forward(self, pos_inputs, dir_inputs):
        pos_feat = self.pos_encoding(pos_inputs)
        dir_feat = self.dir_encoding(dir_inputs)

        density = self.pos_model(pos_feat)
        rgb = self.rgb_model(torch.cat([pos_feat, dir_feat], dim=-1))
        return rgb, density


class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        d_in=3,
        d_in_view=3,
        multires=0,
        multires_view=0,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]
            + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)]
        )

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


if __name__ == '__main__':
    pass

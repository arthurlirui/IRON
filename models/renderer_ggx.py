import glob

import torch
import torch.nn as nn
import numpy as np
import os

### https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/libcore/util.cpp#L651


### https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L477
def smithG1(cosTheta, alpha):
    sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
    tanTheta = sinTheta / (cosTheta + 1e-10)
    root = alpha * tanTheta
    return 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root)))


### https://github.com/mitsuba-renderer/mitsuba3/blob/152352f87b5baea985511b2a80d9f91c3c945a90/src/bsdfs/principledhelpers.h
def calc_dist_params(anisotropic, roughness, has_anisotropic):
    roughness_2 = torch.sqrt(roughness)
    if not has_anisotropic:
        alpha = torch.clamp(roughness_2, min=0.0001)
        return alpha, alpha
    aspect = torch.sqrt(1.0-0.9*anisotropic)
    alpha_u = torch.clamp(roughness_2/aspect, 0.0001)
    alpha_v = torch.clamp(roughness_2*aspect, 0.0001)
    return alpha_u, alpha_v


class CoLocRenderer(nn.Module):
    def __init__(self, rough_plastic, dielectric, conductor, smooth_conductor, use_cuda=False):
        super().__init__()
        self.rough_plastic_renderer = rough_plastic
        self.dielectric_renderer = dielectric
        self.rough_conductor_renderer = conductor
        self.smooth_conductor_renderer = smooth_conductor

    def forward(self, light, distance, normal, viewdir, params={}):
        diffuse_albedo, specular_albedo, alpha = params['diffuse_albedo'], params['specular_albedo'], params['specular_roughness']
        material_vector = params['material_vector']

        res_rp = self.rough_plastic_renderer(light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha)
        res_di = self.dielectric_renderer(light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha)
        res_rc = self.rough_conductor_renderer(light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha)
        res_sc = self.smooth_conductor_renderer(light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha)

        diffuse_rgb = material_vector[..., 0:1] * res_rp["diffuse_rgb"]
        diffuse_rgb += material_vector[..., 1:2] * res_di["diffuse_rgb"]
        diffuse_rgb += material_vector[..., 2:3] * res_rc["diffuse_rgb"]
        diffuse_rgb += material_vector[..., 3:4] * res_sc["diffuse_rgb"]

        specular_rgb = material_vector[..., 0:1] * res_rp["specular_rgb"]
        specular_rgb += material_vector[..., 1:2] * res_di["specular_rgb"]
        specular_rgb += material_vector[..., 2:3] * res_rc["specular_rgb"]
        specular_rgb += material_vector[..., 3:4] * res_sc["specular_rgb"]
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb, "material_map": material_vector}
        return ret


class GGXColocatedRenderer(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        self.MTS_TRANS = torch.from_numpy(
            np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/ext_mts_rtrans_data.txt")).astype(
                np.float32
            )
        )  # 5000 entries, external IOR
        self.MTS_DIFF_TRANS = torch.from_numpy(
            np.loadtxt(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/int_mts_diff_rtrans_data.txt")
            ).astype(np.float32)
        )  # 50 entries, internal IOR
        self.num_theta_samples = 100
        self.num_alpha_samples = 50

        if use_cuda:
            self.MTS_TRANS = self.MTS_TRANS.cuda()
            self.MTS_DIFF_TRANS = self.MTS_DIFF_TRANS.cuda()

    def forward(self, light, distance, normal, viewdir, params={}):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        """
        diffuse_albedo = params['diffuse_albedo']
        specular_albedo = params['specular_albedo']
        alpha = params['specular_roughness']
        # decay light according to squared-distance falloff
        light_intensity = light / (distance * distance + 1e-10)

        # <wo, n> = <w_i, n> = <h, n> in colocated setting
        dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
        # default value of IOR['polypropylene'] / IOR['air'].
        m_eta = 1.48958738
        m_invEta2 = 1.0 / (m_eta * m_eta)

        # clamp alpha for numeric stability
        alpha = torch.clamp(alpha, min=0.0001)

        # specular term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L347
        ## compute GGX NDF: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L191
        cosTheta2 = dot * dot
        root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)
        D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
        ## compute fresnel: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/libcore/util.cpp#L651
        # F = 0.04
        F = 0.03867

        ## compute shadowing term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L520
        G = smithG1(dot, alpha) ** 2  # [..., 1]

        specular_rgb = light_intensity * specular_albedo * F * D * G / (4.0 * dot + 1e-10)

        # diffuse term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L367
        ## compute T12: : https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L183
        ### data_file: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L93
        ### assume eta is fixed
        warpedCosTheta = dot**0.25
        alphaMin, alphaMax = 0, 4
        warpedAlpha = ((alpha - alphaMin) / (alphaMax - alphaMin)) ** 0.25  # [..., 1]
        tx = torch.floor(warpedCosTheta * self.num_theta_samples).long()
        ty = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        t_idx = ty * self.num_theta_samples + tx

        dots_sh = list(t_idx.shape[:-1])
        data = self.MTS_TRANS.view([1] * len(dots_sh) + [-1]).expand(dots_sh + [-1])

        t_idx = torch.clamp(t_idx, min=0, max=data.shape[-1] - 1).long()  # important
        T12 = torch.clamp(torch.gather(input=data, index=t_idx, dim=-1), min=0.0, max=1.0)
        T21 = T12  # colocated setting

        ## compute Fdr: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L249
        t_idx = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        data = self.MTS_DIFF_TRANS.view([1] * len(dots_sh) + [-1]).expand(dots_sh + [-1])
        t_idx = torch.clamp(t_idx, min=0, max=data.shape[-1] - 1).long()  # important
        Fdr = torch.clamp(1.0 - torch.gather(input=data, index=t_idx, dim=-1), min=0.0, max=1.0)  # [..., 1]

        diffuse_rgb = light_intensity * (diffuse_albedo / (1.0 - Fdr + 1e-10) / np.pi) * dot * T12 * T21 * m_invEta2
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb}
        return ret


class SmoothDielectricRenderer(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        if False:
            self.MTS_TRANS = torch.from_numpy(
                np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/ext_mts_rtrans_data.txt")).astype(
                    np.float32
                )
            )  # 5000 entries, external IOR
            self.MTS_DIFF_TRANS = torch.from_numpy(
                np.loadtxt(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/int_mts_diff_rtrans_data.txt")
                ).astype(np.float32)
            )  # 50 entries, internal IOR
            self.num_theta_samples = 100
            self.num_alpha_samples = 50

            if use_cuda:
                self.MTS_TRANS = self.MTS_TRANS.cuda()
                self.MTS_DIFF_TRANS = self.MTS_DIFF_TRANS.cuda()

    def forward(self, light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        """
        # decay light according to squared-distance falloff
        light_intensity = light / (distance * distance + 1e-10)

        # <wo, n> = <w_i, n> = <h, n> in colocated setting
        dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
        # default value of IOR['air'] / IOR['bk7'].
        m_eta = 1.5046
        #m_invEta2 = 1.0 / (m_eta * m_eta)
        m_invEta = 1.0 / m_eta

        # clamp alpha for numeric stability
        #alpha = torch.clamp(alpha, min=0.0001)

        # specular term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L347
        ## compute GGX NDF: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L191
        #cosTheta2 = dot * dot
        #root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)
        #D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
        ## compute fresnel: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/libcore/util.cpp#L651
        # for smooth dielectric surface
        F = 0.04
        specular_rgb = light_intensity * specular_albedo * F
        diffuse_rgb = light_intensity * diffuse_albedo * 0.0001
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb}
        return ret


class ThinDielectricRenderer(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        if False:
            self.MTS_TRANS = torch.from_numpy(
                np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/ext_mts_rtrans_data.txt")).astype(
                    np.float32
                )
            )  # 5000 entries, external IOR
            self.MTS_DIFF_TRANS = torch.from_numpy(
                np.loadtxt(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/int_mts_diff_rtrans_data.txt")
                ).astype(np.float32)
            )  # 50 entries, internal IOR
            self.num_theta_samples = 100
            self.num_alpha_samples = 50

            if use_cuda:
                self.MTS_TRANS = self.MTS_TRANS.cuda()
                self.MTS_DIFF_TRANS = self.MTS_DIFF_TRANS.cuda()

    def forward(self, light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        """
        # decay light according to squared-distance falloff
        light_intensity = light / (distance * distance + 1e-10)

        # <wo, n> = <w_i, n> = <h, n> in colocated setting
        dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
        # default value of IOR['air'] / IOR['bk7'].
        m_eta = 1.5046
        #m_invEta2 = 1.0 / (m_eta * m_eta)
        m_invEta = 1.0 / m_eta

        # clamp alpha for numeric stability
        #alpha = torch.clamp(alpha, min=0.0001)

        # specular term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L347
        ## compute GGX NDF: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L191
        #cosTheta2 = dot * dot
        #root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)
        #D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
        ## compute fresnel: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/libcore/util.cpp#L651
        # for smooth dielectric surface

        # fresnelDielectricExt
        R = 0.04
        T = 1 - R
        if R < 1:
            R += T * T * R / (1 - R * R)
        specular_rgb = light_intensity * specular_albedo * R
        diffuse_rgb = light_intensity * diffuse_albedo * 0.0001
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb}
        return ret


class SmoothConductorCoLocRenderer(nn.Module):
    def __init__(self, ior_path, eta=2.580000, k=8.210000, use_cuda=False):
        super().__init__()

        # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#conductor-ior-list
        eta_list = glob.glob(os.path.join(ior_path, '*.eta.spd'))
        k_list = glob.glob(os.path.join(ior_path, '*.k.spd'))
        # default value is set to be in 850nm illumination for Au
        # Cu: eta=855.063293 0.280000 k=855.063293 5.485625
        # Au: eta=855.063293 0.198125 k=855.063293 5.631250
        # Al: eta=855.063293 2.580000 k=855.063293 8.210000
        self.eta = eta
        self.k = k

    # def fresnelConductorExact(self, cosThetaI, eta, k):
    #     cosThetaI2 = cosThetaI*cosThetaI
    #     sinThetaI2 = 1 - cosThetaI2
    #     sinThetaI4 = sinThetaI2 * sinThetaI2
    #     temp1 = eta*eta - k*k - sinThetaI2
    #     a2pb2 = torch.sqrt(temp1*temp1 + 4*k*k*eta*eta)
    #     a = torch.sqrt(0.5*(a2pb2 + temp1))
    #     term1 = a2pb2 + cosThetaI2
    #     term2 = 2*a*cosThetaI
    #     Rs2 = (term1 - term2) / (term1 + term2)
    #     term3 = a2pb2 * cosThetaI2 + sinThetaI4
    #     term4 = term2 * sinThetaI2
    #     Rp2 = Rs2 * (term3 - term4) / (term3 + term4)
    #     return 0.5 * (Rp2 + Rs2)

    def forward(self, light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha=None):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        ### https://github.com/mitsuba-renderer/mitsuba/blob/master/src/bsdfs/conductor.cpp
        """
        # decay light according to squared-distance falloff
        light_intensity = light / (distance * distance + 1e-10)

        dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999

        #m_eta = 1.48958738
        F = fresnel_conductor_exact(cosThetaI=dot, eta=self.eta, k=self.k)
        specular_rgb = light_intensity * specular_albedo * F
        diffuse_rgb = light_intensity * diffuse_albedo * 0.0001
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb}
        return ret


class RoughConductorCoLocRenderer(nn.Module):
    def __init__(self, ior_path, eta=2.580000, k=8.210000, use_cuda=False):
        super().__init__()

        # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#conductor-ior-list
        eta_list = glob.glob(os.path.join(ior_path, '*.eta.spd'))
        k_list = glob.glob(os.path.join(ior_path, '*.k.spd'))
        # default value is set to be in 850nm illumination for Au
        # Cu: eta=855.063293 0.280000 k=855.063293 5.485625
        # Au: eta=855.063293 0.198125 k=855.063293 5.631250
        # Al: eta=855.063293 2.580000 k=855.063293 8.210000
        self.eta = eta
        self.k = k

    # def fresnelConductorExact(self, cosThetaI, eta, k):
    #     cosThetaI2 = cosThetaI*cosThetaI
    #     sinThetaI2 = 1 - cosThetaI2
    #     sinThetaI4 = sinThetaI2 * sinThetaI2
    #     temp1 = eta*eta - k*k - sinThetaI2
    #     a2pb2 = torch.sqrt(temp1*temp1 + 4*k*k*eta*eta)
    #     a = torch.sqrt(0.5*(a2pb2 + temp1))
    #     term1 = a2pb2 + cosThetaI2
    #     term2 = 2*a*cosThetaI
    #     Rs2 = (term1 - term2) / (term1 + term2)
    #     term3 = a2pb2 * cosThetaI2 + sinThetaI4
    #     term4 = term2 * sinThetaI2
    #     Rp2 = Rs2 * (term3 - term4) / (term3 + term4)
    #     return 0.5 * (Rp2 + Rs2)

    def forward(self, light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha=None):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        ### https://github.com/mitsuba-renderer/mitsuba/blob/master/src/bsdfs/conductor.cpp
        ### https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h
        """
        # decay light according to squared-distance falloff
        light_intensity = light / (distance * distance + 1e-10)

        #H = viewdir
        # <wo, n> = <w_i, n> = <h, n> in colocated setting
        dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
        # default value of IOR['polypropylene'] / IOR['air'].
        #m_eta = 1.48958738
        #m_invEta2 = 1.0 / (m_eta * m_eta)

        # clamp alpha for numeric stability
        alpha = torch.clamp(alpha, min=0.0001)

        # specular term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L347
        ## compute GGX NDF: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L191
        cosTheta2 = dot * dot

        root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)
        D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)

        ## compute fresnel: https://github.com/mitsuba-renderer/mitsuba/blob/10af06f365886c1b6dd8818e0a3841078a62f283/src/bsdfs/roughconductor.cpp#L284
        # F = 0.04
        # F = 0.03867
        ## dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        #dotwih = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        F = fresnel_conductor_exact(cosThetaI=dot, eta=self.eta, k=self.k)

        ## compute shadowing term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L520
        G = smithG1(dot, alpha) ** 2  # [..., 1]

        specular_rgb = light_intensity * specular_albedo * F * D * G / (4.0 * dot + 1e-10)
        diffuse_rgb = light_intensity * diffuse_albedo * 0.0001
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb}
        return ret


def fresnel_dielectric(cosThetaI, cosThetaT, eta):
    #if eta == 1:
    #    cosThetaT = -1*cosThetaI
    #    return 0
    scale = torch.ones_like(cosThetaI) * eta
    mask_costheta = cosThetaI > 0
    scale[mask_costheta] = 1.0/eta[mask_costheta]
    cosThetaTSqr = 1 - (1-cosThetaI**2)*(scale**2)
    #if cosThetaTSqr <= 0.0:
    #    cosThetaT = 0
    #    return 1.0
    #cosThetaI = np.abs(cosThetaI)
    cosThetaI = torch.abs(cosThetaI)
    #cosThetaT = np.sqrt(cosThetaTSqr)
    cosThetaT = torch.sqrt(cosThetaTSqr)
    Rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT)
    Rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT)
    #cosThetaT = -1*cosThetaT if cosThetaI > 0 else cosThetaT
    return 0.5 * (Rs * Rs + Rp * Rp)


def fresnel_conductor_exact(cosThetaI, eta, k):
    cosThetaI2 = cosThetaI*cosThetaI
    sinThetaI2 = 1 - cosThetaI2
    sinThetaI4 = sinThetaI2 * sinThetaI2
    temp1 = eta*eta - k*k - sinThetaI2
    a2pb2 = torch.sqrt(temp1*temp1 + 4*k*k*eta*eta)
    a = torch.sqrt(0.5*(a2pb2 + temp1))
    term1 = a2pb2 + cosThetaI2
    term2 = 2*a*cosThetaI
    Rs2 = (term1 - term2) / (term1 + term2)
    term3 = a2pb2 * cosThetaI2 + sinThetaI4
    term4 = term2 * sinThetaI2
    Rp2 = Rs2 * (term3 - term4) / (term3 + term4)
    return 0.5 * (Rp2 + Rs2)


class RoughPlasticCoLocRenderer(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        self.MTS_TRANS = torch.from_numpy(
            np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "ggx/ext_mts_rtrans_data.txt")).astype(np.float32)
        )  # 5000 entries, external IOR
        self.MTS_DIFF_TRANS = torch.from_numpy(
            np.loadtxt(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/int_mts_diff_rtrans_data.txt")
            ).astype(np.float32)
        )  # 50 entries, internal IOR
        self.num_theta_samples = 100
        self.num_alpha_samples = 50

        if use_cuda:
            self.MTS_TRANS = self.MTS_TRANS.cuda()
            self.MTS_DIFF_TRANS = self.MTS_DIFF_TRANS.cuda()

    def forward(self, light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        """
        # decay light according to squared-distance falloff
        light_intensity = light / (distance * distance + 1e-10)

        # <wo, n> = <w_i, n> = <h, n> in colocated setting
        dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
        # default value of IOR['polypropylene'] / IOR['air'].
        m_eta = 1.48958738
        m_invEta2 = 1.0 / (m_eta * m_eta)

        # clamp alpha for numeric stability
        alpha = torch.clamp(alpha, min=0.0001)

        # specular term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L347
        ## compute GGX NDF: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L191
        cosTheta2 = dot * dot
        root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)
        D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
        ## compute fresnel: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/libcore/util.cpp#L651
        # F = 0.04
        #F = 0.03867
        F = fresnel_dielectric(cosThetaI=dot, cosThetaT=dot, eta=m_eta)

        ## compute shadowing term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L520
        G = smithG1(dot, alpha) ** 2  # [..., 1]

        specular_rgb = light_intensity * specular_albedo * F * D * G / (4.0 * dot + 1e-10)

        # diffuse term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L367
        ## compute T12: : https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L183
        ### data_file: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L93
        ### assume eta is fixed
        warpedCosTheta = dot**0.25
        alphaMin, alphaMax = 0, 4
        warpedAlpha = ((alpha - alphaMin) / (alphaMax - alphaMin)) ** 0.25  # [..., 1]
        tx = torch.floor(warpedCosTheta * self.num_theta_samples).long()
        ty = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        t_idx = ty * self.num_theta_samples + tx

        dots_sh = list(t_idx.shape[:-1])
        data = self.MTS_TRANS.view([1] * len(dots_sh) + [-1]).expand(dots_sh + [-1])

        t_idx = torch.clamp(t_idx, min=0, max=data.shape[-1] - 1).long()  # important
        T12 = torch.clamp(torch.gather(input=data, index=t_idx, dim=-1), min=0.0, max=1.0)
        T21 = T12  # colocated setting

        ## compute Fdr: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L249
        t_idx = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        data = self.MTS_DIFF_TRANS.view([1] * len(dots_sh) + [-1]).expand(dots_sh + [-1])
        t_idx = torch.clamp(t_idx, min=0, max=data.shape[-1] - 1).long()  # important
        Fdr = torch.clamp(1.0 - torch.gather(input=data, index=t_idx, dim=-1), min=0.0, max=1.0)  # [..., 1]

        diffuse_rgb = light_intensity * (diffuse_albedo / (1.0 - Fdr + 1e-10) / np.pi) * dot * T12 * T21 * m_invEta2
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb}
        return ret


class CompositeRenderer(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()
        mts_trans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/ext_mts_rtrans_data.txt")
        self.MTS_TRANS = torch.from_numpy(np.loadtxt(mts_trans_path).astype(np.float32))
        # 5000 entries, external IOR
        mts_diff_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/int_mts_diff_rtrans_data.txt")
        self.MTS_DIFF_TRANS = torch.from_numpy(np.loadtxt(mts_diff_path).astype(np.float32))
        # 50 entries, internal IOR
        self.num_theta_samples = 100
        self.num_alpha_samples = 50

        eta_list = glob.glob('./resource/ior/*.eta.spd')
        k_list = glob.glob('./resource/ior/*.k.spd')
        self.wavelength = 850
        self.MATERIAL_ETA, self.MATERIAL_K = {}, {}
        for i, etapath in enumerate(eta_list):
            eta_name = os.path.basename(etapath).split('.')[0]
            self.MATERIAL_ETA[eta_name] = torch.from_numpy(np.loadtxt(etapath))
        for i, kpath in enumerate(k_list):
            k_name = os.path.basename(kpath).split('.')[0]
            self.MATERIAL_K[k_name] = torch.from_numpy(np.loadtxt(kpath))

        #print(self.MATERIAL_ETA)
        #print(self.MATERIAL_K)
        print('eta:', self.get_eta(), 'K:', self.get_K())

        if use_cuda:
            self.MTS_TRANS = self.MTS_TRANS.cuda()
            self.MTS_DIFF_TRANS = self.MTS_DIFF_TRANS.cuda()
            for k in self.MATERIAL_ETA:
                self.MATERIAL_ETA[k] = self.MATERIAL_ETA[k].cuda()
            for k in self.MATERIAL_K:
                self.MATERIAL_K[k] = self.MATERIAL_K[k].cuda()

    def get_eta(self, wavelength=850):
        eta = {}
        for k in self.MATERIAL_ETA:
            index = torch.min(torch.abs(self.MATERIAL_ETA[k][:, 0] - wavelength), dim=0)[1]
            eta[k] = self.MATERIAL_ETA[k][index, :]
        return eta

    def get_K(self, wavelength=850):
        K = {}
        for k in self.MATERIAL_K:
            index = torch.min(torch.abs(self.MATERIAL_ETA[k][:, 0] - wavelength), dim=0)[1]
            K[k] = self.MATERIAL_K[k][index, :]
        return K

    @staticmethod
    def fnmadd(a, b, c):
        return c - a*b

    @staticmethod
    def fmadd(a, b, c):
        return c + a*b

    @staticmethod
    def fresnel_dielectric(cos_theta, eta):
        eta_it = torch.ones_like(cos_theta) * eta
        eta_it[cos_theta < 0] = 1.0 / eta
        eta_ti = torch.ones_like(cos_theta) * eta
        eta_ti[cos_theta > 0] = 1.0 / eta

        cosThetaTSqr = 1.0 - (1.0 - cos_theta ** 2) * (eta_ti ** 2)
        cos_theta_i_abs = torch.abs(cos_theta)
        cos_theta_t_abs = torch.sqrt(cosThetaTSqr)
        Rs = CompositeRenderer.fnmadd(eta_it, cos_theta_t_abs, cos_theta_i_abs) / CompositeRenderer.fmadd(eta_it, cos_theta_t_abs, cos_theta_i_abs)
        Rp = CompositeRenderer.fnmadd(eta_it, cos_theta_i_abs, cos_theta_t_abs) / CompositeRenderer.fmadd(eta_it, cos_theta_i_abs, cos_theta_t_abs)
        return 0.5 * (Rs * Rs + Rp * Rp)

    @staticmethod
    def fresnel_conductor_exact(cos_theta, eta, k):
        cosThetaI2 = cos_theta * cos_theta
        sinThetaI2 = 1 - cosThetaI2
        sinThetaI4 = sinThetaI2 * sinThetaI2
        temp1 = eta * eta - k * k - sinThetaI2
        a2pb2 = torch.sqrt(temp1 * temp1 + 4 * k * k * eta * eta)
        a = torch.sqrt(0.5 * (a2pb2 + temp1))
        term1 = a2pb2 + cosThetaI2
        term2 = 2 * a * cos_theta
        Rs2 = (term1 - term2) / (term1 + term2)
        term3 = a2pb2 * cosThetaI2 + sinThetaI4
        term4 = term2 * sinThetaI2
        Rp2 = Rs2 * (term3 - term4) / (term3 + term4)
        return 0.5 * (Rp2 + Rs2)

    def metallic_reflection(self, cos_theta, eta, k):
        # m_eta = 1.48958738
        F = CompositeRenderer.fresnel_conductor_exact(cos_theta=cos_theta, eta=eta, k=k)
        return F

    def dielectric_reflection(self, cos_theta, eta):
        F = fresnel_dielectric(cos_theta, cos_theta, eta)
        return F

    def main_metallic_reflection(self, cos_theta, eta, k, specular_albedo=1.0):
        F = self.metallic_reflection(cos_theta, eta, k)
        return specular_albedo * F

    def main_dielectric_reflection(self, D, G, cos_theta, eta, specular_albedo=1.0):
        #F = self.dielectric_reflection(cos_theta, eta)
        F = 0.04
        return specular_albedo * F * D * G / (4.0 * torch.abs(cos_theta))

    def main_specular_reflection(self, D, G, F_dielectric, metallic, spec_tint, cos_theta, color, intensity, eta):
        #cos_theta_i = torch.dot(normal, viewdir)
        #cos_theta_i = cos_theta
        F_principled = self.principled_fresnel(F_dielectric=F_dielectric*color,
                                               metallic=metallic,
                                               spec_tint=spec_tint,
                                               base_color=color,
                                               intensity=intensity,
                                               cos_theta=cos_theta,
                                               eta=eta)
        return F_principled * D * G / (4.0 * torch.abs(cos_theta))

    def secondary_isotropic_specular_reflection(self, cos_theta, clearcoat, eta):
        ## https://github.com/mitsuba-renderer/mitsuba3/blob/16b133ecb940346ce17959589e0ce567eb7181e5/src/bsdfs/principled.cpp#L623
        #cos_theta_i = torch.dot(normal, viewdir)
        Fcc = self.calc_F_Clearcoat(cos_theta=cos_theta, eta=eta)
        Dcc = self.calc_D_Clearcoat(cos_theta=cos_theta, clearcoat=clearcoat)
        Gcc = self.calc_G_Clearcoat(cos_theta=cos_theta, alpha_u=0.25, alpha_v=0.25)
        return clearcoat * 0.25 * Fcc * Dcc * Gcc * torch.abs(cos_theta)

    def diffuse_reflection(self, cos_theta, alpha, diffuse_albedo):
        ## https://github.com/mitsuba-renderer/mitsuba3/blob/16b133ecb940346ce17959589e0ce567eb7181e5/src/bsdfs/principled.cpp#L645
        alpha = torch.clamp(alpha, min=0.0001)
        F = self.schlick_weight(torch.abs(cos_theta))
        f_diff = (1.0 - 0.5 * F) * (1.0 - 0.5 * F)
        Rr = 2.0 * alpha * cos_theta * cos_theta
        f_retro = Rr * (F + F + F * F * (Rr - 1.0))
        return torch.abs(cos_theta) * diffuse_albedo * 1/np.pi * (f_diff + f_retro)

    def diffuse_reflection_ggx(self, light_intensity, cos_theta, alpha, diffuse_albedo, eta=1.48958738):
        ### assume eta is fixed
        #m_eta = 1.48958738
        alpha = torch.clamp(alpha, min=0.0001)
        m_invEta2 = 1.0 / (eta * eta)

        warpedCosTheta = cos_theta ** 0.25
        alphaMin, alphaMax = 0, 4
        warpedAlpha = ((alpha - alphaMin) / (alphaMax - alphaMin)) ** 0.25  # [..., 1]
        tx = torch.floor(warpedCosTheta * self.num_theta_samples).long()
        ty = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        t_idx = ty * self.num_theta_samples + tx

        dots_sh = list(t_idx.shape[:-1])
        data = self.MTS_TRANS.view([1] * len(dots_sh) + [-1]).expand(dots_sh + [-1])

        t_idx = torch.clamp(t_idx, min=0, max=data.shape[-1] - 1).long()  # important
        T12 = torch.clamp(torch.gather(input=data, index=t_idx, dim=-1), min=0.0, max=1.0)
        T21 = T12  # colocated setting

        ## compute Fdr: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L249
        t_idx = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        data = self.MTS_DIFF_TRANS.view([1] * len(dots_sh) + [-1]).expand(dots_sh + [-1])
        t_idx = torch.clamp(t_idx, min=0, max=data.shape[-1] - 1).long()  # important
        Fdr = torch.clamp(1.0 - torch.gather(input=data, index=t_idx, dim=-1), min=0.0, max=1.0)  # [..., 1]

        diffuse_rgb = light_intensity * (diffuse_albedo / (1.0 - Fdr + 1e-10) / np.pi) * cos_theta * T12 * T21 * m_invEta2
        return diffuse_rgb

    def sheen_evaluation(self):
        pass

    def flatness_evaluation(self):
        pass

    def select(self, cond, a, b):
        v = b.clone()
        v[cond] = a[cond]
        return v

    def principled_fresnel(self, F_dielectric, metallic, spec_tint,
                           base_color, intensity,
                           cos_theta, eta,
                           has_metallic=True, has_spec_tint=True):
        ## https://github.com/mitsuba-renderer/mitsuba3/blob/152352f87b5baea985511b2a80d9f91c3c945a90/src/bsdfs/principledhelpers.h#L239
        lum = intensity * torch.ones_like(cos_theta)
        outside_mask = cos_theta > 0.0
        #inside_mask = cos_theta_i < 0.0
        eta = torch.ones_like(cos_theta) * eta
        rcp_eta = 1.0 / eta
        eta_it = self.select(outside_mask, eta, rcp_eta)
        #eta_it = eta.copy()
        #eta_it[inside_mask] = rcp_eta[inside_mask]
        F_schlick = torch.zeros_like(base_color)

        if has_metallic:
            if True:
                schlick_val = self.calc_schlick(base_color, cos_theta, eta)
                F_schlick += schlick_val * metallic.expand(-1, schlick_val.shape[-1])

        if has_spec_tint:
            c_tint = torch.ones_like(base_color)
            masklum = lum.squeeze() > 0
            c_tint[masklum, :] = base_color[masklum, :] / lum[masklum, :]
            #c_tint = self.select(lum > 0, base_color / lum, 1.0)
            F0_spec_tint = c_tint * self.schlick_R0_eta(eta_it)
            F_schlick += (1.0 - metallic) * spec_tint * self.calc_schlick(F0_spec_tint, cos_theta, eta)

        F_front = (1.0 - metallic) * (1.0 - spec_tint) * F_dielectric + F_schlick
        #return self.select(front_side, F_front, bsdf * F_dielectric)
        return F_front

    def calc_schlick(self, R0, cos_theta, eta):
        ### https://github.com/mitsuba-renderer/mitsuba3/blob/152352f87b5baea985511b2a80d9f91c3c945a90/src/bsdfs/principledhelpers.h#L156
        outside_mask = cos_theta.squeeze() > 0
        eta = torch.ones_like(cos_theta) * eta
        eta = torch.clamp(eta, min=1e-4, max=0.99999)
        rcp_eta = 1.0 / eta
        rcp_eta = torch.clamp(rcp_eta, min=1e-4, max=0.99999)
        eta_it = self.select(outside_mask, eta, rcp_eta)
        eta_ti = self.select(outside_mask, rcp_eta, eta)

        cos_theta_t_sqr = 1.0-(1.0 - cos_theta*cos_theta)*eta_ti*eta_ti
        cos_theta_t = torch.sqrt(cos_theta_t_sqr)
        val = self.schlick_weight(torch.abs(cos_theta)) * (1-R0) + R0
        val_neq1 = self.schlick_weight(cos_theta_t) * (1-R0) + R0
        val[eta_it.squeeze() < 1.0, :] = val_neq1[eta_it.squeeze() < 1.0, :]
        return val

    def schlick_weight(self, cos_theta):
        ### https://github.com/mitsuba-renderer/mitsuba3/blob/152352f87b5baea985511b2a80d9f91c3c945a90/src/bsdfs/principledhelpers.h#L141
        m = torch.clamp(1.0-cos_theta, 0.0, 1.0)
        return m**5

    def schlick_R0_eta(self, v):
        return ((v - 1.0) / (v + 1.0))**2

    def calc_F_Clearcoat(self, cos_theta, eta, viewdir=None, normal=None):
        #cos_theta_i = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        Fcc = self.calc_schlick(0.04, cos_theta, eta)
        return Fcc

    def calc_D_Clearcoat(self, cos_theta, clearcoat, viewdir=None, normal=None):
        #dot = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot = torch.clamp(cos_theta, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
        cosTheta2 = dot * dot
        v = (1.0-clearcoat) * 0.1 + clearcoat * 0.001
        root = cosTheta2 + (1.0 - cosTheta2) / (v * v + 1e-10)
        Dcc = 1.0 / (np.pi * v * v * root * root + 1e-10)
        return Dcc

    def calc_G_Clearcoat(self, cos_theta, alpha_u, alpha_v):
        #cos_theta_i = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        Gcc = smithG1(cos_theta, alpha_u) * smithG1(cos_theta, alpha_v)
        return Gcc

    def calc_D_specular(self, cos_theta, alpha):
        cosTheta2 = cos_theta * cos_theta
        root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)
        D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
        return D

    def calc_G_specular(self, cos_theta_i, alpha_u, alpha_v):
        ## compute shadowing term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L520
        G = smithG1(cos_theta_i, alpha_u) * smithG1(cos_theta_i, alpha_v)  # [..., 1]
        return G

    def forward(self, light, distance, normal, viewdir, params={}, switch_dict={}):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        params.keys() = ['anisotropic', 'roughness', 'metallic']
        """
        specular_roughness = torch.clamp(params['specular_roughness'], min=0.00001)
        dielectric_eta = torch.clamp(params['dielectric_eta'], min=1.000001, max=1.999999)
        metallic_eta = torch.clamp(params['metallic_eta'], min=0.099999, max=4.999999)
        metallic_k = torch.clamp(params['metallic_k'], min=0.099999, max=9.999999)
        specular_albedo = torch.clamp(params['specular_albedo'], min=0.00001)
        diffuse_albedo = torch.clamp(params['diffuse_albedo'], min=0.00001)

        #anisotropic = torch.clamp(params['anisotropic'], min=0.00001)
        #flatness = torch.clamp(params['flatness'], min=0.00001)
        #spec_trans = torch.clamp(params['spec_trans'], min=0.00001)
        #metallic = torch.clamp(params['metallic'], min=0.000001, max=0.999999)
        #dielectric = torch.clamp(params['dielectric'], min=0.000001, max=0.999999)
        #clearcoat = torch.clamp(params['clearcoat'], min=0.00001)
        #eta = torch.clamp(params['eta'], min=0.000001)
        eta = 1.48958738
        #spec_tint = torch.clamp(params['spec_tint'], min=0.000001)
        #sheen = torch.clamp(params['sheen'], min=0.00001)

        #brdf = (1.0 - metallic)
        #bsdf = (1.0 - metallic) * spec_trans
        #cos_theta_i = torch.dot(viewdir, normal)
        cos_theta_i = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        cos_theta_i = torch.clamp(cos_theta_i, min=0.00001, max=0.99999)
        cos_theta_o = cos_theta_i  # co-located setting

        #m_eta = 1.45
        #m_inveta = 1/m_eta
        inveta = 1.0 / eta
        #alpha_u, alpha_v = calc_dist_params(anisotropic=anisotropic, roughness=roughness, has_anisotropic=True)
        #alpha = 0.5 * (alpha_u + alpha_v)
        #F_die = fresnel_dielectric(cosThetaI=cos_theta_i, cosThetaT=cos_theta_i, eta=eta)
        alpha_u, alpha_v = specular_roughness, specular_roughness
        D = self.calc_D_specular(cos_theta_i, eta)
        G = self.calc_G_specular(cos_theta_i, alpha_u, alpha_v)

        #cosTheta2 = cos_theta_i * cos_theta_i
        #root = cosTheta2 + (1.0 - cosTheta2) / (alpha_u * alpha_v + 1e-10)
        #D = 1.0 / (np.pi * alpha_u * alpha_v * root * root + 1e-10)
        ## compute fresnel: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/libcore/util.cpp#L651
        # F = 0.04
        #F = 0.03867


        # decay light according to squared-distance falloff
        light_intensity = light / (distance * distance + 1e-10)
        #specular_rgb = light_intensity * specular_albedo * F * D * G / (4.0 * cos_theta_i + 1e-10)

        # if False:
        #     main_specular_rgb = self.main_specular_reflection(D=D,
        #                                                       G=G,
        #                                                       F_dielectric=F_die,
        #                                                       metallic=metallic,
        #                                                       spec_tint=spec_tint,
        #                                                       cos_theta=cos_theta_i,
        #                                                       color=specular_albedo,
        #                                                       intensity=light_intensity,
        #                                                       eta=eta)
        if True:
            eta_cu = 0.28
            k_cu = 5.4856
            eta = 1.5
            main_metallic_rgb = self.main_metallic_reflection(cos_theta_i, metallic_eta, metallic_k, specular_albedo)
            #main_metallic_rgb = self.main_metallic_reflection(cos_theta_i, eta_cu, k_cu, specular_albedo)
            main_dielectric_rgb = self.main_dielectric_reflection(D, G, cos_theta_i, dielectric_eta, specular_albedo)
            main_metallic_rgb *= light_intensity
            main_dielectric_rgb *= light_intensity
            #main_specular_rgb = metallic * main_metallic_rgb + dielectric * main_dielectric_rgb
            #main_specular_rgb = main_metallic_rgb + main_dielectric_rgb
            #main_specular_rgb = main_dielectric_rgb
            main_specular_rgb = main_metallic_rgb
        else:
            main_specular_rgb = torch.zeros_like(specular_albedo)

        #if False:
        #    clearcoat_specular_rgb = self.secondary_isotropic_specular_reflection(cos_theta_i, clearcoat, eta)
        #else:
        #    clearcoat_specular_rgb = torch.zeros_like(specular_albedo)

        #diffuse_rgb = self.diffuse_reflection(cos_theta=cos_theta_i, alpha=1.0-roughness, diffuse_albedo=diffuse_albedo)
        diffuse_rgb = self.diffuse_reflection_ggx(light_intensity=light_intensity,
                                                  cos_theta=cos_theta_i,
                                                  alpha=specular_roughness,
                                                  diffuse_albedo=diffuse_albedo)

        rgb = diffuse_rgb
        #rgb += specular_rgb
        #main_metallic_rgb = light_intensity * main_metallic_rgb
        #main_dielectric_rgb = light_intensity * main_dielectric_rgb
        #rgb += main_metallic_rgb
        rgb += main_specular_rgb

        ret = {"diffuse_rgb": diffuse_rgb,
               "specular_rgb": main_specular_rgb,
               "metallic_rgb": main_metallic_rgb,
               "dielectric_rgb": main_dielectric_rgb,
               "rgb": rgb}
        return ret


if __name__ == '__main__':
    eta = 1.48

    viewdir = np.random.randn(1, 3)
    normal = np.random.randn(1, 3)
    viewdir = viewdir / np.linalg.norm(viewdir)
    normal = normal / np.linalg.norm(normal)
    viewdir = torch.from_numpy(viewdir)
    normal = torch.from_numpy(normal)

    #cosTheta = np.sum(viewdir * normal, axis=-1, keepdims=True)
    cosTheta = torch.sum(viewdir * normal, dim=-1, keepdims=True)
    #print(cosTheta)
    F = fresnel_dielectric(cosThetaI=cosTheta, cosThetaT=cosTheta, eta=1.5046)
    print(F, cosTheta, F/cosTheta)

    # for Au
    # Cu: eta=855.063293 0.280000 k=855.063293 5.485625
    # Au: eta=855.063293 0.198125 k=855.063293 5.631250
    # Al: eta=855.063293 2.580000 k=855.063293 8.210000
    F = fresnel_conductor_exact(cosThetaI=cosTheta, eta=torch.tensor(2.580000), k=torch.tensor(8.210000))
    print(F, cosTheta, F/cosTheta)


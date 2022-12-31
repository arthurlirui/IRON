import os

import numpy as np
import torch


def gamma_correction(image, gamma=2.2):
    return torch.pow(image + 1e-6, 1.0 / gamma)


def inv_gamma_correction(image, gamma=2.2):
    return torch.pow(image, gamma)


def concatenate_result(image_list=[], imarray_length=3):
    rows = []
    all = []
    for i, img in enumerate(image_list):
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))

        rows.append(img)
        if len(rows) == imarray_length:
            tmp = np.concatenate(rows, axis=1)
            all.append(tmp)
            rows = []

    img_all = np.concatenate(all, axis=0)
    while 0 < len(rows) < imarray_length:
        rows.append(np.zeros_like(rows[0]))
    if len(rows) > 0:
        img_tail = np.concatenate(rows, axis=1)
        img_all = np.concatenate([img_all, img_tail], axis=0)
    return img_all



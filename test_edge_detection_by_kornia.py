import os
import kornia
import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import kornia as K


def imshow(input: torch.Tensor):
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    rootpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec'
    filename = 'WIN_20230225_16_39_30_Pro.jpg'
    img_bgr: np.ndarray = cv2.imread(os.path.join(rootpath, filename), cv2.IMREAD_COLOR)
    x_bgr: torch.Tensor = K.utils.image_to_tensor(img_bgr)  # CxHxWx
    x_bgr = x_bgr[None, ...].float() / 255.
    #x_bgr = x_bgr[None, ...].float()

    x_rgb: torch.Tensor = K.color.bgr_to_rgb(x_bgr)
    x_gray = K.color.rgb_to_grayscale(x_rgb)
    imshow(x_bgr)

    # grads: torch.Tensor = K.filters.spatial_gradient(x_gray, order=1)  # BxCx2xHxW
    # grads_x = grads[:, :, 0]
    # grads_y = grads[:, :, 1]
    # imshow(1. - grads_x.clamp(0., 1.))
    #
    # x_sobel: torch.Tensor = K.filters.sobel(x_gray)
    # imshow(x_sobel)
    #
    # x_laplacian: torch.Tensor = K.filters.canny(x_gray)[0]
    # imshow(1. - x_laplacian.clamp(0., 1.))

    from kornia.contrib import EdgeDetector

    #img = torch.rand(1, 3, 320, 320)
    img = x_bgr
    detect = EdgeDetector()
    out = detect(img)
    print(out.shape)
    print(torch.min(out[:]), torch.max(out[:]), torch.median(out[:]))
    #out = (out - torch.median(out[:])) / (2*torch.median(out[:]) - torch.min(out[:]))
    #out = torch.log(-1*out)
    out = -1*out
    out = (out - torch.min(out[:])) / (torch.max(out[:]) - torch.min(out[:]))
    imshow(out.squeeze())
    import matplotlib.pyplot as plt


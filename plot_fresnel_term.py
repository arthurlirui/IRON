import os
from models.renderer_ggx import CompositeRenderer
import numpy as np
import torch


if __name__ == '__main__':
    iorpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/mitsuba-data/ior'
    iorname = 'Al'
    kname = f'{iorname}.k.spd'
    etaname = f'{iorname}.eta.spd'
    eta = np.loadtxt(os.path.join(iorpath, etaname))
    k = np.loadtxt(os.path.join(iorpath, kname))
    #print(eta)
    #print(k)
    eta = torch.tensor(eta)
    k = torch.tensor(k)
    costheta = torch.linspace(0, 1, 200)
    fresnelterm = {}
    for i in range(eta.shape[0]):
        wavelength = eta[i][0]
        etav = eta[i][1]
        kv = k[i][1]
        f = []
        for v in costheta:
            fv = CompositeRenderer.fresnel_conductor_exact(v, etav, kv)
            f.append(fv)
        fresnelterm[i] = f
    print(fresnelterm)

    import matplotlib.pyplot as plt
    indexs = [54, 50, 45, 40, 35, 30, 25, 20]
    #for i in indexs:
    #    plt.plot(costheta.numpy(), fresnelterm[i], color='red', linewidth=2, label=r'$\lambda$=%.2f' % eta[i][0])
    l1 = plt.plot(costheta.numpy(), fresnelterm[54], color='red', linewidth=2, label=r'$\lambda$=%.2f' % eta[54][0])
    l2 = plt.plot(costheta.numpy(), fresnelterm[45], color='green', linewidth=2, label=r'$\lambda$=%.2f' % eta[45][0])
    l3 = plt.plot(costheta.numpy(), fresnelterm[35], color='blue', linewidth=2, label=r'$\lambda$=%.2f' % eta[35][0])
    l4 = plt.plot(costheta.numpy(), fresnelterm[25], color='cyan', linewidth=2, label=r'$\lambda$=%.2f' % eta[25][0])
    plt.legend(loc='best')
    plt.show()
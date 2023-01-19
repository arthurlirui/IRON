import os
from models.renderer_ggx import CompositeRenderer
import numpy as np
import torch
from matplotlib.pyplot import cm



def plot_fresnel_term_for_wavelength():
    iorpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/mitsuba-data/ior'
    #materials = ['Au', 'Cu', 'Al', 'Ag']
    materials = ['Ag']
    color = iter(cm.jet(np.linspace(0, 1, 60)))
    for m in materials:
        iorname = m
        kname = f'{iorname}.k.spd'
        etaname = f'{iorname}.eta.spd'
        eta = np.loadtxt(os.path.join(iorpath, etaname))
        k = np.loadtxt(os.path.join(iorpath, kname))
        # print(eta)
        # print(k)
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
        #print(fresnelterm)

        import matplotlib.pyplot as plt
        #indexs = [54, 50, 45, 40, 35, 30, 25, 20]
        for i, ft in enumerate(fresnelterm):
            #labeli = m + '-' + r'$\lambda$=%.2f $\eta=%.2f$ k=%.2f' % (eta[i][0], eta[i][1], k[i][1])
            labeli = m + '-' + r'%.2fnm (%.2f %.2f)' % (eta[i][0], eta[i][1], k[i][1])
            plt.plot(costheta.numpy(), fresnelterm[i], c=next(color), linewidth=1, label=labeli)

        # for i in indexs:
        #    plt.plot(costheta.numpy(), fresnelterm[i], color='red', linewidth=2, label=r'$\lambda$=%.2f' % eta[i][0])
        #l1 = plt.plot(costheta.numpy(), fresnelterm[54], color=iter(color), linewidth=2, label=r'$\lambda$=%.2f' % eta[54][0])
        #l2 = plt.plot(costheta.numpy(), fresnelterm[45], color='green', linewidth=2, label=r'$\lambda$=%.2f' % eta[45][0])
        #l3 = plt.plot(costheta.numpy(), fresnelterm[35], color='blue', linewidth=2, label=r'$\lambda$=%.2f' % eta[35][0])
        #l4 = plt.plot(costheta.numpy(), fresnelterm[25], color='cyan', linewidth=2, label=r'$\lambda$=%.2f' % eta[25][0])
    #plt.legend(loc='center right')
    plt.title(m + r' - $\lambda$=%.2f$\sim$%.2f nm $\eta$=%.2f$\sim$%.2f $k$=%.2f$\sim$%.2f' % (eta[0][0], eta[-1][0], eta[0][1], eta[-1][1], k[0][1], k[-1][1]))
    plt.colorbar()
    plt.show()

def plot_fresnel_term_for_dielectric():
    iorpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/mitsuba-data/ior'
    #materials = ['Au', 'Cu', 'Al', 'Ag']
    materials = ['Water', 'Plastic', 'Glass', 'Diamond']
    eta_list = [1.3330, 1.49, 1.49, 2.419]
    color = iter(['red', 'green', 'blue', 'cyan'])
    fresnelterm = []
    for i, m in enumerate(materials):
        costheta = torch.linspace(0, 1, 200)

        eta = eta_list[i]

        f = []
        for v in costheta:
            fv = CompositeRenderer.fresnel_dielectric(v, eta)
            f.append(fv)
        fresnelterm.append(f)
    import matplotlib.pyplot as plt
    #indexs = [54, 50, 45, 40, 35, 30, 25, 20]
    for i, v in enumerate(fresnelterm):
        #labeli = m + '-' + r'$\lambda$=%.2f $\eta=%.2f$ k=%.2f' % (eta[i][0], eta[i][1], k[i][1])
        labeli = materials[i] + '-' + r'%.2f' % eta_list[i]
        plt.plot(costheta.numpy(), v, c=next(color), linewidth=1, label=labeli)

        # for i in indexs:
        #    plt.plot(costheta.numpy(), fresnelterm[i], color='red', linewidth=2, label=r'$\lambda$=%.2f' % eta[i][0])
        #l1 = plt.plot(costheta.numpy(), fresnelterm[54], color=iter(color), linewidth=2, label=r'$\lambda$=%.2f' % eta[54][0])
        #l2 = plt.plot(costheta.numpy(), fresnelterm[45], color='green', linewidth=2, label=r'$\lambda$=%.2f' % eta[45][0])
        #l3 = plt.plot(costheta.numpy(), fresnelterm[35], color='blue', linewidth=2, label=r'$\lambda$=%.2f' % eta[35][0])
        #l4 = plt.plot(costheta.numpy(), fresnelterm[25], color='cyan', linewidth=2, label=r'$\lambda$=%.2f' % eta[25][0])
    plt.legend(loc='center right')
    plt.title()
    plt.show()

def plot_fresnel_term_for_materials():
    iorpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/mitsuba-data/ior'
    iorname = 'Al'
    kname = f'{iorname}.k.spd'
    etaname = f'{iorname}.eta.spd'
    eta = np.loadtxt(os.path.join(iorpath, etaname))
    k = np.loadtxt(os.path.join(iorpath, kname))
    # print(eta)
    # print(k)
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
    # for i in indexs:
    #    plt.plot(costheta.numpy(), fresnelterm[i], color='red', linewidth=2, label=r'$\lambda$=%.2f' % eta[i][0])
    l1 = plt.plot(costheta.numpy(), fresnelterm[54], color='red', linewidth=2, label=r'$\lambda$=%.2f' % eta[54][0])
    l2 = plt.plot(costheta.numpy(), fresnelterm[45], color='green', linewidth=2, label=r'$\lambda$=%.2f' % eta[45][0])
    l3 = plt.plot(costheta.numpy(), fresnelterm[35], color='blue', linewidth=2, label=r'$\lambda$=%.2f' % eta[35][0])
    l4 = plt.plot(costheta.numpy(), fresnelterm[25], color='cyan', linewidth=2, label=r'$\lambda$=%.2f' % eta[25][0])
    plt.legend(loc='best')
    plt.show()

def show_colormap():
    import numpy as np
    import matplotlib.pyplot as plt

    cmaps = [('Perceptually Uniform Sequential', [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
             ('Sequential', [
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('Sequential (2)', [
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                 'hot', 'afmhot', 'gist_heat', 'copper']),
             ('Diverging', [
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
             ('Qualitative', [
                 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                 'Dark2', 'Set1', 'Set2', 'Set3',
                 'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('Miscellaneous', [
                 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
                 'gist_ncar'])]

    gradient = np.linspace(0, 1, 60)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - .35 / figh, bottom=.15 / figh, left=0.2, right=0.99)

        #axs.set_title(cmap_category + 'colormaps', fontsize=14)

        #for ax, cmap_name in zip(axs, cmap_list):
        for cmap_name in cmap_list:
            axs.imshow(gradient, aspect='auto', cmap=cmap_name)
            #axs.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10, transform=axs.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        #for ax in axs:
        axs.set_axis_off()

    #for cmap_category, cmap_list in cmaps:
    #    plot_color_gradients(cmap_category, cmap_list)
    #from matplotlib import
    #plot_color_gradients('Miscellaneous',
    #                     ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    #                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    #                      'turbo', 'nipy_spectral', 'gist_ncar'])
    plot_color_gradients('Miscellaneous', ['jet'])

    plt.show()

if __name__ == '__main__':
    #plot_fresnel_term_for_wavelength()
    #plot_fresnel_term_for_dielectric()
    show_colormap()

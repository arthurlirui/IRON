import glob
import os
import imageio
import numpy as np

if __name__ == '__main__':
    if False:
        # process manually segment images set
        rootpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir_all/blend01/images'
        flist = os.listdir(rootpath)
        for f in flist:
            print(f)
            img = imageio.imread(os.path.join(rootpath, f))
            img3c = img[:, :, :3]
            img3c[:, :, 0] = img3c[:, :, 0] * img[:, :, 3]
            img3c[:, :, 1] = img3c[:, :, 1] * img[:, :, 3]
            img3c[:, :, 2] = img3c[:, :, 2] * img[:, :, 3]
            imageio.imwrite(os.path.join(rootpath, f), img3c)
    if False:
        # process blendedmvs
        dataname = '5a2a95f032a1c655cfe3de62'
        rootpath = '/home/lir0b/data/BlendedMVS++/'
        maskpath = os.path.join(rootpath, dataname, 'blended_images')
        imgpath = os.path.join(rootpath, dataname, 'images')
        masklist = sorted(glob.glob(os.path.join(maskpath, '*_masked.jpg')))
        for m in masklist:
            mask = imageio.imread(m)
            maskt = np.zeros_like(mask[:, :, 0])
            maskt[mask[:, :, 0] > 10] = 255
            mname = os.path.basename(m).split('_')[0]+'.jpg'
            imageio.imwrite(os.path.join(rootpath, dataname, 'masks', mname), maskt)

    if True:
        # process manually segment images set
        rootpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir_all/perfume2/'
        flist = sorted(os.listdir(os.path.join(rootpath, 'images')))

        for f in flist:
            print(f)
            img = imageio.imread(os.path.join(rootpath, 'images', f))
            mask = imageio.imread(os.path.join(rootpath, 'masks', f))
            #mask[mask > 0] = 1
            #img[:, :, 0] = img[:, :, 0] * mask
            #img[:, :, 1] = img[:, :, 1] * mask
            #img[:, :, 2] = img[:, :, 2] * mask
            img[mask == 0, 0] = 0
            img[mask == 0, 1] = 0
            img[mask == 0, 2] = 0
            img4c = np.concatenate([img, mask[:, :, np.newaxis]], axis=-1)
            imageio.imwrite(os.path.join(rootpath, 'images', f.split('.')[0]+'.png'), img4c)

    if False:
        # process manually segment images set
        rootpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir_all/banana_flash/'
        flist = sorted(os.listdir(os.path.join(rootpath, 'images_ori')))

        for i, f in enumerate(flist):
            print(f)
            fname = f.split('.')[0]
            img = imageio.imread(os.path.join(rootpath, 'images_ori', f))
            #mask = imageio.imread(os.path.join(rootpath, 'masks', f'%04d'%i + '.png'))
            #img4c = np.concatenate([img, mask[:, :, np.newaxis]], axis=-1)
            imageio.imwrite(os.path.join(rootpath, 'images', f'%04d'%i + '.png'), img)
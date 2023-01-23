import os


if __name__ == '__main__':
    rootpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir_all/dragon_all/train/rgb/images'
    flist = os.listdir(rootpath)
    for i in range(0, 200):
        if not os.path.exists(os.path.join(rootpath, f'{i}.exr')):
            print(f'{i}.exr')
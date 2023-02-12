import os
import models.dataset


if __name__ == '__main__':
    if False:
        rootpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_flashlight/bagel/train/image'
        filename = '0.exr'
        from models.dataset import exr_reader
        exrreader = exr_reader(reader_name='pyexr')
        imgexr = exrreader(os.path.join(rootpath, filename))
        print(imgexr.shape)
        print(imgexr[250, 150, :])
        import cv2
        cv2.imshow('pyexr', imgexr)
        cv2.waitKey()

    if True:
        import numpy as np
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'ggx', 'ext_mts_rtrans_data.txt')
        MTS_TRANS = np.loadtxt(filepath).astype(np.float32)
        # 5000 entries, external IOR
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'ggx', 'int_mts_diff_rtrans_data.txt')
        MTS_DIFF_TRANS = np.loadtxt(filepath).astype(np.float32)
        # 50 entries, internal IOR

        print(np.sum(MTS_TRANS.reshape(100, -1), axis=0))

        dot_sh = [128, 32]



import os
import models.dataset


if __name__ == '__main__':
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


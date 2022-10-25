import os
import imageio

def process_image():
    imgpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_flashlight_real/buddha/'
    trainpath = os.path.join(imgpath, 'train')
    testpath = os.path.join(imgpath, 'test')
    filelist = os.listdir(os.path.join(trainpath, 'image_bak'))
    for file in filelist:
        img = imageio.v3.imread(os.path.join(trainpath, 'image_bak', file))
        new_file = file.split('.')[0]
        print(new_file)
        imageio.v3.imwrite(os.path.join(trainpath, 'image', new_file + '.png'), img)

    filelist = os.listdir(os.path.join(testpath, 'image_bak'))
    for file in filelist:
        img = imageio.v3.imread(os.path.join(testpath, 'image_bak', file))
        new_file = file.split('.')[0]
        imageio.v3.imwrite(os.path.join(testpath, 'image', new_file + '.png'), img)


def process_json():
    json_path = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_flashlight_real/buddha/test'
    import json
    jsons = json.load(open(os.path.join(json_path, 'cam_dict_norm.json')))
    new_filename = 'cam_dict_norm_new.json'
    print(jsons)
    new_jsons = {}
    print(jsons.keys())
    for key in jsons:
        new_key = key.split('.')[0]+'.png'
        new_jsons[new_key] = jsons[key]
    json.dumps(new_jsons, open(os.path.join(json_path, new_filename), 'w'))


if __name__ == '__main__':
    process_json()

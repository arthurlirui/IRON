import open3d as o3d
import json
import numpy as np
from pprint import pprint


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)
            img_size = camera_dict[img_name]['img_size']
            frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)


if __name__ == '__main__':
    import os

    #base_dir = './'
    sphere_radius = 1.0
    if False:
        dataname = 'corn'
        base_dir = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir/{dataname}'
        #train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict.json')))
        train_cam_norm_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
        #colored_camera_dicts = [([0, 1, 0], train_cam_dict), ([1, 0, 0], train_cam_norm_dict)]
        colored_camera_dicts = [([0, 1, 0], train_cam_norm_dict)]

    if False:
        dataname = 'pepper_flash2'
        base_dir = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir/{dataname}'
        #train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict.json')))
        train_cam_norm_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict.json')))
        #colored_camera_dicts = [([0, 1, 0], train_cam_dict), ([1, 0, 0], train_cam_norm_dict)]
        colored_camera_dicts = [([1, 0, 0], train_cam_norm_dict)]

    if False:
        dataname = 'xmen'
        base_dir = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_flashlight_real/{dataname}'
        #train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict.json')))
        train_cam_norm_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
        colored_camera_dicts = [([1, 0, 0], train_cam_norm_dict)]

    if False:
        dataname = 'custard_apple4'
        base_dir = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir/{dataname}'
        train_cam_dict = json.load(open(os.path.join(base_dir, 'cam_dict.json')))
        train_cam_norm_dict = json.load(open(os.path.join(base_dir, 'cam_dict_norm.json')))
        #colored_camera_dicts = [([0, 1, 0], train_cam_dict), ([1, 0, 0], train_cam_norm_dict)]
        colored_camera_dicts = [([1, 0, 0], train_cam_norm_dict)]
        print(len(train_cam_norm_dict))
        #colored_camera_dicts = [([1, 0, 0], train_cam_norm_dict)]

    if True:
        dataname = 'cloth4'
        base_dir = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir_select/{dataname}/train'
        #base_dir = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/data_nir/{dataname}/train'
        #train_cam_dict = json.load(open(os.path.join(base_dir, 'cam_dict.json')))
        #train_cam_norm_dict = json.load(open(os.path.join(base_dir, 'cam_dict_norm.json')))

        #dir0 = os.path.join(base_dir, 'train', 'rgb')
        #dir1 = os.path.join(base_dir, 'train', 'nir')
        #train_cam_0 = json.load(open(os.path.join(dir0, 'cam_dict_norm.json')))
        #train_cam_1 = json.load(open(os.path.join(dir1, 'cam_dict_norm.json')))

        #train_cam_0 = json.load(open(os.path.join(base_dir, 'cam_dict.json')))
        train_cam_1 = json.load(open(os.path.join(base_dir, 'cam_dict_norm.json')))

        # colored_camera_dicts = [([0, 1, 0], train_cam_dict), ([1, 0, 0], train_cam_norm_dict)]
        #colored_camera_dicts = [([1, 0, 0], train_cam_norm_dict), ([0, 1, 0], train_cam_dict)]
        #colored_camera_dicts = [([1, 0, 0], train_cam_0), ([0, 1, 0], train_cam_1)]
        colored_camera_dicts = [([1, 0, 0], train_cam_1)]
        #pprint(train_cam_dict)
        #print(len(train_cam_norm_dict))

    if False:
        dataname1 = 'custard_apple3_nir'
        dataname2 = 'custard_apple3_env'
        base_dir1 = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir/{dataname1}'
        base_dir2 = f'/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir/{dataname2}'
        train_cam_norm_dict1 = json.load(open(os.path.join(base_dir1, 'cam_dict.json')))
        train_cam_norm_dict2 = json.load(open(os.path.join(base_dir2, 'cam_dict.json')))
        colored_camera_dicts = [([1, 0, 0], train_cam_norm_dict1), ([0, 1, 0], train_cam_norm_dict2)]
        #print(len(train_cam_norm_dict))

    #train_cam_dict = json.load(open(os.path.join(base_dir, 'images/work_video_flash/cam_dict_norm.json')))
    #train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict.json')))
    #train_cam_norm_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
    #test_cam_dict = json.load(open(os.path.join(base_dir, 'images/work_video_flash/cam_dict_norm.json')))
    #path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
    #path_cam_dict = json.load(open(os.path.join(base_dir, 'images/work_video_flash/cam_dict_norm.json')))
    camera_size = 0.1
    # colored_camera_dicts = [([0, 1, 0], train_cam_dict),
    #                         ([0, 0, 1], test_cam_dict),
    #                         ([1, 1, 0], path_cam_dict)]
    #colored_camera_dicts = [([0, 1, 0], train_cam_dict), ([1, 0, 0], train_cam_norm_dict)]
    #colored_camera_dicts = [([1, 0, 0], train_cam_norm_dict)]
    #colored_camera_dicts = [([1, 0, 0], train_cam_dict)]

    #geometry_file = os.path.join(base_dir, 'work', 'mvs', 'mesh_norm.ply')
    #geometry_file = os.path.join(base_dir, 'work', 'mvs', 'meshed_trim_3.ply')
    geometry_file = os.path.join('/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_nir_all/deer3/mvs/meshed_trim_3.ply')
    #geometry_file = os.path.join(base_dir, 'meshed_trim_3.ply')
    #geometry_type = 'mesh'
    geometry_type = 'mesh'
    geometry_file = None
    geometry_type = None

    visualize_cameras(colored_camera_dicts,
                      sphere_radius,
                      camera_size=camera_size,
                      geometry_file=geometry_file,
                      geometry_type=geometry_type)

import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scene_hs.colmap_loader import Camera, Image

def read_extrinsics_custom_json(path):
    
    with open(path, 'r') as cameras_file:
        cameras_data = json.load(cameras_file)

    images = {}
    for key, val in cameras_data.items():

        image_id = int(key)
        w2c = np.array(val['w2c'])
        
        rotation_matrix = R.from_matrix(w2c[0:3, 0:3])
        qvec_xyzw = rotation_matrix.as_quat()
        qvec_wxyz = np.array(
            [qvec_xyzw[3], qvec_xyzw[0], qvec_xyzw[1], qvec_xyzw[2]])
        
        tvec = w2c[0:3, -1]
        
        images[image_id] = Image(
            id=image_id, qvec=qvec_wxyz, tvec=tvec,
            camera_id=image_id, name=val['name'],
            xys=[], point3D_ids=[])
        
    return images


def read_intrinsics_custom_json(path):
    
    with open(path, 'r') as cameras_file:
        cameras_data = json.load(cameras_file)

    cameras = {}
    for key, val in cameras_data.items():

        image_id = int(key)
        camera_id = image_id
        width = int(val['width'])
        height = int(val['height'])
        model = 'PINHOLE'
        params = [val['K'][0][0], val['K'][1][1], val['K'][0][2], val['K'][1][2]]
    
        cameras[camera_id] = Camera(
            id=camera_id, model=model,
            width=width, height=height,
            params=params)
        
    return cameras
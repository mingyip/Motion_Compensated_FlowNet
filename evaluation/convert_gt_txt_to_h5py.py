import numpy as np
import h5py

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    Xx = 1 - 2*qy*qy - 2*qz*qz
    Xy = 2*qx*qy + 2*qz*qw
    Xz = 2*qx*qz - 2*qy*qw
        
    Yx = 2*qx*qy - 2*qz*qw
    Yy = 1 - 2*qx*qx - 2*qz*qz
    Yz = 2*qy*qz + 2*qx*qw

    Zx = 2*qx*qz + 2*qy*qw
    Zy = 2*qy*qz - 2*qx*qw
    Zz = 1 - 2*qx*qx - 2*qy*qy
    
    rot = np.array([[Xx, Yx, Zx], [Xy, Yy, Zy], [Xz, Yz, Zz]])
    return rot


filelist = np.array(
    [
    # "boxes_6dof",
    # "boxes_rotation"],
    "boxes_translation",
    "calibration",
    "dynamic_6dof",
    "dynamic_rotation",
    "dynamic_translation",
    "hdr_boxes",
    "hdr_poster",
    "poster_6dof",
    "poster_rotation",
    "poster_translation",
    "shapes_6dof",
    "shapes_rotation",
    "shapes_translation",
    "simulation_3planes",
    "simulation_3walls",
    "slider_close",
    "slider_depth",
    "slider_far",
    "slider_hdr_close",
    "slider_hdr_far",
    ]
)


for idx, filename in enumerate(filelist):

    timestamp = []
    pose = []

    
    with open("/mnt/Data3/eth_data/" + filename + "/groundtruth.txt", "r") as f:
        for line in f:

            ts, px, py, pz, qx, qy, qz, qw = line.split()
            R = quaternion_to_rotation_matrix(float(qx), float(qy), float(qz), float(qw))
            p = np.eye(4)
            p[:3, :3] = R
            p[0, 3] = float(px)
            p[1, 3] = float(py)
            p[2, 3] = float(pz)

            pose.append(p)
            timestamp.append(float(ts))


    with h5py.File(f"/mnt/Data3/eth_data/{filename}/{filename}_gt.h5", 'w') as h5_file:
        # pose = h5_file.create_dataset("davis/left/pose", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)

        h5_file.create_dataset("davis/left/pose", data=np.array(pose), dtype=np.dtype(np.float64))
        h5_file.create_dataset("davis/left/pose_ts", data=np.array(timestamp), dtype=np.dtype(np.float64))


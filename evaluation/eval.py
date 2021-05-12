import io
import os
import pprint
from tqdm import trange
from tqdm import tqdm
import numpy as np
from datetime import datetime
from losses import *

from test_opengv import RelativePoseDataset

from scipy.linalg import logm
import torch
import h5py
import cv2
import pyopengv

from config import configs
from data_loader import EventData
from EVFlowNet import EVFlowNet
from dataset import DynamicH5Dataset
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

# from event_utils import binary_search_h5_gt_timestamp
from vis_utils import cvshow_all, cvshow_all_eval, warp_events_with_flow_torch, get_forward_backward_flow_torch
# from vis_utils import vis_events_and_flows


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def pixel_bearing_many(pixels):
    """Unit vectors pointing to the pixel viewing directions."""
    x = pixels[:, 0]
    y = pixels[:, 1]
    l = np.sqrt(x * x + y * y + 1.0)
    return np.column_stack((x / l, y / l, 1.0 / l))

def rotation_matrix_to_quaternion(rot):
    
    qw = np.sqrt(1 + rot[..., 0,0] + rot[..., 1,1] + rot[..., 2,2]) / 2
    qx = (rot[..., 2, 1] - rot[..., 1, 2]) / (4 * qw)
    qy = (rot[..., 0, 2] - rot[..., 2, 0]) / (4 * qw)
    qz = (rot[..., 1, 0] - rot[..., 0, 1]) / (4 * qw)

    return (qw, qx, qy, qz)

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    Xx = 1-2*np.multiply(qy, qy)-2*np.multiply(qz, qz)
    Xy = 2*np.multiply(qx, qy)+2*np.multiply(qz, qw)
    Xz = 2*np.multiply(qx, qz)-2*np.multiply(qy, qw)
    
    
    Yx = 2*np.multiply(qx, qy)-2*np.multiply(qz, qw)
    Yy = 1-2*np.multiply(qx, qx)-2*np.multiply(qz, qz)
    Yz = 2*np.multiply(qy, qz)+2*np.multiply(qx, qw)
    
    
    Zx = 2*np.multiply(qx, qz)+2*np.multiply(qy, qw)
    Zy = 2*np.multiply(qy, qz)-2*np.multiply(qx, qw)
    Zz = 1-2*np.multiply(qx, qx)-2*np.multiply(qy, qy)
    
    rot = np.array([[Xx, Yx, Zx], [Xy, Yy, Zy], [Xz, Yz, Zz]])
    if len(rot.shape) == 3:
        rot = rot.transpose(2,0,1)
    return rot

def absolute_to_relative_pose(T_wc):

    T_c = []

    last_frame = np.eye(4)
    for T_ in T_wc:
        T_c.append(np.linalg.inv(T_) @ last_frame)

        last_frame = T_

    return np.array(T_c)

def relative_to_absolute_pose(T_c):

    T_wc = []

    total_trans = np.eye(4)
    for T in T_c:
        
        total_trans = total_trans @ np.linalg.inv(T)
        T_wc.append(total_trans)

    return np.array(T_wc)

def inverse_se3_matrix(P):
    R = P[:3, :3]
    t = P[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv

def create_se3_matrix_with_R_t(R, t):
    P = np.eye(4)
    P[0:3, 0:3] = R
    P[0:3, 3]   = np.squeeze(t)
    return P

def visualize_trajectory(world_frame, image_path_name, show=False, rotate=None):

    def axisEqual3D(ax):
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    # Visualize path 
    x = world_frame[:, 0, 3]
    y = world_frame[:, 1, 3]
    # z = np.zeros_like(x)
    z = world_frame[:, 2, 3]
    idx = np.arange(len(x))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(x, y, z, 'gray')
    axisEqual3D(ax)
    ax.scatter3D(x, y, z, c=idx, cmap='hsv')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if rotate == 'x':
        ax.view_init(elev=0., azim=0)
    elif rotate == 'y':
        ax.view_init(elev=0., azim=90)
    elif rotate == 'z':
        ax.view_init(elev=90., azim=0)

    plt.savefig(image_path_name)
    if show:
        plt.show()
    plt.close()

def get_gt_pose_from_idx(gt_path, idx):
    with h5py.File(gt_path, "r") as gt_file:
        return gt_file['davis']['left']['pose'][idx]

def get_gt_timestamp_from_idx(gt_path, idx):
    with h5py.File(gt_path, "r") as gt_file:
        return gt_file['davis']['left']['pose_ts'][idx]

def binary_search_h5_gt_timestamp(gt_path, interp_ts):
    with h5py.File(gt_path, "r") as gt_file:

        # <KeysViewHDF5 ['blended_image_rect', 
        #               'blended_image_rect_ts', 
        #               'depth_image_raw', 
        #               'depth_image_raw_ts', 
        #               'depth_image_rect', 
        #               'depth_image_rect_ts', 
        #               'flow_dist', 
        #               'flow_dist_ts', 
        #               'odometry', 
        #               'odometry_ts', 
        #               'pose', 
        #               'pose_ts']>

        # print(gt_file["davis/left/pose"].dtype)
        # print(gt_file["davis/left/pose_ts"].dtype)
        # print(len(gt_file["davis/left/pose"]))
        # print(len(gt_file["davis/left/pose_ts"]))
        # raise

        # print(gt_file["davis/left/pose"][100])
        # print(gt_file["davis/left/pose_ts"][100])

        # print(list(gt_file["davis/left/pose_ts"]))

        # # Get the data
        # # data = list(gt_file[a_group_key])
        # raise


        return np.searchsorted(gt_file['davis']['left']['pose_ts'], interp_ts, side='right', sorter=None)

def get_interpolated_gt_pose(gt_path, interp_ts):

    pose_idx = binary_search_h5_gt_timestamp(gt_path, interp_ts)

    # Calculate interpolation ratio
    pt1_ts = get_gt_timestamp_from_idx(gt_path, pose_idx-1)
    pt2_ts = get_gt_timestamp_from_idx(gt_path, pose_idx)
    ratio = (pt2_ts - interp_ts) / (pt2_ts - pt1_ts)

    # print(pt1_ts, pt2_ts, interp_ts)
    # raise

    # Get 4x4 begin and end Pose 
    pt1_pose = get_gt_pose_from_idx(gt_path, pose_idx-1)
    pt2_pose = get_gt_pose_from_idx(gt_path, pose_idx)

    # Interpolate translation vector t
    trans = ratio * pt1_pose[0:3, 3] + (1-ratio) * pt2_pose[0:3, 3]

    # Interpolate rotation matrix R
    key_rot = Rotation.from_matrix([pt1_pose[0:3, 0:3], pt2_pose[0:3, 0:3]])
    slerp = Slerp([0, 1], key_rot)
    interp_rot = slerp([ratio]).as_matrix().squeeze()

    # Compose the interpolated matrix
    mat = np.eye(4)
    mat[0:3, 0:3] = interp_rot
    mat[0:3, 3] = trans

    return mat


# Params
outdoor1_params = {
    'sensor_size': (256, 336),
    'dataset_path': 'data/outdoor_day1_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/outdoor_day1/outdoor_day1_gt.hdf5',
    'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[223.9940010790056, 0, 170.7684322973841], [0, 223.61783486959376, 128.18711828338436], [0, 0, 1]])
}

outdoor2_params = {
    'sensor_size': (256, 336),
    'dataset_path': 'data/outdoor_day2_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/outdoor_day2/outdoor_day2_gt.hdf5',
    'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[223.9940010790056, 0, 170.7684322973841], [0, 223.61783486959376, 128.18711828338436], [0, 0, 1]])
}

hdr_boxes_params = {
    'sensor_size': (176, 240),
    'dataset_path': '/mnt/Data3/eth_data/hdr_boxes/event_cnn/hdr_boxes.h5',
    'gt_path': '/mnt/Data3/eth_data/hdr_boxes/hdr_boxes_gt.h5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[199.092366542, 0, 132.192071378], [0, 198.82882047, 110.712660011], [0, 0, 1]])
}

poster_translation_params = {
    'sensor_size': (176, 240),
    'dataset_path': '/mnt/Data3/eth_data/poster_translation/event_cnn/poster_translation.h5',
    'gt_path': '/mnt/Data3/eth_data/poster_translation/poster_translation_gt.h5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[199.092366542, 0, 132.192071378], [0, 198.82882047, 110.712660011], [0, 0, 1]])
}

indoor1_params = {
    'sensor_size': (256, 336),
    'dataset_path': '/mnt/Data3/mvsec/data/indoor_flying1/event_cnn/indoor_flying1_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/indoor_flying1/indoor_flying1_gt.hdf5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[226.38018519795807, 0, 173.6470807871759], [0, 226.15002947047415, 133.73271487507847], [0, 0, 1]])
}

indoor2_params = {
    'sensor_size': (256, 336),
    'dataset_path': '/mnt/Data3/mvsec/data/indoor_flying2/event_cnn/indoor_flying2_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/indoor_flying2/indoor_flying2_gt.hdf5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[226.38018519795807, 0, 173.6470807871759], [0, 226.15002947047415, 133.73271487507847], [0, 0, 1]])
}

indoor3_params = {
    'sensor_size': (256, 336),
    'dataset_path': '/mnt/Data3/mvsec/data/indoor_flying3/event_cnn/indoor_flying3_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/indoor_flying3/indoor_flying3_gt.hdf5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[226.38018519795807, 0, 173.6470807871759], [0, 226.15002947047415, 133.73271487507847], [0, 0, 1]])
}

indoor4_params = {
    'sensor_size': (256, 336),
    'dataset_path': '/mnt/Data3/mvsec/data/indoor_flying4/event_cnn/indoor_flying4_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/indoor_flying4/indoor_flying4_gt.hdf5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[226.38018519795807, 0, 173.6470807871759], [0, 226.15002947047415, 133.73271487507847], [0, 0, 1]])
}

outdoor_night1_params = {
    'sensor_size': (256, 336),
    'dataset_path': '/mnt/Data3/mvsec/data/outdoor_night1/event_cnn/outdoor_night1_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/outdoor_night1/outdoor_night1_gt.hdf5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[225.7028124771077, 0, 167.3925764568589], [0, 225.3928747331328, 126.81201949043754], [0, 0, 1]])
}

outdoor_night2_params = {
    'sensor_size': (256, 336),
    'dataset_path': '/mnt/Data3/mvsec/data/outdoor_night2/event_cnn/outdoor_night2_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/outdoor_night2/outdoor_night2_gt.hdf5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[225.7028124771077, 0, 167.3925764568589], [0, 225.3928747331328, 126.81201949043754], [0, 0, 1]])
}

outdoor_night3_params = {
    'sensor_size': (256, 336),
    'dataset_path': '/mnt/Data3/mvsec/data/outdoor_night3/event_cnn/outdoor_night3_data.h5',
    'gt_path': '/mnt/Data3/mvsec/data/outdoor_night3/outdoor_night3_gt.hdf5',
    # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
    'camera_intrinsic': np.array([[225.7028124771077, 0, 167.3925764568589], [0, 225.3928747331328, 126.81201949043754], [0, 0, 1]])
}

# in1_params = {
#     'sensor_size': (256, 336),
#     'dataset_path': '/mnt/Data3/mvsec/data/outdoor_night1/event_cnn/outdoor_night1_data.h5',
#     'gt_path': '/mnt/Data3/mvsec/data/outdoor_night1/outdoor_night1_gt.hdf5',
#     # 'dist_coeffs': np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645]),
#     'camera_intrinsic': np.array([[223.9940010790056, 0, 170.7684322973841], [0, 223.61783486959376, 128.18711828338436], [0, 0, 1]])
# }

experiment_params = [
    {
        'name': 'outdoor1_full_1',
        'dataset': 'outdoor1',
        'start_frame': 10,
        'end_frame': 1500,
        'select_events': 'only_pos',
        'voxel_method': {'method': 'k_events',
                        'k': 60000,
                        't': 0.5,
                        'sliding_window_w': 60000,
                        'sliding_window_t': 0.1},
        'voxel_threshold': 5,
        # 'model': 'data/saver/evflownet_0906_041812_outdoor_dataset1/model1',
        # 'model': '/home/mingyip/Desktop/Motion_Compensated_FlowNet/data/saver/evflownet_1014_085648/model0',
        'model': '/home/mingyip/Desktop/Motion_Compensated_FlowNet/data/saver/evflownet_1014_072622/model13',
        'findE_threshold': 0.4,
        'findE_prob': 0.999,
        'reproject_err_threshold': 0.05
    },
    # {
    #     'name': 'poster_translation',
    #     'dataset': 'poster_translation',
    #     'start_frame': 15,
    #     'end_frame': 510,
    #     'select_events': 'mixed',
    #     'voxel_method': {'method': 'k_events',
    #                     'k': 20000,
    #                     't': 0.5,
    #                     'sliding_window_w': 20000,
    #                     'sliding_window_t': 0.1},
    #     'voxel_threshold': 5,
    #     'model': 'data/saver/evflownet_0906_041812_outdoor_dataset1/model1',
    #     'findE_threshold': 0.1,
    #     'findE_prob': 0.999,
    #     'reproject_err_threshold': 0.05
    # }
]

def main():
    args = configs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_interval = args.logging_interval

    if args.training_instance:
        args.load_path = os.path.join(args.load_path, args.training_instance)
    else:
        args.load_path = os.path.join(args.load_path, "evflownet_{}".format(datetime.now().strftime("%m%d_%H%M%S")))
    if not os.path.exists(args.load_path):
        os.makedirs(args.load_path)



    for ep in experiment_params:

        rpe_stats = []
        rre_stats = []
        trans_e_stats = []


        rpe_rre_info = []
        trans_e_info = []
        gt_interpolated = []
        predict_camera_frame = []
        predict_ts = []


        print(f"{ep['name']}")
        base_path = f"results/{ep['name']}"
        if not os.path.exists(base_path):
            os.makedirs(base_path)


        if ep['dataset'] == 'outdoor1':
            dataset_param = outdoor1_params
        elif ep['dataset'] == 'outdoor2':
            dataset_param = outdoor2_params
        elif ep['dataset'] == 'poster_6dof':
            dataset_param = poster_6dof_params
        elif ep['dataset'] == 'hdr_boxes':
            dataset_param = hdr_boxes_params
        elif ep['dataset'] == 'poster_translation':
            dataset_param = poster_translation_params
        elif ep['dataset'] == 'indoor1':
            dataset_param = indoor1_params
        elif ep['dataset'] == 'indoor2':
            dataset_param = indoor2_params
        elif ep['dataset'] == 'indoor3':
            dataset_param = indoor3_params
        elif ep['dataset'] == 'indoor4':
            dataset_param = indoor4_params            
        elif ep['dataset'] == 'outdoor_night1':
            dataset_param = outdoor_night1_params
        elif ep['dataset'] == 'outdoor_night2':
            dataset_param = outdoor_night2_params
        elif ep['dataset'] == 'outdoor_night3':
            dataset_param = outdoor_night3_params


        with open(f"{base_path}/config.txt", "w") as f:
            f.write("experiment params:\n")
            f.write(pprint.pformat(ep))
            f.write("\n\n\n")
            f.write("dataset params:\n")
            f.write(pprint.pformat(dataset_param))



        print("Load Data Begin. ")
        start_frame = ep['start_frame']
        end_frame = ep['end_frame']
        model_path = ep['model']
        voxel_method = ep['voxel_method']
        select_events = ep['select_events']
        voxel_threshold = ep['voxel_threshold']
        findE_threshold = ep['findE_threshold']
        findE_prob = ep['findE_prob']
        reproject_err_threshold = ep['reproject_err_threshold']


        # Set parameters
        gt_path = dataset_param['gt_path']
        sensor_size = dataset_param['sensor_size']
        camIntrinsic = dataset_param['camera_intrinsic']
        h5Dataset = DynamicH5Dataset(dataset_param['dataset_path'], voxel_method=voxel_method)
        h5DataLoader = torch.utils.data.DataLoader(dataset=h5Dataset, batch_size=1, num_workers=6, shuffle=False)
        
        # model
        print("Load Model Begin. ")
        EVFlowNet_model = EVFlowNet(args).to(device)
        EVFlowNet_model.load_state_dict(torch.load(model_path))
        EVFlowNet_model.eval()
        # EVFlowNet_model.load_state_dict(torch.load('data/model/evflownet_1001_113912_outdoor2_5k/model0'))

        # optimizer
        optimizer = torch.optim.Adam(EVFlowNet_model.parameters(), lr=args.initial_learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
        loss_fun = TotalLoss(args.smoothness_weight, args.photometric_loss_weight)



        print("Start Evaluation. ")    
        for iteration, item in enumerate(tqdm(h5DataLoader)):

            if iteration < start_frame:
                continue

            if iteration > end_frame:
                break

            voxel = item['voxel'].to(device)
            events = item['events'].to(device)
            frame = item['frame'].to(device)
            frame_ = item['frame_'].to(device)
            num_events = item['num_events'].to(device)
            image_name = "{}/img_{:07d}.png".format(base_path, iteration)

            events_vis = events[0].detach().cpu()
            flow_dict = EVFlowNet_model(voxel)
            flow_vis = flow_dict["flow3"][0].detach().cpu()

            # Compose the event images and warp the events with flow
            if select_events == 'only_pos':
                ev_bgn, ev_end, ev_img, timestamps = get_forward_backward_flow_torch(events_vis, flow_vis, voxel_threshold, 1, sensor_size)

            elif select_events == 'only_neg':
                ev_bgn, ev_end, ev_img, timestamps = get_forward_backward_flow_torch(events_vis, flow_vis, voxel_threshold, -1, sensor_size)

            elif select_events == 'mixed':
                ev_bgn_pos, ev_end_pos, ev_img, timestamps = get_forward_backward_flow_torch(events_vis, flow_vis, voxel_threshold, 1, sensor_size)
                ev_bgn_neg, ev_end_neg, ev_img_neg, timestamps_neg = get_forward_backward_flow_torch(events_vis, flow_vis, voxel_threshold, -1, sensor_size)

                ev_bgn_x = torch.cat([ev_bgn_pos[0], ev_bgn_neg[0]])
                ev_bgn_y = torch.cat([ev_bgn_pos[1], ev_bgn_neg[1]])
                ev_end_x = torch.cat([ev_end_pos[0], ev_end_neg[0]])
                ev_end_y = torch.cat([ev_end_pos[1], ev_end_neg[1]])
                ev_bgn = (ev_bgn_x, ev_bgn_y)
                ev_end = (ev_end_x, ev_end_y)


            start_t = item['timestamp_begin'].cpu().numpy()[0]
            end_t = item['timestamp_end'].cpu().numpy()[0]

            # Convert to numpy format
            ev_img_raw = torch_to_numpy(ev_img[0])
            ev_img_bgn = torch_to_numpy(ev_img[1])
            ev_img_end = torch_to_numpy(ev_img[2])
            ev_bgn_xs = torch_to_numpy(ev_bgn[0])
            ev_bgn_ys = torch_to_numpy(ev_bgn[1])
            ev_end_xs = torch_to_numpy(ev_end[0])
            ev_end_ys = torch_to_numpy(ev_end[1])

            timestamps_before = torch_to_numpy(timestamps[0])
            timestamps_after = torch_to_numpy(timestamps[1])
            frame_vis = torch_to_numpy(item['frame'][0])
            frame_vis_ = torch_to_numpy(item['frame_'][0])
            flow_vis = torch_to_numpy(flow_dict["flow3"][0])


            METHOD = "opencv"
            # METHOD = "opengv"

            if METHOD == "opencv":

                ######### Opencv (calculate R and t) #########
                p1 = np.dstack([ev_bgn_xs, ev_bgn_ys]).squeeze()
                p2 = np.dstack([ev_end_xs, ev_end_ys]).squeeze()

                E, mask = cv2.findEssentialMat(p1, p2, cameraMatrix=camIntrinsic, method=cv2.RANSAC, prob=findE_prob, threshold=findE_threshold)
                points, R, t, mask1, triPoints = cv2.recoverPose(E, p1, p2, cameraMatrix=camIntrinsic, mask=mask, distanceThresh=5000)

            elif METHOD == "opengv":

                #### Calculate bearing vector manually ####
                ev_bgn_xs_undistorted = (ev_bgn_xs - 170.7684322973841) / 223.9940010790056
                ev_bgn_ys_undistorted = (ev_bgn_ys - 128.18711828338436) / 223.61783486959376
                ev_end_xs_undistorted = (ev_end_xs - 170.7684322973841) / 223.9940010790056
                ev_end_ys_undistorted = (ev_end_ys - 128.18711828338436) / 223.61783486959376

                bearing_p1 = np.dstack([ev_bgn_xs_undistorted, ev_bgn_ys_undistorted, np.ones_like(ev_bgn_xs)]).squeeze()
                bearing_p2 = np.dstack([ev_end_xs_undistorted, ev_end_ys_undistorted, np.ones_like(ev_end_xs)]).squeeze()

                bearing_p1 /= np.linalg.norm(bearing_p1, axis=1)[:, None]
                bearing_p2 /= np.linalg.norm(bearing_p2, axis=1)[:, None]

                bearing_p1 = bearing_p1.astype('float64')
                bearing_p2 = bearing_p2.astype('float64')

                # focal_length = 223.75
                # reproject_err_threshold = 0.1
                ransac_threshold = 1e-6
                ransac_transformation = pyopengv.relative_pose_ransac(bearing_p1, bearing_p2, "NISTER", threshold=ransac_threshold, iterations=1000, probability=0.999)
                R = ransac_transformation[:, 0:3]
                t = ransac_transformation[:, 3]

            # Interpolate Tw1 and Tw2
            Tw1 = get_interpolated_gt_pose(gt_path, start_t)
            Tw2 = get_interpolated_gt_pose(gt_path, end_t)
            Tw2_inv = inverse_se3_matrix(Tw2)

            # r1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            # r2 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            # r3 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            # r4 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            # r5 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            # r6 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            # t = r5 @ t

            normed_t = t.squeeze() / np.linalg.norm(t)
            gt_t = Tw2[0:3, 3] - Tw1[0:3, 3]
            normed_gt = gt_t / np.linalg.norm(gt_t)

            # print()
            # print(np.rad2deg(np.arccos(np.dot(normed_t, normed_gt))))
            # print(np.rad2deg(np.arccos(np.dot(r1@normed_t, normed_gt))))
            # print(np.rad2deg(np.arccos(np.dot(r2@normed_t, normed_gt))))
            # print(np.rad2deg(np.arccos(np.dot(r3@normed_t, normed_gt))))
            # print(np.rad2deg(np.arccos(np.dot(r4@normed_t, normed_gt))))
            # print(np.rad2deg(np.arccos(np.dot(r5@normed_t, normed_gt))))
            # print(np.rad2deg(np.arccos(np.dot(r6@normed_t, normed_gt))))


            rpe = np.rad2deg(np.arccos(np.dot(normed_t, normed_gt)))
            # raise

            # if iteration == 121:
            #     print(Tw1)
            #     print(Tw2)
            #     print(Tw2_inv)
            #     print(Tw2_inv @ Tw1)

            predict_ts.append(start_t)

            # Store gt vector for later visulizaiton
            gt_interpolated.append(Tw1)
            gt_scale = np.linalg.norm(Tw2[0:3, 3] - Tw1[0:3, 3])
            pd_scale = np.linalg.norm(t)

            t *= gt_scale / pd_scale  # scale translation vector with gt_scale

            # Predicted relative pose 
            P = create_se3_matrix_with_R_t(R, t)
            P_inv = inverse_se3_matrix(P)
            # print(P @ P_inv)




            # Calculate the rpe
            E = Tw2_inv @ Tw1 @ P
            trans_e = np.linalg.norm(E[0:3, 3])

            E_inv = Tw2_inv @ Tw1 @ P_inv
            trans_e_inv = np.linalg.norm(E_inv[0:3, 3])


            # print(Tw2_inv @ Tw1)
            # print(P_inv)
            # print(E_inv)
            # print()

            # print() 
            # print(t)
            # print(Tw1[0:3, 3] - Tw2[0:3, 3])
            # print(Tw1[0:3, 3] - Tw2[0:3, 3] - t.T)

            # print()


            # print(t)
            # print(Tw1[0:3, 3] - Tw2[0:3, 3])
            # print(np.dot(np.linalg.norm(t), np.linalg.norm(Tw1[0:3, 3] - Tw2[0:3, 3])))
            # print(np.arccos(np.dot(np.linalg.norm(t), np.linalg.norm(Tw1[0:3, 3] - Tw2[0:3, 3]))))
            # raise

            rre = np.linalg.norm(logm(E[:3, :3]))
            rre_inv = np.linalg.norm(logm(E_inv[:3, :3]))

            rpe_stats.append(rpe)
            rre_stats.append(rre_inv)
            rpe_rre_info.append([rpe, rre, rre_inv])


            if trans_e_inv/gt_scale > 1.85:
                predict_camera_frame.append(P)

                trans_e_info.append([trans_e, trans_e_inv, gt_scale, trans_e/gt_scale, trans_e_inv/gt_scale, trans_e/gt_scale])
                print(trans_e/gt_scale, trans_e_inv/gt_scale, trans_e/gt_scale, " || ", rpe, rre, rre_inv)

                trans_e_stats.append(trans_e/gt_scale)
            else:                
                trans_e_info.append([trans_e, trans_e_inv, gt_scale, trans_e/gt_scale, trans_e_inv/gt_scale, trans_e_inv/gt_scale])
                print(trans_e/gt_scale, trans_e_inv/gt_scale, trans_e_inv/gt_scale, " || ", rpe, rre, rre_inv)

                trans_e = trans_e_inv
                predict_camera_frame.append(P_inv)

                trans_e_stats.append(trans_e_inv/gt_scale)

            # raise

            # if trans_e/gt_scale > 1.85:

            #     trans_e_info.append([trans_e, trans_e_inv, gt_scale, trans_e/gt_scale, trans_e_inv/gt_scale, trans_e_inv/gt_scale])
            #     print(trans_e, trans_e_inv, gt_scale, trans_e/gt_scale, trans_e_inv/gt_scale, trans_e_inv/gt_scale)


            #     trans_e = trans_e_inv
            #     predict_camera_frame.append(P_inv)
            # else:
            #     predict_camera_frame.append(P)


            #     trans_e_info.append([trans_e, trans_e_inv, gt_scale, trans_e/gt_scale, trans_e_inv/gt_scale, trans_e/gt_scale])
            #     print(trans_e, trans_e_inv, gt_scale, trans_e/gt_scale, trans_e_inv/gt_scale, trans_e/gt_scale)


            cvshow_all_eval(ev_img_raw, ev_img_bgn, ev_img_end, (ev_bgn_xs, ev_bgn_ys), \
                (ev_end_xs, ev_end_ys), timestamps_before, timestamps_after, frame_vis, \
                frame_vis_, flow_vis, image_name, sensor_size, trans_e, gt_scale)


            predict_world_frame = relative_to_absolute_pose(np.array(predict_camera_frame))
            visualize_trajectory(predict_world_frame, "{}/path_{:07d}.png".format(base_path, iteration))
            visualize_trajectory(np.array(gt_interpolated), "{}/gt_path_{:07d}.png".format(base_path, iteration))


        rpe_stats = np.array(rpe_stats)
        rre_stats = np.array(rre_stats)
        trans_e_stats = np.array(trans_e_stats)

        with open(f"{base_path}/final_stats.txt", "w") as f:
            f.write("rpe_median, arpe_deg, arpe_outliner_10, arpe_outliner_15\n")
            f.write(f"{np.median(rpe_stats)}, {np.mean(rpe_stats)}, {100*len(rpe_stats[rpe_stats>10])/len(rpe_stats)}, {100*len(rpe_stats[rpe_stats>15])/len(rpe_stats)}\n\n")
            
            print("rpe_median, arpe_deg, arpe_outliner_10, arpe_outliner_15")
            print(f"{np.median(rpe_stats)}, {np.mean(rpe_stats)}, {100*len(rpe_stats[rpe_stats>10])/len(rpe_stats)}, {100*len(rpe_stats[rpe_stats>15])/len(rpe_stats)}\n")

            f.write("rre_median, arre_rad, arre_outliner_0.05, arpe_outliner_0.1\n")
            f.write(f"{np.median(rre_stats)}, {np.mean(rre_stats)}, {100*len(rre_stats[rre_stats>0.05])/len(rre_stats)}, {100*len(rre_stats[rre_stats>0.1])/len(rre_stats)}\n\n")

            print("rre_median, arre_rad, arre_outliner_0.05, arpe_outliner_0.1")
            print(f"{np.median(rre_stats)}, {np.mean(rre_stats)}, {100*len(rre_stats[rre_stats>0.05])/len(rre_stats)}, {100*len(rre_stats[rre_stats>0.1])/len(rre_stats)}\n\n")

            f.write("trans_e_median, trans_e_avg, trans_e_outliner_0.5, trans_e_outliner_1.0\n")
            f.write(f"{np.median(trans_e_stats)}, {np.mean(trans_e_stats)}, {100*len(trans_e_stats[trans_e_stats>0.5])/len(trans_e_stats)}, {100*len(trans_e_stats[trans_e_stats>1.0])/len(trans_e_stats)}\n")

            print("trans_e_median, trans_e_avg, trans_e_outliner_0.5, trans_e_outliner_1.0\n")
            print(f"{np.median(trans_e_stats)}, {np.mean(trans_e_stats)}, {100*len(trans_e_stats[trans_e_stats>0.5])/len(trans_e_stats)}, {100*len(trans_e_stats[trans_e_stats>1.0])/len(trans_e_stats)}\n")




        with open(f"{base_path}/rpe_rre.txt", "w") as f:
            for row in rpe_rre_info:
                for item in row:
                    f.write(f"{item}, ")
                f.write("\n")

        with open(f"{base_path}/trans_e.txt", "w") as f:
            for row in trans_e_info:
                for item in row:
                    f.write(f"{item}, ")
                f.write("\n")
        
        with open(f"{base_path}/predict_pose.txt", "w") as f:
            for p in predict_world_frame:
                f.write(f"{p}\n")
        
        with open(f"{base_path}/gt_pose.txt", "w") as f:
            for p in np.array(gt_interpolated):
                f.write(f"{p}\n")

        with open(f"{base_path}/predict_tum.txt", "w") as f:
            for ts, p in zip(predict_ts, predict_world_frame):
                qx, qy, qz, qw = rotation_matrix_to_quaternion(p[:3, :3])
                tx, ty, tz = p[:3, 3]
                f.write(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

        with open(f"{base_path}/gt_tum.txt", "w") as f:
            for ts, p in zip(predict_ts, np.array(gt_interpolated)):
                qx, qy, qz, qw = rotation_matrix_to_quaternion(p[:3, :3])
                tx, ty, tz = p[:3, 3]
                f.write(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

        rotation_matrix_to_quaternion

        predict_world_frame = relative_to_absolute_pose(np.array(predict_camera_frame))
        visualize_trajectory(predict_world_frame, f"{base_path}/final_path00.png", show=True)
        visualize_trajectory(predict_world_frame, f"{base_path}/final_path01.png", rotate='x')
        visualize_trajectory(predict_world_frame, f"{base_path}/final_path02.png", rotate='y')
        visualize_trajectory(predict_world_frame, f"{base_path}/final_path03.png", rotate='z')



    

if __name__ == "__main__":
    main()

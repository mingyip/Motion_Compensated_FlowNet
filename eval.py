import io
import os
from tqdm import trange
from tqdm import tqdm
import numpy as np
from datetime import datetime
from losses import *

from test_opengv import RelativePoseDataset

import torch
import h5py
# import cv2
import pyopengv

from config import configs
from data_loader import EventData
from EVFlowNet import EVFlowNet
from dataset import DynamicH5Dataset
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from event_utils import binary_search_h5_gt_timestamp
from vis_utils import cvshow_all, cvshow_all_eval, warp_events_with_flow_torch, get_forward_backward_flow_torch
# from vis_utils import vis_events_and_flows


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

def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def interp_rotation_matrix(start_R, end_R, start_time, end_time, slerp_time):

    rotations = Rot.from_matrix([start_R, end_R])
    key_times = [start_time, end_time]

    slerp = Slerp(key_times, rotations)
    interp_rots = slerp([slerp_time])

    return interp_rots.as_matrix().squeeze()
 
def interp_rigid_matrix(start_R, end_R, start_time, end_time, slerp_time):

    ratio = (slerp_time - start_time) / (end_time - start_time)
    slerp_translation = ratio * start_R[0:3, 3] + (1 - ratio) * end_R[0:3, 3]
    slerp_rotation = interp_rotation_matrix(start_R[0:3, 0:3], end_R[0:3, 0:3], start_time, end_time, slerp_time)

    interp_rigid = np.eye(4, 4)
    interp_rigid[0:3, 0:3] = slerp_rotation
    interp_rigid[0:3, 3] = slerp_translation

    return interp_rigid


def inverse_rigid_matrix(matrix):

    inv_matrix = np.zeros((len(matrix), 4, 4))
    R = matrix[:, 0:3, 0:3]
    t = matrix[:, 0:3, 3]
    R_inv = R.transpose(0, 2, 1)


    for i, (ro, tn) in enumerate(zip(R_inv, t)):

        inv_matrix[i, 0:3, 0:3] = ro
        inv_matrix[i, 0:3, 3] = -ro @ tn
        inv_matrix[i, 3, 3] = 1

    return inv_matrix

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

def visualize_trajectory_coordinate(t):

    def get_img_from_fig(fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    x = t[:, 0]
    y = t[:, 1]
    z = t[:, 2]
    idx = np.arange(len(x))

    fig = plt.figure()
    ax=plt.axes(projection="3d")
    ax.plot3D(x, y, z, 'gray')
    ax.scatter3D(x, y, z, c=idx, cmap='hsv')
    plt.show()

    # plot_img_np = get_img_from_fig(fig)
    # print(plot_img_np.shape)
    # cv2.imshow("abc", plot_img_np)
    # cv2.waitKey(100000)

def visualize_trajectory(world_frame, image_path_name, xmax=None, xmin=None, ymax=None, ymin=None, zmax=None, zmin=None):
    # Visualize path 
    x = world_frame[:, 0, 3]
    y = world_frame[:, 1, 3]
    # z = np.zeros_like(x)
    z = world_frame[:, 2, 3]
    idx = np.arange(len(x))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    if xmax is not None: plt.xlim(right=xmax)
    if xmin is not None: plt.xlim(left=xmin)
    if ymax is not None: plt.ylim(right=ymax)
    if ymax is not None: plt.ylim(right=ymin)
    if zmax is not None: plt.zlim(right=zmax)
    if zmax is not None: plt.zlim(right=zmin)

    ax.plot3D(x, y, z, 'gray')
    ax.scatter3D(x, y, z, c=idx, cmap='hsv')
    # plt.show()
    plt.savefig(image_path_name)
    plt.close()

def get_gt_pose_from_idx(gt_path, idx):
    with h5py.File(gt_path, "r") as gt_file:
        return gt_file['davis']['left']['pose'][idx]

def get_gt_timestamp_from_idx(gt_path, idx):
    with h5py.File(gt_path, "r") as gt_file:
        return gt_file['davis']['left']['pose_ts'][idx]

def get_interpolated_gt_pose(gt_path, interp_ts):

    pose_idx = binary_search_h5_gt_timestamp(gt_path, 0, None, interp_ts, side='right')

    # Calculate interpolation ratio
    pt1_ts = get_gt_timestamp_from_idx(gt_path, pose_idx)
    pt2_ts = get_gt_timestamp_from_idx(gt_path, pose_idx+1)
    ratio = (pt2_ts - interp_ts) / (pt2_ts - pt1_ts)

    # Get 4x4 begin and end Pose 
    pt1_pose = get_gt_pose_from_idx(gt_path, pose_idx)
    pt2_pose = get_gt_pose_from_idx(gt_path, pose_idx+1)

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

    # TODO: remove this part
    voxel_method = {'method': 'k_events',
                    'k': 60000,
                    't': 0.5,
                    'sliding_window_w': 60000,
                    'sliding_window_t': 0.1}


    print("Load Data Begin. ")
    gt_path = "/mnt/Data3/mvsec/data/outdoor_day1/outdoor_day1_gt.hdf5"  # outdoor2

    # h5Dataset = DynamicH5Dataset('data/office.h5', voxel_method=voxel_method)
    h5Dataset = DynamicH5Dataset('data/outdoor_day1_data.h5', voxel_method=voxel_method)
    # h5Dataset = DynamicH5Dataset('data/outdoor_day2_data.h5', voxel_method=voxel_method)
    h5DataLoader = torch.utils.data.DataLoader(dataset=h5Dataset, batch_size=1, num_workers=6, shuffle=False)
    camIntrinsic = np.array([[223.9940010790056, 0, 170.7684322973841], [0, 223.61783486959376, 128.18711828338436], [0, 0, 1]])
    predict_camera_frame = []
    gt_interpolated = []

    # model
    print("Load Model Begin. ")
    EVFlowNet_model = EVFlowNet(args).to(device)
    EVFlowNet_model.load_state_dict(torch.load('data/saver/evflownet_0906_041812_outdoor_dataset1/model1'))
    # EVFlowNet_model.load_state_dict(torch.load('data/model/evflownet_1001_113912_outdoor2_5k/model0'))

    # optimizer
    optimizer = torch.optim.Adam(EVFlowNet_model.parameters(), lr=args.initial_learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    loss_fun = TotalLoss(args.smoothness_weight, args.photometric_loss_weight)

    gt_idx_bgn = None
    gt_idx_end = None
    voxel_ts = []


    f = open("trans_e.txt", "w")
    EVFlowNet_model.eval()
    print("Start Evaluation. ")
    for iteration, item in enumerate(tqdm(h5DataLoader)):

        if iteration < 100:
            continue
        if iteration > 200:
            break

        voxel = item['voxel'].to(device)
        events = item['events'].to(device)
        frame = item['frame'].to(device)
        frame_ = item['frame_'].to(device)
        num_events = item['num_events'].to(device)
        flow_dict = EVFlowNet_model(voxel)

        sensor_size = (256, 336)
        image_name="results/img_{:07d}.png".format(iteration)

        events_vis = events[0].detach().cpu()
        flow_vis = flow_dict["flow3"][0].detach().cpu()

        # Compose the event image and warp the event image with flow
        ev_bgn, ev_end, ev_img, timestamps = get_forward_backward_flow_torch(events_vis, flow_vis, sensor_size)

        start_t = item['timestamp_begin'].cpu().numpy()[0]
        end_t = item['timestamp_end'].cpu().numpy()[0]
        voxel_ts.append(start_t)

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

        # Opencv
        # p1 = np.dstack([ev_bgn_xs, ev_bgn_ys]).squeeze()
        # p2 = np.dstack([ev_end_xs, ev_end_ys]).squeeze()

        # Opengv
        ev_bgn_xs = (ev_bgn_xs - 170.7684322973841) / 223.9940010790056
        ev_bgn_ys = (ev_bgn_ys - 128.18711828338436) / 223.61783486959376
        ev_end_xs = (ev_end_xs - 170.7684322973841) / 223.9940010790056
        ev_end_ys = (ev_end_ys - 128.18711828338436) / 223.61783486959376

        bearing_p1 = np.dstack([ev_bgn_xs, ev_bgn_ys, np.ones_like(ev_bgn_xs)]).squeeze()
        bearing_p2 = np.dstack([ev_end_xs, ev_end_ys, np.ones_like(ev_end_xs)]).squeeze()

        bearing_p1 /= np.linalg.norm(bearing_p1, axis=1)[:, None]
        bearing_p2 /= np.linalg.norm(bearing_p2, axis=1)[:, None]

        ransac_transformation = pyopengv.relative_pose_ransac(bearing_p1, bearing_p2, "NISTER", 0.01, 1000)
        R = ransac_transformation[:, 0:3]
        t = ransac_transformation[:, 3]

        if gt_idx_bgn is None:
            gt_idx_bgn = binary_search_h5_gt_timestamp(gt_path, 0, None, start_t, side='right')

        # Interpolate Tw1 and Tw2
        Tw1 = get_interpolated_gt_pose(gt_path, start_t)
        Tw2 = get_interpolated_gt_pose(gt_path, end_t)
        Tw2_inv = np.eye(4)
        Tw2_inv[0:3, 0:3] = Tw2[0:3, 0:3].transpose()
        Tw2_inv[0:3, 3] = - Tw2[0:3, 0:3].transpose() @ Tw2[0:3, 3]

        # Store gt vector for later visulizaiton
        gt_interpolated.append(Tw1)
        # gt_interpolated.append(Tw2)
        gt_scale = np.linalg.norm(Tw2[0:3, 3] - Tw1[0:3, 3])
        pd_scale = np.linalg.norm(t)
        print("scale", gt_scale, pd_scale)
        t *= gt_scale / pd_scale

        # Compose scaled predicted pose


        # Calculate the rpe
        p = np.eye(4)
        p[0:3, 0:3] = R
        p[0:3, 3]   = np.squeeze(t)
        predict_camera_frame.append(p)

        E = Tw2_inv @ Tw1 @ p
        trans_e = np.linalg.norm(E[0:3, 3])
        print(trans_e, gt_scale, trans_e/gt_scale)  # 0.9732871048392959 0.8097398058707193 1.2019751255685314

        # Calculate the inv rpe
        p_inv = np.eye(4)
        p_inv[0:3, 0:3] = R.T
        p_inv[0:3, 3]   = -R.T @ np.squeeze(t)

        E = Tw2_inv @ Tw1 @ p_inv
        trans_e = np.linalg.norm(E[0:3, 3])
        print(trans_e, gt_scale, trans_e/gt_scale) # 0.9139365323304159 0.8097398058707193 1.1286792691976568


        cvshow_all_eval(ev_img_raw, ev_img_bgn, ev_img_end, (ev_bgn_xs, ev_bgn_ys), \
            (ev_end_xs, ev_end_ys), timestamps_before, timestamps_after, frame_vis, \
            frame_vis_, flow_vis, image_name, sensor_size, trans_e, gt_scale)

        predict_world_frame = relative_to_absolute_pose(np.array(predict_camera_frame))
        visualize_trajectory(predict_world_frame, "results/path_{:07d}.png".format(iteration))
        visualize_trajectory(np.array(gt_interpolated), "results/gt_path_{:07d}.png".format(iteration))

        
        f.write(f"{trans_e} {gt_scale} {trans_e/gt_scale}\n")
    f.close()

    # gt_idx_end = gt_pt2_idx
    
    # with h5py.File(gt_path, "r") as gt_file:
    #     gt_pose = gt_file['davis']['left']['pose']
    #     gt_ts = gt_file['davis']['left']['pose_ts']

    #     gt_pose = gt_pose[gt_idx_bgn:gt_idx_end]
    #     gt_ts = gt_ts[gt_idx_bgn:gt_idx_end]

    # # print(gt_pose.shape)
    # # print(gt_ts.shape)

    # # print(gt_pose[0])

    # qw, qx, qy, qz = rotation_matrix_to_quaternion(gt_pose[:, 0:3, 0:3])

    # # print(qw.shape)
    # # print(qw[0], qx[0], qy[0], qz[0])

    # # rot = quaternion_to_rotation_matrix(qx[0], qy[0], qz[0], qw[0])
    # # print(rot)

    # # raise

    # f = open("gt.txt", "w")
    # for w, x, y, z, p, t in zip(qw, qx, qy, qz, gt_pose, gt_ts):

    #     tx = p[0, 3]
    #     ty = p[1, 3]
    #     tz = p[2, 3]

    #     f.write(f"{t} {tx} {ty} {tz} {x} {y} {z} {w}\n")
    # f.close()





    # gt_interpolated = np.array(gt_interpolated)
    # gt_interpolated = relative_to_absolute_pose(gt_interpolated)

    voxel_ts = np.array(voxel_ts)
    predict_camera_frame = np.array(predict_camera_frame)
    predict_world_frame = relative_to_absolute_pose(predict_camera_frame)


    # print(predict_world_frame.shape)
    # # raise
    # qw, qx, qy, qz = rotation_matrix_to_quaternion(predict_world_frame[:, 0:3, 0:3])
    # f = open("pred.txt", "w")
    # for w, x, y, z, p, t in zip(qw, qx, qy, qz, predict_world_frame, voxel_ts):

    #     tx = p[0, 3]
    #     ty = p[1, 3]
    #     tz = p[2, 3]

    #     f.write(f"{t} {tx} {ty} {tz} {x} {y} {z} {w}\n")
    # f.close()

    # gt_interpolated = np.array(gt_interpolated)
    # visualize_trajectory(gt_interpolated)
    # relative_pose_error(gt_interpolated, predict_camera_frame)
    visualize_trajectory(predict_world_frame)



    

if __name__ == "__main__":
    main()

import io
import os
from tqdm import trange
from tqdm import tqdm
import numpy as np
from datetime import datetime
from losses import *

import torch
import h5py

from config import configs
from data_loader import EventData
from EVFlowNet import EVFlowNet
from dataset import DynamicH5Dataset
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from event_utils import binary_search_h5_gt_timestamp
from vis_utils import cvshow_all, cvshow_all_eval, warp_events_with_flow_torch, get_forward_backward_flow_torch
# from vis_utils import vis_events_and_flows


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

def visualize_trajectory(world_frame):
    # Visualize path 
    x = world_frame[:, 1, 3]
    y = world_frame[:, 2, 3]
    # z = np.zeros_like(x)
    # z = world_frame[:, 2, 3]
    # idx = np.arange(len(x))

    plot = plt.figure()
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, color="red", headwidth=2, headlength=3)    
    plt.show(plot)


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
                    'k': 20000,
                    't': 0.5,
                    'sliding_window_w': 20000,
                    'sliding_window_t': 0.1}


    print("Load Data Begin. ")
    gt_path = "/mnt/Data3/mvsec/data/outdoor_day1/outdoor_day1_gt.hdf5"  # outdoor2

    # h5Dataset = DynamicH5Dataset('data/office.h5', voxel_method=voxel_method)
    h5Dataset = DynamicH5Dataset('data/office_spiral.h5', voxel_method=voxel_method)
    # h5Dataset = DynamicH5Dataset('data/outdoor_day2_data.h5', voxel_method=voxel_method)
    h5DataLoader = torch.utils.data.DataLoader(dataset=h5Dataset, batch_size=1, num_workers=6, shuffle=False)
    camIntrinsic = np.array([[223.9940010790056, 0, 170.7684322973841], [0, 223.61783486959376, 128.18711828338436], [0, 0, 1]])
    predict_camera_frame = []
    gt_interpolated = []

    # model
    print("Load Model Begin. ")
    EVFlowNet_model = EVFlowNet(args).to(device)
    EVFlowNet_model.load_state_dict(torch.load('data/model/evflownet_0922_032701_office_spiral/model1'))

    # optimizer
    optimizer = torch.optim.Adam(EVFlowNet_model.parameters(), lr=args.initial_learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    loss_fun = TotalLoss(args.smoothness_weight, args.photometric_loss_weight)

    EVFlowNet_model.eval()
    print("Start Evaluation. ")
    for iteration, item in enumerate(tqdm(h5DataLoader)):

        # if iteration < 100:
        #     continue
        if iteration > 500:
            break

        voxel = item['voxel'].to(device)
        events = item['events'].to(device)
        frame = item['frame'].to(device)
        frame_ = item['frame_'].to(device)
        num_events = item['num_events'].to(device)
        flow_dict = EVFlowNet_model(voxel)

        sensor_size = (176, 240)
        image_name="results/img_{:07d}.png".format(iteration)

        events_vis = events[0].detach().cpu()
        flow_vis = flow_dict["flow3"][0].detach().cpu()

        # Compose the event image and warp the event image with flow
        ev_bgn, ev_end, ev_img, timestamps = get_forward_backward_flow_torch(events_vis, flow_vis, sensor_size)

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

        p1 = np.dstack([ev_bgn_xs, ev_bgn_ys]).squeeze()
        p2 = np.dstack([ev_end_xs, ev_end_ys]).squeeze()

        E, mask = cv2.findEssentialMat(p1, p2, cameraMatrix=camIntrinsic, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        points, R, t, mask = cv2.recoverPose(E, p1, p2, mask=mask)

        S = np.eye(4)
        S[0:3, 0:3] = R
        S[0:3, 3]   = np.squeeze(t)
        predict_camera_frame.append(S)

        cvshow_all_eval(ev_img_raw, ev_img_bgn, ev_img_end, (ev_bgn_xs, ev_bgn_ys), \
            (ev_end_xs, ev_end_ys), timestamps_before, timestamps_after, frame_vis, \
            frame_vis_, flow_vis, image_name, sensor_size)

    # gt_interpolated = np.array(gt_interpolated)
    # gt_interpolated = relative_to_absolute_pose(gt_interpolated)

    predict_camera_frame = np.array(predict_camera_frame)
    predict_world_frame = relative_to_absolute_pose(predict_camera_frame)


    # gt_interpolated = np.array(gt_interpolated)
    # visualize_trajectory(gt_interpolated)
    # relative_pose_error(gt_interpolated, predict_camera_frame)
    visualize_trajectory(predict_world_frame)



    

if __name__ == "__main__":
    main()

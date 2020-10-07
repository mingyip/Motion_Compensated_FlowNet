#!/usr/bin/env python
import numpy as np
import torch
import math
import cv2 as cv


from event_utils import events_to_image_torch, events_to_timestamp_image_torch

"""
Generates an RGB image where each point corresponds to flow in that direction from the center,
as visualized by flow_viz_tf.
Output: color_wheel_rgb: [1, width, height, 3]
def draw_color_wheel_tf(width, height):
    color_wheel_x = tf.lin_space(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = tf.lin_space(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = tf.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_flow = tf.stack([color_wheel_X, color_wheel_Y], axis=2)
    color_wheel_flow = tf.expand_dims(color_wheel_flow, 0)
    color_wheel_rgb, flow_norm, flow_ang = flow_viz_tf(color_wheel_flow)
    return color_wheel_rgb
"""

def draw_color_wheel_np(width, height):
    color_wheel_x = np.linspace(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = np.linspace(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = np.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_rgb = flow_viz_np(color_wheel_X, color_wheel_Y)
    return color_wheel_rgb

"""
Visualizes optical flow in HSV space using TensorFlow, with orientation as H, magnitude as V.
Returned as RGB.
Input: flow: [batch_size, width, height, 2]
Output: flow_rgb: [batch_size, width, height, 3]
def flow_viz_tf(flow):
    flow_norm = tf.norm(flow, axis=3)
    
    flow_ang_rad = tf.atan2(flow[:, :, :, 1], flow[:, :, :, 0])
    flow_ang = (flow_ang_rad / math.pi) / 2. + 0.5
    
    const_mat = tf.ones(tf.shape(flow_norm))
    hsv = tf.stack([flow_ang, const_mat, flow_norm], axis=3)
    flow_rgb = tf.image.hsv_to_rgb(hsv)
    return flow_rgb, flow_norm, flow_ang_rad
"""

def flow_viz_np(flow_x, flow_y):
    import cv2
    flows = np.stack((flow_x, flow_y), axis=2)
    flows[np.isinf(flows)] = 0
    flows[np.isnan(flows)] = 0
    mag = np.linalg.norm(flows, axis=2)
    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb


def get_forward_backward_flow_torch(events, flow, filter_threshold=5, sensor_size=(180, 240)):

    eps = torch.finfo(flow.dtype).eps
    xs, ys, ts, ps = events

    xs = xs.type(torch.long).to(flow.device)
    ys = ys.type(torch.long).to(flow.device)
    ts = ts.to(flow.device)
    ps = ps.type(torch.long).to(flow.device)

    # Crop out the car part
    xs = xs[ys < 200]
    ts = ts[ys < 200]
    ps = ps[ys < 200]
    ys = ys[ys < 200]

    ts_forward = (ts[-1] - ts) / (ts[-1] - ts[0] + eps)
    ts_backward = (ts[0] - ts) / (ts[-1] - ts[0] + eps)

    xs_end = xs + ts_forward * flow[0,ys,xs]
    ys_end = ys + ts_forward * flow[1,ys,xs]
    xs_bgn = xs + ts_backward * flow[0,ys,xs]
    ys_bgn = ys + ts_backward * flow[1,ys,xs]

    xs_end = torch.clamp(xs_end, min=0, max=sensor_size[1])
    ys_end = torch.clamp(ys_end, min=0, max=sensor_size[0])
    xs_bgn = torch.clamp(xs_bgn, min=0, max=sensor_size[1])
    ys_bgn = torch.clamp(ys_bgn, min=0, max=sensor_size[0])

    timestamp_before = events_to_timestamp_image_torch(xs*1.0, ys*1.0, ts, ps, sensor_size=sensor_size, padding=False)
    timestamp_after  = events_to_timestamp_image_torch(xs_end, ys_end, ts, ps, sensor_size=sensor_size, padding=False)
    timestamp_before = timestamp_before[0]
    timestamp_after  = timestamp_after[0]

    # Construct event images here
    xs_bgn = xs_bgn[ps==1]
    xs_end = xs_end[ps==1]
    ys_bgn = ys_bgn[ps==1]
    ys_end = ys_end[ps==1]

    xs = xs[ps==1]
    ys = ys[ps==1]
    ts = ts[ps==1]
    ps = ps[ps==1]

    ev_img_bgn = events_to_image_torch(xs_bgn, ys_bgn, torch.ones_like(ps), sensor_size=sensor_size, interpolation='bilinear', padding=False)
    ev_img_end = events_to_image_torch(xs_end, ys_end, torch.ones_like(ps), sensor_size=sensor_size, interpolation='bilinear', padding=False)
    ev_img_raw = events_to_image_torch(xs*1.0, ys*1.0, torch.ones_like(ps), sensor_size=sensor_size, interpolation='bilinear', padding=False)

    ev_img_end[ev_img_end < filter_threshold] = 0
    ev_img_bgn[ev_img_bgn < filter_threshold] = 0

    # print(ev_img_bgn.shape, ev_img_end.shape)
    # raise

    xs_bgn_clean = []
    ys_bgn_clean = []
    xs_end_clean = []
    ys_end_clean = []
    for x0, y0, x1, y1 in zip(xs_bgn.long(), ys_bgn.long(), xs_end.long(), ys_end.long()):
        if x0 < sensor_size[1] and x0 > 0 and \
                y0 < sensor_size[0] and y0 > 0 and \
                x1 < sensor_size[1] and x1 > 0 and \
                y1 < sensor_size[0] and y1 > 0 and \
                ev_img_bgn[y0, x0] and ev_img_end[y1, x1]:
            xs_bgn_clean.append(x0)
            ys_bgn_clean.append(y0)
            xs_end_clean.append(x1)
            ys_end_clean.append(y1)        

    xs_bgn_clean = torch.FloatTensor(xs_bgn_clean)
    ys_bgn_clean = torch.FloatTensor(ys_bgn_clean)
    xs_end_clean = torch.FloatTensor(xs_end_clean)
    ys_end_clean = torch.FloatTensor(ys_end_clean)

    return (xs_bgn_clean, ys_bgn_clean), (xs_end_clean, ys_end_clean), (ev_img_raw, ev_img_bgn, ev_img_end), (timestamp_before, timestamp_after)


def warp_events_with_flow_torch(events, flow, sensor_size=(180, 240)):

    eps = torch.finfo(flow.dtype).eps
    xs, ys, ts, ps = events

    xs = xs.type(torch.long).to(flow.device)
    ys = ys.type(torch.long).to(flow.device)
    ts = ts.to(flow.device)
    ps = ps.type(torch.long).to(flow.device)

    # We select positive events only
    xs = xs[ps==1]
    ys = ys[ps==1]
    ts = ts[ps==1]
    ps = ps[ps==1]

    # TODO: Check if ts is correct calibration here
    ts = (ts[-1] - ts) / (ts[-1] - ts[0] + eps)
    
    xs_ = xs + ts * flow[0,ys,xs]
    ys_ = ys + ts * flow[1,ys,xs]

    img = events_to_image_torch(xs*1.0, ys*1.0, ps, sensor_size=sensor_size, interpolation='bilinear', padding=False)
    img_ = events_to_image_torch(xs_, ys_, ps, sensor_size=sensor_size, interpolation='bilinear', padding=False)

    # timestamp = events_to_timestamp_image_torch(xs*1.0, ys*1.0, ts, ps, sensor_size=sensor_size, padding=False)
    # timestamp_ = events_to_timestamp_image_torch(xs_, ys_, ts, ps, sensor_size=sensor_size, padding=False)

    # timestamp = timestamp[0].cpu().numpy()
    # timestamp_ = timestamp_[0].cpu().numpy()

    return img, img_


def cvshow_all_eval(ev_img_raw, ev_img_bgn, ev_img_end, ev_bgn, ev_end, timestamps_before, timestamps_after, frame_vis, frame_vis_, flow, image_name, sensor_size, trans_e, gt_scale):

    ev_img_raw = cv.cvtColor(ev_img_raw, cv.COLOR_GRAY2RGB)
    ev_img_bgn = cv.cvtColor(ev_img_bgn, cv.COLOR_GRAY2RGB)
    ev_img_end = cv.cvtColor(ev_img_end, cv.COLOR_GRAY2RGB)
    
    t1_mask = timestamps_before == 0
    t2_mask = timestamps_after == 0
    timestamps_before = (timestamps_before * 255).astype(np.uint8)
    timestamps_after = (timestamps_after * 255).astype(np.uint8)
    timestamps_before = cv.applyColorMap(timestamps_before, cv.COLORMAP_PARULA)
    timestamps_after = cv.applyColorMap(timestamps_after, cv.COLORMAP_PARULA)
    timestamps_before[t1_mask] = 0
    timestamps_after[t2_mask] = 0

    frame_vis = cv.cvtColor(frame_vis, cv.COLOR_GRAY2RGB)
    frame_vis_ = cv.cvtColor(frame_vis_, cv.COLOR_GRAY2RGB)

    flow = flow_viz_np(flow[0], flow[1])
    flow_masked = np.copy(flow)
    flow_masked[ev_img_raw == 0] = 0
    flow[-50:, :50, :] = draw_color_wheel_np(50, 50)

    # top = np.hstack([ev_img_raw, np.zeros_like(ev_img_raw), timestamps_before/255., flow/255., frame_vis/255.])
    # bot = np.hstack([ev_img_bgn, ev_img_end, timestamps_after/255., flow_masked/255., frame_vis_/255.])
    top = np.hstack([ev_img_raw, np.zeros_like(ev_img_raw), timestamps_before/255., flow/255., flow_masked/255.])
    bot = np.hstack([ev_img_bgn, ev_img_end, ev_img_bgn-ev_img_end, ev_img_end-ev_img_bgn, frame_vis_/255.])
    final = np.vstack([top, bot])
    cv.putText(final, "trans_e: " + str(trans_e), (340, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv.putText(final, "gt_scale: " + str(gt_scale), (340, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv.putText(final, "normed_trans_e: " + str(trans_e/gt_scale), (340, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv.imshow("Image", final)
    cv.imwrite(image_name, final*255)
    cv.waitKey(1)


 
# def cvshow_all(voxel, flow=None, frame=None, frame_=None, compensated=None, timestamp1=None, timestamp2=None, image_name="image.png", photometric_vis=None, sensor_size=(256, 336)):
def cvshow_all(voxel, flow=None, frame=None, frame_=None, compensated=None, image_name="image.png", sensor_size=(256, 336)):

    # TODO: check voxel, frame, flow shape
    # assert voxel.shape[1:] == frame.shape
    # assert flow.shape[1:] == frame.shape


    if frame is None: frame = np.zeros(sensor_size)
    if frame_ is None: frame_ = np.zeros(sensor_size)
    if compensated is None: compensated = np.zeros(sensor_size)

    # vis_warp, vis_prev, vis_next, vis_dist = photometric_vis

    voxel = cv.cvtColor(voxel, cv.COLOR_GRAY2RGB)
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    frame_ = cv.cvtColor(frame_, cv.COLOR_GRAY2RGB)
    compensated = cv.cvtColor(compensated, cv.COLOR_GRAY2BGR)
    # vis_warp = cv.cvtColor(vis_warp, cv.COLOR_GRAY2BGR)
    # vis_prev = cv.cvtColor(vis_prev, cv.COLOR_GRAY2BGR)
    # vis_next = cv.cvtColor(vis_next, cv.COLOR_GRAY2BGR)
    # vis_dist = cv.cvtColor(vis_dist, cv.COLOR_GRAY2BGR)

    
    # t1_mask = timestamp1 == 0
    # t2_mask = timestamp2 == 0

    # # timestamp1 = cv.cvtColor(timestamp1, cv.COLOR_GRAY2BGR)
    # # timestamp2 = cv.cvtColor(timestamp2, cv.COLOR_GRAY2BGR)

    # timestamp1 = (timestamp1 * 255).astype(np.uint8)
    # timestamp2 = (timestamp2 * 255).astype(np.uint8)
    # # print(timestamp1.shape, timestamp1.dtype)

    # timestamp1 = cv.applyColorMap(timestamp1, cv.COLORMAP_PARULA)
    # timestamp2 = cv.applyColorMap(timestamp2, cv.COLORMAP_PARULA)

    # timestamp1[t1_mask] = 0
    # timestamp2[t2_mask] = 0

    # # print(timestamp1.shape, timestamp1.dtype)
    # # raise


    flow = flow_viz_np(flow[0], flow[1])
    flow_masked = np.copy(flow)
    flow_masked[voxel == 0] = 0
    flow[-50:, :50, :] = draw_color_wheel_np(50, 50)

    top = np.hstack([voxel, flow/255, frame/255])
    mid = np.hstack([compensated, flow_masked/255, frame_/255])
    # bot = np.hstack([vis_prev, vis_warp, vis_dist, np.abs(vis_prev-vis_next)])
    # ts = np.hstack([timestamp1, timestamp2, np.zeros_like(timestamp2)])
    final = np.vstack([top, mid])

    cv.imshow("Image", final)
    cv.imwrite(image_name, final*255)
    cv.waitKey(1)













# def vis_events_and_flows(voxel, events, flow, frame, frame_, sensor_size=(180, 240), image_name="img.png"):

#     xs = events[:, 0]
#     ys = events[:, 1]
#     ts = events[:, 2] 
#     ps = events[:, 3]

#     img, img_ = warp_events_with_flow_torch((xs, ys, ts, ps), flow, sensor_size=sensor_size)
#     img = img.cpu().numpy()
#     img_ = img_.cpu().numpy()

#     cvshow_all(voxel=img, flow=flow, frame=frame, frame_=frame_, compensated=img_, image_name=image_name)

# def cvshow_voxel_grid(voxelgrid, cmp=cv.COLORMAP_JET):
#     mask = get_voxel_grid_as_image(voxelgrid, True, False)
#     sidebyside = get_voxel_grid_as_image(voxelgrid, True, True)
#     sidebyside = sidebyside.astype(np.uint8)
#     color_img = cv.applyColorMap(sidebyside, cmp)
#     color_img[mask==0] = 0

#     cv.imshow("Image", color_img)
#     cv.waitKey(50000)
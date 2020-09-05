#!/usr/bin/env python
import numpy as np
import torch
import math
import cv2 as cv


from event_utils import events_to_image_torch

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

def warp_events_with_flow_torch(events, flow, sensor_size=(180, 240)):

    eps = torch.finfo(flow.dtype).eps
    xs, ys, ts, ps = events

    xs = xs[0]
    ys = ys[0]
    ts = ts[0]
    ps = ps[0]
    flow = flow[0]

    xs = xs.type(torch.long).to(flow.device)
    ys = ys.type(torch.long).to(flow.device)
    ts = ts.to(flow.device)
    ps = ps.type(torch.long).to(flow.device)

    xs = xs[ps==1]
    ys = ys[ps==1]
    ts = ts[ps==1]
    ps = ps[ps==1]

    # TODO: Check if ts is correct calibration here
    ts = (ts - ts[0]) / (ts[-1] - ts[0] + eps)
    
    xs_ = xs + (ts[-1]-ts) * flow[0,ys,xs]
    ys_ = ys + (ts[-1]-ts) * flow[1,ys,xs]

    img = events_to_image_torch(xs*1.0, ys*1.0, ps, sensor_size=sensor_size, interpolation='bilinear', padding=False)
    img_ = events_to_image_torch(xs_, ys_, ps, sensor_size=sensor_size, interpolation='bilinear', padding=False)
    return img, img_

def vis_events_and_flows(voxel, events, flow, sensor_size=(180, 240), image_name="img.png"):

    xs = events[:, 0]
    ys = events[:, 1]
    ts = events[:, 2] 
    ps = events[:, 3]

    img, img_ = warp_events_with_flow_torch((xs, ys, ts, ps), flow, sensor_size=sensor_size)
    img = img.cpu().numpy()
    img_ = img_.cpu().numpy()

    cvshow_all(voxel=img, flow=flow[0].cpu().numpy(), frame=None, compensated=img_, image_name=image_name)

def cvshow_voxel_grid(voxelgrid, cmp=cv.COLORMAP_JET):
    mask = get_voxel_grid_as_image(voxelgrid, True, False)
    sidebyside = get_voxel_grid_as_image(voxelgrid, True, True)
    sidebyside = sidebyside.astype(np.uint8)
    color_img = cv.applyColorMap(sidebyside, cmp)
    color_img[mask==0] = 0

    cv.imshow("Image", color_img)
    cv.waitKey(50000)

def cvshow_all(voxel, flow=None, frame=None, compensated=None, image_name="image.png", cmp=cv.COLORMAP_JET):

    # TODO: check voxel, frame, flow shape
    # assert voxel.shape[1:] == frame.shape
    # assert flow.shape[1:] == frame.shape

    # TODO: check datatype tensor
    voxel = cv.cvtColor(voxel, cv.COLOR_GRAY2BGR)
    flow = flow_viz_np(flow[0], flow[1]) * 255
    
    if frame is None:
        frame = np.zeros_like(voxel)
    else:
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)

    if compensated is None:
        compensated = np.zeros_like(frame)
    else:
        compensated = cv.cvtColor(compensated, cv.COLOR_GRAY2BGR)
    
    flow_masked = np.copy(flow)
    flow_masked[voxel == 0] = 0

    bot = np.hstack([compensated, flow_masked / 255])
    flow[-40:, :40, :] = draw_color_wheel_np(40, 40)
    top = np.hstack([voxel, flow/255])
    final = np.vstack([top, bot])


    cv.imshow("Image", final)
    cv.imwrite(image_name, final*255)
    cv.waitKey(1)
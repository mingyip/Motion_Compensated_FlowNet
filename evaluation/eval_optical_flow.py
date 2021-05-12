import numpy as np
import h5py
import cv2


from vis_utils import flow_viz_np


gt_path = '/mnt/Data3/mvsec/data/outdoor_day1/outdoor_day1_gt.hdf5'

with h5py.File(gt_path, "r") as gt_file:


    for idx, flow in enumerate(gt_file['davis']['left']['flow_dist']):

        print(flow.shape)
        cv2.imwrite(f"flow_{idx}.png", flow_viz_np(flow[0], flow[1]))


from torch.utils.data import Dataset
import numpy as np
import random
import time
import torch
import h5py
import os
import cv2
import imutils
from scipy import ndimage, misc
from numpy import arange
# local modules
from utils.data_augmentation import Compose, RobustNorm
from utils.data import data_sources
from event_utils import events_to_voxel_torch, \
                        events_to_image_torch, \
                        events_to_neg_pos_voxel_torch, \
                        binary_search_h5_dset, \
                        binary_search_torch_tensor                        




class BaseDataset(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized frames and optic flow if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            method={'method':'k_events'}
            Default is 'between_frames'.
    """

    def get_frame(self, index):
        """
        Get frame at index
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        """
        raise NotImplementedError

    def get_events_count_images(self, idx0, idx1):
        """
        Get events count images between idx0, idx1
        """
        raise NotImplementedError

    def get_events_time_images(self, idx0, idx1):
        """
        Get events time images between idx0, idx1
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def get_events_by_idx(self, idx):
        """
        Get events of an idx
        """
        raise NotImplementedError

    def get_events_by_idx_list(self, idx):
        """
        Get events of a list idx
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError

    def find_ts_frame_index(self, timestamp):
        """
        Given a timestamp, find the frame index
        """
        raise NotImplementedError

    def find_ts_frame(self, timestamp):
        """
        Given a timestamp, get frame and flow
        """
        raise NotImplementedError

    def __init__(self, data_path,
                transforms={}, sensor_resolution=None, preload_events=True,
                voxel_method={'method': 'k_events'}, num_bins=9,
                max_length=None, combined_voxel_channels=True):

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = sensor_resolution
        self.data_source_idx = -1
        self.has_flow = False
        self.preload_events = preload_events

        self.sensor_resolution, self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = \
            None, None, None, None, None, None

        self.load_data(data_path)

        if self.sensor_resolution is None or self.has_flow is None or self.t0 is None \
                or self.tk is None or self.num_events is None or self.frame_ts is None \
                or self.num_frames is None:
            raise Exception("Dataloader failed to intialize all required members")

        # self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk - self.t0
        self.set_voxel_method(voxel_method)

        self.normalize_voxels = False
        if 'RobustNorm' in transforms.keys():
            vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
            del (transforms['RobustNorm'])
            self.normalize_voxels = True
            self.vox_transform = Compose(vox_transforms_list)

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        if max_length is not None:
            self.length = min(self.length, max_length + 1)


    def __getitem__(self, index, seed=None, return_frame=False):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        # index = 20
        # index = 330
        # index = 1830
        # index = 1230

        # # TODO: set max frames and skip 
        # # TODO: fix the last frame
        # # TODO: set crop size

        # if index < 200:
        #     index += 1000

        H, W = self.sensor_resolution
        idx0, idx1 = self.event_indices[index]    
        xs, ys, ts, ps = self.get_events(idx0, idx1)

        ys = ys[xs < W]
        ts = ts[xs < W]
        ps = ps[xs < W]
        xs = xs[xs < W]

        xs = xs[ys < H]
        ts = ts[ys < H]
        ps = ps[ys < H]
        ys = ys[ys < H]

        num_events = len(xs)

        if self.voxel_method['method'] == 'between_frames' or \
            self.voxel_method['method'] == 't_seconds':
            xs = np.hstack([xs, np.zeros((100000 - num_events))])
            ys = np.hstack([ys, np.zeros((100000 - num_events))])
            ts = np.hstack([ts, np.zeros((100000 - num_events))])
            ps = np.hstack([ps, np.zeros((100000 - num_events))])

        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy((ts-ts[0]).astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32))
        

        frame_idx_start, frame_idx_end = self.frame_indices[index]
        frame = self.get_frame(frame_idx_start)
        frame_ = self.get_frame(frame_idx_end)

        voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)
        voxel = self.transform_voxel(voxel, seed)

        item = {'events': torch.stack([xs, ys, ts, ps], dim=0),
                    'voxel': voxel,
                    'frame': frame,
                    'frame_': frame_,
                    'num_events': num_events,
                    'timestamp_begin': self.timestamps[index][0],
                    'timestamp_end': self.timestamps[index][1],
                    # 'timestamp_begin': self.timestamps[index][0] - self.get_events_by_idx(0)[2],
                    # 'timestamp_end': self.timestamps[index][1] - self.get_events_by_idx(0)[2],
                    'data_source_idx': self.data_source_idx,
                    'dt': ts[-1] - ts[0]}

        return item

    def __len__(self):
        return self.length
        # return 1000

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        frame_indices = []
        for start_t, end_t in self.timestamps:
            start_idx = binary_search_h5_dset(self.frame_ts, start_t)
            end_idx = binary_search_h5_dset(self.frame_ts, end_t)
            frame_indices.append([start_idx, end_idx])
            # assert start_idx != end_idx, "frame start_idx shoudn't be the same as end_idx"
        return frame_indices        

    def compute_event_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        event_indices = []
        for ts in self.timestamps:
            start_t, end_t = ts
            start_idx = self.find_ts_index(start_t)
            end_idx = self.find_ts_index(end_t)
            event_indices.append([start_idx, end_idx])
        return np.array(event_indices)

    def compute_timestamps_t_seconds(self):
        """
        For each block of time (using t_seconds), find the start and
        end indices of the corresponding events
        """
        timestamps = []
        for i in range(self.__len__()):
            t0 = self.voxel_method['sliding_window_t'] * i
            t1 = t0 + self.voxel_method['t']
            timestamps.append([t0, t1])
        return np.array(timestamps)

    def compute_timestamps_k_events(self):
        """
        For each block of time (using k_events), find the start and
        end indices of the corresponding events
        """
        timestamps = []
        for i in range(self.__len__()):
            idx0 = i * self.voxel_method['sliding_window_w']
            idx1 = idx0 + self.voxel_method['k']
            event_0 = self.get_events_by_idx(idx0)
            event_1 = self.get_events_by_idx(idx1)
            timestamps.append([event_0[2], event_1[2]])
        return np.array(timestamps)

    def compute_event_indices_k_events(self):
        """
        For each block of time (using k_events), find the start and
        end indices of the corresponding events
        """
        k_events = self.voxel_method['sliding_window_w']
        event_indices = []
        for i in range(self.__len__()):
            event_indices.append([i * k_events, i * k_events + self.voxel_method['k']])
        return np.array(event_indices)

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        
        if self.voxel_method['method'] == 't_seconds':
            self.length = max(int((self.duration - voxel_method['t']) / voxel_method['sliding_window_t'] + 1), 0)
            self.timestamps = self.compute_timestamps_t_seconds()
            self.event_indices = self.compute_event_indices()
            self.frame_indices = self.compute_frame_indices()
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - 1
            self.timestamps = np.dstack([self.frame_ts[:-1], self.frame_ts[1:]])[0]
            self.frame_indices = np.array([[i, i+1] for i in range(self.__len__() + 1)])
            self.event_indices = self.compute_event_indices()
        elif self.voxel_method['method'] == 'k_events':
            self.length = max(int((self.num_events - voxel_method['k']) / voxel_method['sliding_window_w'] + 1), 0)
            self.timestamps = self.compute_timestamps_k_events()
            self.event_indices = self.compute_event_indices_k_events()
            self.frame_indices = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    # def get_event_indices(self, index):
    #     """
    #     Get start and end indices of events at index
    #     """
    #     idx0, idx1 = self.event_indices[index]
    #     if not (idx0 >= 0 and idx1 <= self.num_events):
    #         raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
    #     return idx0, idx1

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)

        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        return voxel_grid

    def transform_frame(self, frame, seed):
        """
        Augment frame and turn into tensor
        """
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow


class DynamicH5Dataset(BaseDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            frame = h5_file['images']['image{:09d}'.format(index)][:]
        return frame

    def get_flow(self, index):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            flow = h5_file['flow']['flow{:09d}'.format(index)][:]
        return flow

    def get_events_count_images(self, idx0, idx1):
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        p_mask = ps == 1.0
        n_mask = ps == -1.0

        pos_image = np.zeros(self.sensor_resolution)
        neg_image = np.zeros(self.sensor_resolution)

        np.add.at(pos_image, tuple([ys[p_mask], xs[p_mask]]), 1)
        np.add.at(neg_image, tuple([ys[n_mask], xs[n_mask]]), 1)

        return pos_image, neg_image

    def get_events_time_images(self, idx0, idx1):
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        p_mask = ps == 1.0
        n_mask = ps == -1.0

        pos_image = np.zeros(self.sensor_resolution)
        neg_image = np.zeros(self.sensor_resolution)


        for x, y, t, p in zip(xs, ys, ts, ps):
            if p == 1.0:
                pos_image[y, x] = t
            else:
                neg_image[y, x] = t

        # print(np.max(pos_image), np.min(pos_image))
        pos_image /= np.max(pos_image)
        neg_image /= np.max(neg_image)
        # print(np.max(pos_image), np.min(pos_image))
        # raise

        return pos_image, neg_image


    def get_events(self, idx0, idx1):
        if self.preload_events:
            xs = self.xs[idx0:idx1]
            ys = self.ys[idx0:idx1]
            ts = self.ts[idx0:idx1]
            ps = self.ps[idx0:idx1]
        else:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                xs = h5_file['events/xs'][idx0:idx1]
                ys = h5_file['events/ys'][idx0:idx1]
                ts = h5_file['events/ts'][idx0:idx1]
                ps = h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def get_events_by_idx(self, idx):
        if self.preload_events:
            xs = self.xs[idx]
            ys = self.ys[idx]
            ts = self.ts[idx]
            ps = self.ps[idx]
        else:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                xs = h5_file['events/xs'][idx]
                ys = h5_file['events/ys'][idx]
                ts = h5_file['events/ts'][idx]
                ps = h5_file['events/ps'][idx] * 2.0 - 1.0
        return xs, ys, ts, ps

    def get_events_by_idx_list(self, idx):
        idx = list(idx)
        if self.preload_events:
            xs = self.xs[idx]
            ys = self.ys[idx]
            ts = self.ts[idx]
            ps = self.ps[idx]
        else:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                xs = h5_file['events/xs'][idx]
                ys = h5_file['events/ys'][idx]
                ts = h5_file['events/ts'][idx]
                ps = h5_file['events/ps'][idx] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_all(self):
        start = time.time()
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            self.xs = h5_file['events/xs'][:]
            self.ys = h5_file['events/ys'][:]
            self.ts = h5_file['events/ts'][:]
            self.ps = h5_file['events/ps'][:] * 2.0 - 1.0
        print("Finished loading events: ", time.time()-start)


    def load_data(self, data_path):
        self.h5_file_path = data_path

        try:
            h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.preload_events:
            self.load_all()

        if self.sensor_resolution is None:
            self.sensor_resolution = h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]

        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in h5_file.keys() and len(h5_file['flow']) > 0
        self.t0 = h5_file['events/ts'][0]
        self.tk = h5_file['events/ts'][-1]
        self.num_events = h5_file.attrs["num_events"]
        self.num_frames = h5_file.attrs["num_imgs"]

        self.frame_ts = []
        for img_name in h5_file['images']:
            self.frame_ts.append(h5_file['images/{}'.format(img_name)].attrs['timestamp'])

        data_source = h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

        h5_file.close()

    def find_ts_index(self, timestamp):
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            idx = binary_search_h5_dset(h5_file['events/ts'], timestamp)
        return idx

    def find_ts_frame_index(self, timestamp):
        indices = np.array(self.frame_indices)
        indices = indices[:, 1]
        _, _, ts, _ = self.get_events_by_idx_list(indices)
        
        idx = binary_search_h5_dset(ts, timestamp)
        return idx
    
    def find_ts_frame(self, timestamp):
        idx = self.find_ts_frame_index(timestamp) - 1

        # frame = self.get_frame(idx)
        flow = self.get_flow(idx)
        # return frame, flow
        return flow


    # def compute_frame_indices(self):
    #     with h5py.File(self.h5_file_path, 'r') as h5_file:
    #         frame_indices = []
    #         start_idx = 0
    #         for img_name in h5_file['images']:
    #             end_idx = h5_file['images/{}'.format(img_name)].attrs['event_idx']
    #             frame_indices.append([start_idx, end_idx])
    #             start_idx = end_idx
    #     return frame_indices
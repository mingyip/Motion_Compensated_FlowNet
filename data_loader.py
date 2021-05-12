import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from config import configs
from PIL import Image
import numpy as np
import os
import h5py
import cv2

_MAX_SKIP_FRAMES = 6
_TEST_SKIP_FRAMES = 4
_N_SKIP = 1

class EventData(Dataset):
    """
    args:
    data_folder_path:the path of data
    split:'train' or 'test'
    """
    def __init__(self, data_folder_path, split, count_only=False, time_only=False, skip_frames=False):
        self._data_folder_path = data_folder_path
        self._split = split
        self._count_only = count_only
        self._time_only = time_only
        self._skip_frames = skip_frames
        self.args = configs()
        self.event_data_paths, self.n_ima = self.read_file_paths(self._data_folder_path, self._split)

    def get_frame(self, index):
        with h5py.File('data/outdoor_day1_data.h5', 'r') as h5_file:
            frame = h5_file['images']['image{:09d}'.format(index)][:]
        return frame

    def get_flow(self, index):
        with h5py.File('data/outdoor_day1_data.h5', 'r') as h5_file:
            flow = h5_file['flow']['flow{:09d}'.format(index)][:]
        return flow

    def get_events(self, idx0, idx1):
        with h5py.File('data/outdoor_day1_data.h5', 'r') as h5_file:
            xs = h5_file['events/xs'][idx0:idx1]
            ys = h5_file['events/ys'][idx0:idx1]
            ts = h5_file['events/ts'][idx0:idx1]
            ps = h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

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


    def __getitem__(self, index):

        with h5py.File('data/outdoor_day1_data.h5', 'r') as h5_file:
            self.t0 = h5_file['events/ts'][0]
            self.tk = h5_file['events/ts'][-1]
            self.num_frames = h5_file.attrs["num_imgs"]
            self.num_events = h5_file.attrs["num_events"]



        # 获得image_times event_count_images event_time_images image_iter prefix cam
        image_iter = 0
        for i in self.n_ima:
            if index < i:
                break
            image_iter += 1
        image_iter -= 1
        if image_iter % 2 == 0:
            cam = 'left'
        else:
            cam = 'right'
        prefix = self.event_data_paths[image_iter]
        image_iter = index - self.n_ima[image_iter]


        filename = "/mnt/Data1/dataset/evflow-data/outdoor_day2/left_events/left_event{:05d}.png".format(image_iter)
        event_count_images, event_time_images, image_times = np.load(filename + ".npy", encoding='bytes', allow_pickle=True)

        # # print(event_count_images)
        # event_count_images *= 10000

        # print(index)
        # h0_stack = event_count_images[0,:,:,0] + event_count_images[1,:,:,0] + event_count_images[2,:,:,0] + event_count_images[3,:,:,0] + event_count_images[4,:,:,0] + event_count_images[5,:,:,0]
        # cv2.imwrite("6.png", h0_stack)
        # # stack = np.hstack([h0_stack, h1_stack, h1_stack - h0_stack])
        # # cv2.imwrite("1.png", h0_stack)
        # # cv2.imwrite("2.png", h1_stack)

        # for i in range(len(event_count_images)):
        #     cv2.imwrite("{}.png".format(i), event_count_images[i,:,:,0])
        # cv2.imshow("count_images", stack)
        # cv2.waitKey(10000)
        # raise

        event_count_images = torch.from_numpy(event_count_images.astype(np.int16))
        event_time_images = torch.from_numpy(event_time_images.astype(np.float32))
        image_times = torch.from_numpy(image_times.astype(np.float64))

        if self._split is 'test':
            if self._skip_frames:
                n_frames = _TEST_SKIP_FRAMES
            else:
                n_frames = 1
        else:
            n_frames = np.random.randint(low=1, high=_MAX_SKIP_FRAMES+1) * _N_SKIP

        # print(n_frames)
        # raise
        timestamps = [image_times[0], image_times[n_frames]]
        event_count_image, event_time_image = self._read_events(event_count_images, event_time_images, n_frames)

        prev_img_name = "/mnt/Data1/dataset/evflow-data/outdoor_day2/left_images/left_image{:05d}.png".format(image_iter)
        next_img_name = "/mnt/Data1/dataset/evflow-data/outdoor_day2/left_images/left_image{:05d}.png".format(image_iter + n_frames)
        prev_img_path = prev_img_name
        next_img_path = next_img_name

        prev_image = Image.open(prev_img_path)
        next_image = Image.open(next_img_path)

        #transforms
        rand_flip = np.random.randint(low=0, high=2)
        rand_rotate = np.random.randint(low=-30, high=30)
        x = np.random.randint(low=1, high=(event_count_image.shape[1]-self.args.image_height))
        y = np.random.randint(low=1, high=(event_count_image.shape[2]-self.args.image_width))
        if self._split == 'train':
            if self._count_only:
                event_count_image = F.to_pil_image(event_count_image / 255.)
                # random_flip
                if rand_flip == 0:
                    event_count_image = event_count_image.transpose(Image.FLIP_LEFT_RIGHT)
                # random_rotate
                event_image = event_count_image.rotate(rand_rotate)
                # random_crop
                event_image = F.to_tensor(event_image) * 255.
                event_image = event_image[:,x:x+self.args.image_height,y:y+self.args.image_width]
            elif self._time_only:
                event_time_image = F.to_pil_image(event_time_image)
                # random_flip
                if rand_flip == 0:
                    event_time_image = event_time_image.transpose(Image.FLIP_LEFT_RIGHT)
                # random_rotate
                event_image = event_time_image.rotate(rand_rotate)
                # random_crop
                event_image = F.to_tensor(event_image)
                event_image = event_image[:,x:x+self.args.image_height,y:y+self.args.image_width]
            else:

                
                event_count_image = F.to_pil_image(event_count_image / 255.)
                event_time_image = F.to_pil_image(event_time_image)

                # random_flip
                if rand_flip == 0:
                    event_count_image = event_count_image.transpose(Image.FLIP_LEFT_RIGHT)
                    event_time_image = event_time_image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # print(np.max(np.array(event_time_image)))
                # raise
                # random_rotate
                event_count_image1 = event_count_image.rotate(rand_rotate)
                event_time_image = event_time_image.rotate(rand_rotate)


                # event_count_image = np.array(event_count_image)
                # event_count_image1 = np.array(event_count_image1)
                # out = np.hstack([event_count_image[...,1], event_count_image1[...,1]])
                # cv2.imshow("image", out*255)
                # cv2.waitKey(100000)
                # raise
                # random_crop
                event_count_image = F.to_tensor(event_count_image)
                event_time_image = F.to_tensor(event_time_image) * 255.
                event_image = torch.cat((event_count_image,event_time_image), dim=0)
                event_image = event_image[...,x:x+self.args.image_height,y:y+self.args.image_width]

            if rand_flip == 0:
                prev_image = prev_image.transpose(Image.FLIP_LEFT_RIGHT)
                next_image = next_image.transpose(Image.FLIP_LEFT_RIGHT)
            prev_image = prev_image.rotate(rand_rotate)
            next_image = next_image.rotate(rand_rotate)
            prev_image = F.to_tensor(prev_image)
            next_image = F.to_tensor(next_image)
            prev_image = prev_image[...,x:x+self.args.image_height,y:y+self.args.image_width]
            next_image = next_image[...,x:x+self.args.image_height,y:y+self.args.image_width]


            event_count_image = np.array(event_count_image)
            event_time_image = np.array(event_time_image)
            prev_image = np.array(prev_image)
            next_image = np.array(next_image)

            # print(event_time_image.shape)
            # out1 = np.hstack((event_count_image[0]*255, event_count_image[1]*255))
            # out2 = np.hstack((event_time_image[0], event_time_image[1]))
            # # out3 = np.hstack((prev_image, next_image))
            # out = np.vstack((out1, out2))
            # cv2.imshow("image", out)
            # cv2.waitKey(100000)
            # raise


            # print(np.max(np.array(next_image)))
            # print(np.max(np.array(prev_image)))
            # print(torch.max(event_image[0]))
            # print(torch.max(event_image[1]))
            # print(torch.max(event_image[2]))
            # print(torch.max(event_image[3]))
            # raise
            # event_time_image = np.array(event_time_image)
            # print(event_time_image.shape)
            # cv2.imshow("image", event_time_image[0])
            # cv2.waitKey(10000)
            # raise

        else:
            if self._count_only:
                event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_count_image / 255.), 
                                            (self.args.image_height, self.args.image_width)))
                event_image = event_image * 255.
            elif self._time_only:
                event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_time_image), 
                                            (self.args.image_height, self.args.image_width)))
            else:
                event_image = torch.cat((event_count_image / 255.,event_time_image), dim=0)
                event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_image), 
                                            (self.args.image_height, self.args.image_width)))
                event_image[:2,...] = event_image[:2,...] * 255.
            prev_image = F.to_tensor(F.center_crop(prev_image, (self.args.image_height, self.args.image_width)))
            next_image = F.to_tensor(F.center_crop(next_image, (self.args.image_height, self.args.image_width)))

        
        # print(event_image.shape)
        # print(prev_image.shape)
        # print(next_image.shape)
        # print(timestamps)
        # raise

        return event_image, prev_image, next_image, timestamps

    def __len__(self):
        return self.n_ima[-1]

    def _read_events(self,
                     event_count_images,
                     event_time_images,
                     n_frames):
        #event_count_images = event_count_images.reshape(shape).type(torch.float32)
        event_count_image = event_count_images[:n_frames, :, :, :]
        event_count_image = torch.sum(event_count_image, dim=0).type(torch.float32)
        # p = torch.max(event_count_image)
        event_count_image = event_count_image.permute(2,0,1)

        #event_time_images = event_time_images.reshape(shape).type(torch.float32)
        
        event_time_image = event_time_images[:n_frames, :, :, :]
        event_time_image = torch.max(event_time_image, dim=0)[0]
        
        event_time_image /= torch.max(event_time_image)

        # print(event_time_images)
        # print("min:", torch.min(event_time_image).item(), torch.max(event_time_image).item())
        # raise
        event_time_image = event_time_image.permute(2,0,1)

        '''
        if self._count_only:
            event_image = event_count_image
        elif self._time_only:
            event_image = event_time_image
        else:
            event_image = torch.cat([event_count_image, event_time_image], dim=2)

        event_image = event_image.permute(2,0,1).type(torch.float32)
        '''

        return event_count_image, event_time_image

    def read_file_paths(self,
                        data_folder_path,
                        split,
                        sequence=None):
        """
        return: event_data_paths,paths of event data (left and right in one folder is two)
        n_ima: the sum number of event pictures in every path and the paths before
        """
        event_data_paths = []
        n_ima = 0
        if sequence is None:
            bag_list_file = open(os.path.join(data_folder_path, "{}_bags.txt".format(split)), 'r')
            lines = bag_list_file.read().splitlines()
            bag_list_file.close()
        else:
            if isinstance(sequence, (list, )):
                lines = sequence
            else:
                lines = [sequence]
        
        n_ima = [0]
        for line in lines:
            bag_name = line

            event_data_paths.append(os.path.join(data_folder_path,bag_name))
            num_ima_file = open(os.path.join(data_folder_path, bag_name, 'n_images.txt'), 'r')
            num_imas = num_ima_file.read()
            num_ima_file.close()
            num_imas_split = num_imas.split(' ')
            n_left_ima = int(num_imas_split[0]) - _MAX_SKIP_FRAMES
            n_ima.append(n_left_ima + n_ima[-1])
            
            n_right_ima = int(num_imas_split[1]) - _MAX_SKIP_FRAMES
            if n_right_ima > 0 and not split is 'test':
                n_ima.append(n_right_ima + n_ima[-1])
            else:
                n_ima.append(n_ima[-1])
            event_data_paths.append(os.path.join(data_folder_path,bag_name))

        return event_data_paths, n_ima

if __name__ == "__main__":
    data = EventData('/media/cyrilsterling/D/EV-FlowNet-pth/data/mvsec/', 'train')
    EventDataLoader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=True)
    it = 0
    for i in EventDataLoader:
        a = i[0][0].numpy()
        b = i[1][0].numpy()
        c = i[2][0].numpy()
        cv2.namedWindow('a')
        cv2.namedWindow('b')
        cv2.namedWindow('c')
        a = a[2,...]+a[3,...]
        print(np.max(a))
        a = (a-np.min(a))/(np.max(a)-np.min(a))
        b = np.transpose(b,(1,2,0))
        c = np.transpose(c,(1,2,0))
        cv2.imshow('a',a)
        cv2.imshow('b',b)
        cv2.imshow('c',c)
        cv2.waitKey(1)
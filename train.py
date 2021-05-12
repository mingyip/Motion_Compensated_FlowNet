import os
from tqdm import trange
from tqdm import tqdm
import numpy as np
from datetime import datetime
from losses import *

import torch

from config import configs
from data_loader import EventData
from EVFlowNet import EVFlowNet
from dataset import DynamicH5Dataset


from vis_utils import cvshow_all, warp_events_with_flow_torch
# from vis_utils import vis_events_and_flows


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()



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
                    'sliding_window_w': 2500,
                    'sliding_window_t': 0.1}


    # EventDataset = EventData(args.data_path, 'train')
    # EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)

    # h5Dataset = DynamicH5Dataset('data/office.h5', voxel_method=voxel_method)
    h5Dataset = DynamicH5Dataset('data/outdoor_day1_data.h5', voxel_method=voxel_method)
    # h5Dataset = DynamicH5Dataset('data/office_spiral.h5', voxel_method=voxel_method)
    # h5Dataset = DynamicH5Dataset('data/indoor_flying1_data.h5', voxel_method=voxel_method)
    h5DataLoader = torch.utils.data.DataLoader(dataset=h5Dataset, batch_size=6, num_workers=6, shuffle=True)

    # model
    EVFlowNet_model = EVFlowNet(args).to(device)
    # EVFlowNet_model.load_state_dict(torch.load('data/saver/evflownet_0906_041812_outdoor_dataset1/model1'))

    # optimizer
    optimizer = torch.optim.Adam(EVFlowNet_model.parameters(), lr=args.initial_learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    loss_fun = TotalLoss(args.smoothness_weight, args.photometric_loss_weight)

    EVFlowNet_model.train()
    for epoch in range(100):
        total_loss = 0.0
        running_loss = 0.0
        print(f'****************** epoch: {epoch} ******************')

        for iteration, item in enumerate(tqdm(h5DataLoader)):

            voxel = item['voxel'].to(device)
            events = item['events'].to(device)
            frame = item['frame'].to(device)
            frame_ = item['frame_'].to(device)
            num_events = item['num_events'].to(device)

            optimizer.zero_grad()
            flow_dict = EVFlowNet_model(voxel)
            loss, ev_loss, smooth_loss, ph_loss = loss_fun(flow_dict, events, frame, frame_, num_events, EVFlowNet_model)

            if iteration % log_interval == 0:
                print(f'iteration: {iteration} avg loss: {running_loss//log_interval} event loss: {int(ev_loss)} smooth loss: {int(smooth_loss)}, photo loss: {int(ph_loss)}')
                running_loss = 0.0
                # sensor_size = (176, 240)
                sensor_size = (256, 336)
                image_name="results/img_{:03}_{:07d}.png".format(epoch, iteration)

                events_vis = events[0].detach().cpu()
                flow_vis = flow_dict["flow3"][0].detach().cpu()

                # Compose the event image and warp the event image with flow
                ev_img, ev_img_ = warp_events_with_flow_torch(events_vis, flow_vis, sensor_size)

                # Convert to numpy format
                ev_img = torch_to_numpy(ev_img)
                ev_img_ = torch_to_numpy(ev_img_)
                frame_vis = torch_to_numpy(item['frame'][0])
                frame_vis_ = torch_to_numpy(item['frame_'][0])
                flow_vis = torch_to_numpy(flow_dict["flow3"][0])

                cvshow_all(ev_img, flow_vis, frame_vis, frame_vis_, ev_img_, image_name, sensor_size)

            if iteration % 1000 == 999:
                print("scheduler.step()")
                scheduler.step()
                torch.save(EVFlowNet_model.state_dict(), args.load_path+'/model%d'%epoch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            running_loss += loss.item()

        if epoch % 4 == 3:
            print("scheduler.step()")
            scheduler.step()

        torch.save(EVFlowNet_model.state_dict(), args.load_path+'/model%d'%epoch)
        print(f'Epoch {epoch} - Avg loss: {total_loss / len(h5DataLoader)}')


    

if __name__ == "__main__":
    main()

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


from vis_utils import vis_events_and_flows

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

    # EventDataset = EventData(args.data_path, 'train')
    # EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)

    h5Dataset = DynamicH5Dataset('data/office.h5')
    # h5Dataset = DynamicH5Dataset('data/outdoor_day1_data.h5', imsize=(256, 336))
    # h5Dataset = DynamicH5Dataset('data/outdoor_day2_data.h5', imsize=(256, 346))
    h5DataLoader = torch.utils.data.DataLoader(dataset=h5Dataset, batch_size=10, num_workers=6, shuffle=True)

    # model
    EVFlowNet_model = EVFlowNet(args).to(device)

    # optimizer
    optimizer = torch.optim.Adam(EVFlowNet_model.parameters(), lr=args.initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    loss_fun = TotalLoss(args.smoothness_weight)




    EVFlowNet_model.train()
    for epoch in range(100):
        total_loss = 0.0
        running_loss = 0.0
        print(f'****************** epoch: {epoch} ******************')

        for iteration, item in enumerate(tqdm(h5DataLoader)):

            voxel = item['voxel'].to(device)
            events = item['events'].to(device)

            optimizer.zero_grad()
            flow_dict = EVFlowNet_model(voxel)
            loss, ev_loss, smooth_loss = loss_fun(flow_dict, events, EVFlowNet_model)

            if iteration % log_interval == 0:
                print(f'iteration: {iteration} avg loss: {running_loss//log_interval} event loss: {int(ev_loss)} smooth loss: {int(smooth_loss)}')
                running_loss = 0.0

                flow_vis = flow_dict["flow3"].clone().detach()[0].unsqueeze(0)
                voxel_vis = np.sum(voxel.cpu().numpy().squeeze(), axis=0)
                events_vis = events[0].clone().detach().unsqueeze(0)

                vis_events_and_flows(voxel_vis, events_vis, flow_vis, 
                                sensor_size=flow_vis.shape[-2:],
                                image_name="results/img_{:03}_{:07d}.png".format(epoch, iteration))

            # if iteration % 100 == 99:
            #     print("scheduler.step()")
            #     scheduler.step()

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

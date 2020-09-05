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

from event_utils import cvshow_voxel_grid, cvshow_all, events_to_image_torch

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

    xs, ys, ts, ps = events
    # crop = CropParameters(sensor_size[0], sensor_size[1], 4)

    img, img_ = warp_events_with_flow_torch((xs, ys, ts, ps), flow, sensor_size=sensor_size)
    # img = crop.pad(img).cpu().numpy()
    # img_ = crop.pad(img_).cpu().numpy()

    img = img.cpu().numpy()
    img_ = img_.cpu().numpy()

    # print(np.max(voxel), np.min(voxel), np.max(img), np.min(img), np.max(img_), np.min(img_))

    # print(voxel.shape)
    cvshow_all(voxel=img, flow=flow[0].cpu().numpy(), frame=None, compensated=img_, image_name=image_name)


def main():
    args = configs()

    if args.training_instance:
        args.load_path = os.path.join(args.load_path, args.training_instance)
    else:
        args.load_path = os.path.join(args.load_path,
                                      "evflownet_{}".format(datetime.now()
                                                            .strftime("%m%d_%H%M%S")))
    if not os.path.exists(args.load_path):
        os.makedirs(args.load_path)

    # EventDataset = EventData(args.data_path, 'train')
    # EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)

    h5Dataset = DynamicH5Dataset('data/office.h5')
    # h5Dataset = DynamicH5Dataset('data/outdoor_day1_data.h5')
    # h5Dataset = DynamicH5Dataset('data/outdoor_day2_data.h5')
    h5DataLoader = torch.utils.data.DataLoader(dataset=h5Dataset, batch_size=6, num_workers=6, shuffle=True)


    # for event_image, prev_image, next_image, _ in tqdm(EventDataLoader):

    #     print(event_image.shape, event_image.dtype, event_image.device)
    #     print(torch.max(event_image[:, 0:2]), torch.min(event_image[:, 0:2]))
    #     print(torch.max(event_image[:, 2:4]), torch.min(event_image[:, 2:4]))


    #     print(prev_image.shape, prev_image.dtype, prev_image.device, torch.max(prev_image), torch.min(prev_image))
    #     print(next_image.shape, next_image.dtype, next_image.device, torch.max(next_image), torch.min(next_image))

    # #     # torch.Size([16, 4, 256, 256]) torch.float32 cpu
    # #     # tensor(0.0902) tensor(0.)
    # #     # tensor(255.) tensor(0.)
    # #     # torch.Size([16, 1, 256, 256]) torch.float32 cpu tensor(1.) tensor(0.)
    # #     # torch.Size([16, 1, 256, 256]) torch.float32 cpu tensor(1.) tensor(0.)

    #     raise
    #     continue
        # print("EventDataLoader")



    # for event_image, prev_image, next_image in tqdm(h5DataLoader):
    #     print("h5DataLoader")

    #     print(event_image.shape, event_image.dtype, event_image.device)
    #     print(torch.max(event_image[:, 0:2]), torch.min(event_image[:, 0:2]))
    #     print(torch.max(event_image[:, 2:4]), torch.min(event_image[:, 2:4]))

    #     print(prev_image.shape, prev_image.dtype, prev_image.device, torch.max(prev_image), torch.min(prev_image))
    #     print(next_image.shape, next_image.dtype, next_image.device, torch.max(next_image), torch.min(next_image))

    #     raise
    # raise


    # model
    EVFlowNet_model = EVFlowNet(args).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #para = np.load('D://p.npy', allow_pickle=True).item()
    #EVFlowNet_model.load_state_dict(para)
    # EVFlowNet_model.load_state_dict(torch.load(args.load_path+'/../model'))

    #EVFlowNet_parallelmodel = torch.nn.DataParallel(EVFlowNet_model)
    # optimizer
    optimizer = torch.optim.Adam(EVFlowNet_model.parameters(), lr=args.initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    loss_fun = TotalLoss(args.smoothness_weight)

    iteration = 0
    size = 0
    EVFlowNet_model.train()
    for epoch in range(100):
        loss_sum = 0.0
        print('*****************************************')
        print('epoch:'+str(epoch))
        for iteration, item in enumerate(tqdm(h5DataLoader)):

            # print(item)
            voxel = item['voxel'].to(device)

            xs, ys, ts, ps = item['events']

            xs = xs.to(device)
            ys = ys.to(device)
            ts = ts.to(device)
            ps = ps.to(device)

            if iteration % 10 == 0:
                xs_ = xs[0].clone().detach().unsqueeze(0)
                ys_ = ys[0].clone().detach().unsqueeze(0)
                ts_ = ts[0].clone().detach().unsqueeze(0)
                ps_ = ps[0].clone().detach().unsqueeze(0)

            optimizer.zero_grad()
            flow_dict = EVFlowNet_model(voxel)

            loss = loss_fun(flow_dict, (xs, ys, ts, ps), EVFlowNet_model)

            if iteration % 10 == 0:
                print('iteration:', iteration)
                print('loss:', loss_sum/10)
                loss_sum = 0.0

                flow = flow_dict["flow3"].clone().detach()
                flow = flow[0].unsqueeze(0)
                # print(torch.max(flow), torch.min(flow))
                # print(flow.shape)
                
                # flow = item['flow'][0:1].to(device)


                voxel_ = voxel.cpu().numpy().squeeze()
                voxel_ = np.sum(voxel_, axis=0)

                vis_events_and_flows(voxel_,
                                (xs_, ys_, ts_, ps_), 
                                flow, 
                                sensor_size=flow.shape[-2:],
                                image_name="results/img{:07d}.png".format(epoch * 10000 + iteration))

            # if iteration %1000 == 999:
            #     print("scheduler.step()")
                
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            iteration += 1
            size += 1

        
        scheduler.step()
        torch.save(EVFlowNet_model.state_dict(), args.load_path+'/model%d'%epoch)
        
        print('iteration:', iteration)
        print('loss:', loss_sum/size)
        size = 0


    

if __name__ == "__main__":
    main()

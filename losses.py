import torch
import torchvision.transforms.functional as F
import cv2
import numpy as np

def warp_images_with_flow(images, flow):
    """
    Generates a prediction of an image given the optical flow, as in Spatial Transformer Networks.
    """
    dim3 = 0
    if images.dim() == 3:
        dim3 = 1
        images = images.unsqueeze(0)
        flow = flow.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]
    flow_x, flow_y = flow[:,0,...],flow[:,1,...]
    coord_y, coord_x = torch.meshgrid(torch.arange(height), torch.arange(width))

    pos_x = coord_x.type(torch.float32).cuda().unsqueeze(0) - flow_x
    pos_y = coord_y.type(torch.float32).cuda().unsqueeze(0) - flow_y
    pos_x = (pos_x-(width-1)/2)/((width-1)/2)
    pos_y = (pos_y-(height-1)/2)/((height-1)/2)

    pos = torch.stack((pos_x,pos_y),3).type(torch.float32)
    result = torch.nn.functional.grid_sample(images, pos, mode='bilinear', padding_mode='zeros')
    if dim3 == 1:
        result = result.squeeze()
        
    return result


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3, calc_mean=True):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """

    if calc_mean:
        loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    else:
        loss = torch.sum(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


def compute_smoothness_loss(flow, is_8connected=True):
    """
    Local smoothness loss, as defined in equation (5) of the paper.
    The neighborhood here is defined as the 8-connected region around each pixel.
    """

    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]
    
    if is_8connected:
        smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                        charbonnier_loss(flow_ucrop - flow_dcrop) + \
                        charbonnier_loss(flow_ulcrop - flow_drcrop) + \
                        charbonnier_loss(flow_dlcrop - flow_urcrop)
        # smoothness_loss /= 4.
    else:
        smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                        charbonnier_loss(flow_ucrop - flow_dcrop)
    
    return smoothness_loss



def compute_photometric_loss(prev_images, next_images, flow_dict):
    """
    Multi-scale photometric loss, as defined in equation (3) of the paper.
    """
    total_photometric_loss = 0.
    loss_weight_sum = 0.
    for i in range(len(flow_dict)):
        for image_num in range(prev_images.shape[0]):
            flow = flow_dict["flow{}".format(i)][image_num]
            height = flow.shape[1]
            width = flow.shape[2]

            prev_images_resize = F.to_tensor(F.resize(F.to_pil_image(prev_images[image_num].cpu()),
                                                    [height, width])).cuda()
            next_images_resize = F.to_tensor(F.resize(F.to_pil_image(next_images[image_num].cpu()),
                                                    [height, width])).cuda()

            prev_images_warped = warp_images_with_flow(prev_images_resize, flow)
            distance = prev_images_warped - next_images_resize

            if i == 3 and image_num == 0:
                vis_warp = prev_images_warped.clone().detach().cpu().numpy().squeeze()
                vis_prev = prev_images_resize.clone().detach().cpu().numpy().squeeze()
                vis_next = next_images_resize.clone().detach().cpu().numpy().squeeze()
                vis_dist = distance.clone().detach().cpu().numpy().squeeze()
                photometric_vis = [vis_warp, vis_prev, vis_next, vis_dist]
            #     img = np.hstack([vis_warp, vis_prev, vis_next, vis_dist])
            #     cv2.imshow("warp image", img)
            #     cv2.waitKey(1)

            
            photometric_loss = charbonnier_loss(distance, calc_mean=False)
            total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum
    return total_photometric_loss, photometric_vis


def compute_event_flow_loss(events, num_events, flow_dict):

    # TODO: move device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xs = events[:, 0] 
    ys = events[:, 1] 
    ts = events[:, 2] 
    ps = events[:, 3] 

    loss_weight_sum = 0.
    total_event_loss = 0.

    eps = torch.finfo(xs.dtype).eps
    for batch_idx, (x, y, t, p, n) in enumerate(zip(xs, ys, ts, ps, num_events)):

        x = x[:n]
        y = y[:n]
        t = t[:n]
        p = p[:n]

        for flow_idx in range(len(flow_dict)):
            flow = flow_dict["flow{}".format(flow_idx)][batch_idx]

            neg_mask = p == -1
            pos_mask = p == 1
            t = (t - t[0]) / (t[-1] - t[0] + eps)

            # Resize the event image to match the flow dimension
            x_ = x / 2**(3-flow_idx)
            y_ = y / 2**(3-flow_idx)

            # Positive events
            xp = x_[pos_mask].to(device).type(torch.long)
            yp = y_[pos_mask].to(device).type(torch.long)
            tp = t[pos_mask].to(device).type(torch.float)

            # Negative events
            xn = x_[neg_mask].to(device).type(torch.long)
            yn = y_[neg_mask].to(device).type(torch.long)
            tn = t[neg_mask].to(device).type(torch.float)

            # # Timestamp for {Forward, Backward} x {Postive, Negative}
            # t_fp = tp[-1] - tp   # t[-1] should be 1
            # t_bp = tp[0]  - tp   # t[0] should be 0
            # t_fn = tn[-1] - tn
            # t_bn = tn[0]  - tn

            # fp_loss = event_loss((xp, yp, t_fp), flow)
            # bp_loss = event_loss((xp, yp, t_bp), flow)
            # fn_loss = event_loss((xn, yn, t_fn), flow)
            # bn_loss = event_loss((xn, yn, t_bn), flow)

            fp_loss = event_loss((xp, yp, tp), flow, forward=True)
            bp_loss = event_loss((xp, yp, tp), flow, forward=False)
            fn_loss = event_loss((xn, yn, tn), flow, forward=True)
            bn_loss = event_loss((xn, yn, tn), flow, forward=False)

            loss_weight_sum += 4
            total_event_loss += fp_loss + bp_loss + fn_loss + bn_loss
            # total_event_loss += fp_loss + bp_loss

    # total_event_loss /= loss_weight_sum
    return total_event_loss


def event_loss(events, flow, forward=True):

    eps = torch.finfo(flow.dtype).eps
    H, W = flow.shape[1:]
    # x, y, t = events_average(events, (H, W))
    x, y, t = events

    # Estimate events position after flow
    if forward:
        t_ = t[-1] - t + eps
    else:
        t_ = t[0] - t - eps

    x_ = torch.clamp(x + t_ * flow[0,y,x], min=0, max=W-1)
    y_ = torch.clamp(y + t_ * flow[1,y,x], min=0, max=H-1)

    x0 = torch.floor(x_)
    x1 = torch.ceil(x_)
    y0 = torch.floor(y_)
    y1 = torch.ceil(y_)

    # Interpolation ratio
    x0_ratio = 1 - (x_ - x0)
    x1_ratio = 1 - (x1 - x_)
    y0_ratio = 1 - (y_ - y0)
    y1_ratio = 1 - (y1 - y_)

    Ra = x0_ratio * y0_ratio
    Rb = x1_ratio * y0_ratio
    Rc = x0_ratio * y1_ratio
    Rd = x1_ratio * y1_ratio

    # R_sum = Ra + Rb + Rc + Rd
    # Ra /= R_sum
    # Rb /= R_sum
    # Rc /= R_sum
    # Rd /= R_sum

    # Prevent R and T to be zero
    Ra = Ra+eps; Rb = Rb+eps; Rc = Rc+eps; Rd = Rd+eps

    Ta = Ra * t_
    Tb = Rb * t_
    Tc = Rc * t_
    Td = Rd * t_

    # Ta = Ta+eps; Tb = Tb+eps; Tc = Tc+eps; Td = Td+eps

    # Calculate interpolation flatterned index of 4 corners for all events
    Ia = (x0 + y0 * W).type(torch.long)
    Ib = (x1 + y0 * W).type(torch.long)
    Ic = (x0 + y1 * W).type(torch.long)
    Id = (x1 + y1 * W).type(torch.long)

    # Compute the nominator and denominator
    numerator = torch.zeros((W*H), dtype=flow.dtype, device=flow.device)
    denominator = torch.zeros((W*H), dtype=flow.dtype, device=flow.device)

    # denominator.index_add_(0, Ia, Ra)
    # denominator.index_add_(0, Ib, Rb)
    # denominator.index_add_(0, Ic, Rc)
    # denominator.index_add_(0, Id, Rd)

    denominator.index_add_(0, Ia, torch.ones_like(Ra))
    denominator.index_add_(0, Ib, torch.ones_like(Rb))
    denominator.index_add_(0, Ic, torch.ones_like(Rc))
    denominator.index_add_(0, Id, torch.ones_like(Rd))

    numerator.index_add_(0, Ia, Ta)
    numerator.index_add_(0, Ib, Tb)
    numerator.index_add_(0, Ic, Tc)
    numerator.index_add_(0, Id, Td)

    loss = (numerator / (denominator + eps)) ** 2
    return loss.sum()


def events_test_loss(flow_dict):

    """
    This loss func is created for testing only.
    Test if the model can learn a constant square flow
    """

    f0 = flow_dict['flow0'].clone()
    f1 = flow_dict['flow1'].clone()
    f2 = flow_dict['flow2'].clone()
    f3 = flow_dict['flow3'].clone()

    f0[:, 0, 0:20, 0:20] -= 2.5
    f1[:, 0, 0:40, 0:40] -= 5
    f2[:, 0, 0:80, 0:80] -= 10
    f3[:, 0, 0:160, 0:160] += 20

    loss = torch.mean(f0 ** 2) + \
            torch.mean(f1 ** 2) + \
            torch.mean(f2 ** 2) + \
            torch.mean(f3 ** 2)

    return loss


class TotalLoss(torch.nn.Module):
    def __init__(self, smoothness_weight, photometric_loss_weight):
        super(TotalLoss, self).__init__()
        self._smoothness_weight = smoothness_weight
        self._photometric_loss_weight = photometric_loss_weight

    def forward(self, flow_dict, events, frame, frame_, num_events, EVFlowNet_model):

        # smoothness loss
        smoothness_loss = 0
        for i in range(len(flow_dict)):
            smoothness_loss += compute_smoothness_loss(flow_dict["flow{}".format(i)])
        smoothness_loss *= self._smoothness_weight

        # Photometric loss.
        # photometric_loss, photometric_vis = compute_photometric_loss(frame,  frame_, flow_dict)
        # photometric_loss *= self._photometric_loss_weight

        # Event compensation loss.
        event_loss = compute_event_flow_loss(events, num_events, flow_dict)

        loss = event_loss + smoothness_loss
        return loss, event_loss.item(), smoothness_loss.item(), 0


if __name__ == "__main__":
    '''
    a = torch.rand(7,7)
    b = torch.rand(7,7)
    flow = {}
    flow['flow0'] = torch.rand(1,2,3,3)
    loss = compute_photometric_loss(a,b,0,flow)
    print(loss)
    '''
    a = torch.rand(1,5,5).cuda()
    #b = torch.rand(5,5)*5
    b = torch.rand((5,5)).type(torch.float32).cuda()
    b.requires_grad = True
    #c = torch.rand(5,5)*5
    c = torch.rand((5,5)).type(torch.float32).cuda()
    c.requires_grad = True
    d = torch.stack((b,c),0)
    print(a)
    print(b)
    print(c)
    r = warp_images_with_flow(a,d)
    print(r)
    r = torch.mean(r)
    r.backward()
    print(b.grad)
    print(c.grad)
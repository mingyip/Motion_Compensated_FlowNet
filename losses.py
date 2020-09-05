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
    flow_x,flow_y = flow[:,0,...],flow[:,1,...]
    coord_x, coord_y = torch.meshgrid(torch.arange(height), torch.arange(width))


    pos_x = coord_x.reshape(height,width).type(torch.float32).cuda() + flow_x
    pos_y = coord_y.reshape(height,width).type(torch.float32).cuda() + flow_y
    pos_x = (pos_x-(height-1)/2)/((height-1)/2)
    pos_y = (pos_y-(width-1)/2)/((width-1)/2)

    pos = torch.stack((pos_y,pos_x),3).type(torch.float32)
    result = torch.nn.functional.grid_sample(images, pos, mode='bilinear', padding_mode='zeros')
    if dim3 == 1:
        result = result.squeeze()
        
    return result


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


def compute_smoothness_loss(flow):
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
    
    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                      charbonnier_loss(flow_ucrop - flow_dcrop) + \
                      charbonnier_loss(flow_ulcrop - flow_drcrop) + \
                      charbonnier_loss(flow_dlcrop - flow_urcrop)
    # smoothness_loss /= 4.
    
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
            
            next_images_warped = warp_images_with_flow(next_images_resize, flow)

            distance = next_images_warped - prev_images_resize
            photometric_loss = charbonnier_loss(distance)
            total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum

    return total_photometric_loss


def compute_event_flow_loss(events, flow_dict):


    # TODO: move device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xs, ys, ts, ps = events
    eps = torch.finfo(xs.dtype).eps

    loss_weight_sum = 0.
    total_event_loss = 0.


    
    for batch_idx, (x, y, t, p) in enumerate(zip(xs, ys, ts, ps)):
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


# def events_average(events, imsize):

#     x, y, t = 



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
        
    # print(torch.max(t_), torch.min(t_))
    # print()

    x_ = torch.clamp(x + t_ * flow[0,y,x], min=0, max=W-1)
    y_ = torch.clamp(y + t_ * flow[1,y,x], min=0, max=H-1)

    x_next = x_.detach().type(torch.long)
    y_next = y_.detach().type(torch.long)

    # if H > 200 and forward:
    #     img = torch.zeros((H, W), device=x_next.device, dtype=torch.long)
    #     img.index_put_((y_next,   x_next), torch.ones_like(x_next, dtype=torch.long), accumulate=True)
    #     img = img.cpu().numpy().astype(np.int8)
    #     # print(img.shape, img.dtype)

    #     print(img[10:20])
    #     cv2.imshow("test_window", img * 10000)
    #     cv2.waitKey(1)


    x0 = torch.floor(x_)
    x1 = x0 + 1
    y0 = torch.floor(y_)
    y1 = y0 + 1

    # Interpolation ratio
    x0_ = x_ - x0
    x1_ = x1 - x_
    y0_ = y_ - y0
    y1_ = y1 - y_

    Ra = x0_ * y0_
    Rb = x1_ * y0_
    Rc = x0_ * y1_
    Rd = x1_ * y1_

    # Prevent R and T to be zero
    Ra = Ra+eps; Rb = Rb+eps; Rc = Rc+eps; Rd = Rd+eps

    Ta = Ra * t_
    Tb = Rb * t_
    Tc = Rc * t_
    Td = Rd * t_

    # Ta = Ta+eps; Tb = Tb+eps; Tc = Tc+eps; Td = Td+eps

    # Calculate interpolation flatterned index of 4 corners for all events
    x1_idx = torch.clamp(x1, max=W-1)
    y1_idx = torch.clamp(y1, max=H-1)
    x0_idx = x0
    y0_idx = y0

    Ia = (x1_idx + y1_idx * W).type(torch.long)
    Ib = (x0_idx + y1_idx * W).type(torch.long)
    Ic = (x1_idx + y0_idx * W).type(torch.long)
    Id = (x0_idx + y0_idx * W).type(torch.long)

    # Compute the nominator and denominator
    numerator = torch.zeros((W*H), dtype=flow.dtype, device=flow.device)
    denominator = torch.zeros((W*H), dtype=flow.dtype, device=flow.device)

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

    f0 = flow_dict['flow0'].clone()
    f1 = flow_dict['flow1'].clone()
    f2 = flow_dict['flow2'].clone()
    f3 = flow_dict['flow3'].clone()

    f0[:, :, 0:20, 0:20] += 20
    f1[:, :, 0:40, 0:40] += 40
    f2[:, :, 0:80, 0:80] += 80
    f3[:, :, 0:160, 0:160] += 160

    loss = torch.mean(f0 ** 2) + \
            torch.mean(f1 ** 2) + \
            torch.mean(f2 ** 2) + \
            torch.mean(f3 ** 2)

    return loss



class TotalLoss(torch.nn.Module):
    def __init__(self, smoothness_weight, weight_decay_weight=1e-4):
        super(TotalLoss, self).__init__()
        self._smoothness_weight = smoothness_weight
        self._weight_decay_weight = weight_decay_weight

    def forward(self, flow_dict, events, EVFlowNet_model):

        # weight decay loss
        # weight_decay_loss = 0
        # for i in EVFlowNet_model.parameters():
        #     weight_decay_loss += torch.sum(i**2)/2*self._weight_decay_weight / 10

        # smoothness loss
        smoothness_loss = 0
        for i in range(len(flow_dict)):
            smoothness_loss += compute_smoothness_loss(flow_dict["flow{}".format(i)])
        smoothness_loss *= self._smoothness_weight

        # # Photometric loss.
        # photometric_loss = compute_photometric_loss(prev_image,
        #                                             next_image,
        #                                             flow_dict)

        # Event compensation loss.
        event_loss = compute_event_flow_loss(events, flow_dict)

        # print(event_loss.item(), weight_decay_loss.item(), smoothness_loss.item(), self._smoothness_weight)
        print(event_loss.item(), smoothness_loss.item())
        loss = event_loss + smoothness_loss
        # loss = event_loss + weight_decay_loss


        return loss

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
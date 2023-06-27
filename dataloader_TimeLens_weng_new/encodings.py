import numpy as np
import torch


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param padding if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)
    return img


def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search sorted pytorch tensor
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        if t[l] == x:
            return l
        if t[r] == x:
            return r
            
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    if len(ts) < 3:
        return torch.zeros([B] + list(sensor_size))
    bins = []
    dt = ts[-1]-ts[0] + 1e-6
    delta_t = dt / B
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = ts[0] + delta_t*bi
            tend = tstart + delta_t
            beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
            end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


def events_to_image(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into an image.
    xs, ys, ps: torch.tensor, [N]
    """
    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img
    

def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    """
    Generate a two-channel event image containing event counters.
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])
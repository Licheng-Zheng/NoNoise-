# I got the code from here: https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/__init__.py
# Using this because idk how to implement it, and using it as a proof of concept for the evaluation pipeline


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def _to_tensor_3d_hsi(x):
    """Convert numpy (H,W,B) or torch (H,W,B) to torch (1,1,B,H,W) float tensor."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif torch.is_tensor(x):
        t = x
    else:
        raise TypeError("Input must be numpy.ndarray or torch.Tensor")

    if t.ndim != 3:
        raise ValueError(f"Expected 3D array/tensor (H,W,B). Got shape {tuple(t.shape)}")

    # Expect (H, W, B). If it's (B,H,W) try to detect and permute.
    H, W, B = t.shape
    # Heuristic: if first dim is very small and third is large, assume (B,H,W)
    if B <= 8 and H > 8 and W > 8:
        # Likely already (H,W,B)
        pass
    elif H <= 8 and B > 8:
        # Likely (B,H,W)
        t = t.permute(1, 2, 0)
        H, W, B = t.shape

    t = t.to(torch.float32)
    # (H,W,B) -> (B,H,W) -> (1,1,B,H,W)
    t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    return t


def ssim_hsi(img1, img2, window_size: int = 11, normalize: bool = False, device: str | None = None) -> float:
    """
    Compute SSIM for hyperspectral images (3D cubes) of arbitrary size.

    Parameters:
        img1, img2: numpy arrays or torch tensors with shape (H, W, B) or (B, H, W).
        window_size: Odd integer size for the gaussian window (default 11).
        normalize: If True, scales inputs independently to [0,1] based on their min/max.
        device: Optional torch device string (e.g., 'cuda', 'cpu'). If None, uses img1 device or CPU.

    Returns:
        Scalar SSIM value as Python float.
    """
    t1 = _to_tensor_3d_hsi(img1)
    t2 = _to_tensor_3d_hsi(img2)

    if t1.shape != t2.shape:
        raise ValueError(f"Input shapes must match after conversion. Got {tuple(t1.shape)} vs {tuple(t2.shape)}")

    if normalize:
        def norm01(t):
            tmin = t.amin()
            tmax = t.amax()
            return (t - tmin) / (tmax - tmin + 1e-12)
        t1 = norm01(t1)
        t2 = norm01(t2)

    if device is not None:
        t1 = t1.to(device)
        t2 = t2.to(device)

    val = ssim3D(t1, t2, window_size=window_size, size_average=True)
    return float(val.detach().cpu().item())

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
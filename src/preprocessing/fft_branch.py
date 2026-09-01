import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            is_video = True
        else:
            is_video = False
            B = x.size(0)

        gray_weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
        x_gray = (x * gray_weights).sum(dim=1, keepdim=True)

        x_fft = torch.fft.fft2(x_gray.float())
        x_fft = torch.fft.fftshift(x_fft)

        x_real = x_fft.real
        x_imag = x_fft.imag

        x_out = torch.cat([x_real, x_imag], dim=1)

        if is_video:
            x_out = x_out.view(B, T, 2, H, W)

        return x_out
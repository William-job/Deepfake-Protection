import torch
import torch.nn as nn
from src.preprocessing.rgb_branch import RGBBranch
from src.preprocessing.fft_branch import FFTBranch
from src.preprocessing.residual_branch import ResidualBranch


class MultiScalePreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = RGBBranch()
        self.fft_branch = FFTBranch()
        self.residual_branch = ResidualBranch()

    def forward(self, x):
        x_rgb = self.rgb_branch(x)
        x_fft = self.fft_branch(x)
        x_res = self.residual_branch(x)

        return {
            "rgb": x_rgb,
            "fft": x_fft,
            "residual": x_res,
        }
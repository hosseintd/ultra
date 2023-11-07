import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO! self.residuals = nn.ModuleList([ResidualBlock(in_channels, mid_channels) for _ in range(n)])

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class CBSModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,d=1):
        super(CBSModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=autopad(kernel_size, padding, d))
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.silu(x)  # SiLU activation
        return x

class CSP1_X(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super(CSP1_X, self).__init__()
        self.branch1 = nn.Sequential(
            CBSModule(in_channels, out_channels),
            *[ResidualBlock(out_channels, out_channels) for _ in range(n)],
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        return torch.cat([branch1_output, branch2_output], dim=1)

class CSP2_X(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super(CSP1_X, self).__init__()
        self.branch1 = nn.Sequential(
            CBSModule(in_channels, out_channels),
            *[ResidualBlock(out_channels, out_channels) for _ in range(2*n)],
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.ac = nn.LeakyReLU(out_channels)
        self.cbs = 
    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        concated = torch.cat([branch1_output, branch2_output], dim=1)
        out = 
        return 


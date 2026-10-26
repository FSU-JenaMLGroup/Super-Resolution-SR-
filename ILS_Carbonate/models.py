import torch.nn as nn
from torch import Tensor
#==================================================================================================================================================================================
"""
SECTION: Build Model
"""
# ****************************************************** Function: This Class Defines Structure of FSRCNN Model. *******************************************************
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=1//2,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=1//2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    padding=1//2,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        return x
# ********************************************************************************************************************************
class ResidualBlock2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=3//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=3//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=(3, 3), padding=3//2, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        return x
# ********************************************************************************************************************************
class FSRCNN_RB(nn.Module):
    def __init__(self, in_channels: int = 1, d: int=56, s: int=20, scale_factor: int=4):
        super().__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=d,kernel_size=(1,3),padding=(0,1),groups=1,bias=False), nn.PReLU(d))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=d,kernel_size=(3,1),padding=(1,0),groups=1,bias=False), nn.PReLU(d))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=d,kernel_size=(3,3),padding=(1,1),groups=1,bias=False), nn.PReLU(d))
        self.layer1 = nn.Sequential(ResidualBlock(d, s), nn.PReLU(s),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer2 = nn.Sequential(ResidualBlock2(s, s), nn.PReLU(s),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer3 = nn.Sequential(ResidualBlock2(s, s), nn.PReLU(s),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer4 = nn.Sequential(ResidualBlock2(s, s), nn.PReLU(s),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer5 = nn.Sequential(ResidualBlock2(s, s), nn.PReLU(s),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer6 = nn.Sequential(ResidualBlock2(s, s), nn.PReLU(s),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer7 = nn.Sequential(ResidualBlock2(s, s), nn.PReLU(s),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer8 = nn.Sequential(ResidualBlock(s, d), nn.PReLU(d),
            # nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=d, kernel_size=(3, 3), padding=3//2, bias=False),
        # # nn.BatchNorm2d(d),
        # nn.PReLU(d)
        # )
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=32, kernel_size=(3, 3), padding=3//2, bias=False),
            nn.PixelShuffle(4),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=3//2, bias=False),
        # nn.BatchNorm2d(d),
        nn.PReLU(d)
        )
        self.layer9 = nn.Conv2d(in_channels=d, out_channels=32, kernel_size=(3, 3), padding=3//2, bias=False)
        self.layer10 = nn.PixelShuffle(4)
        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding=3//2, bias=False),
        # nn.BatchNorm2d(d),
        nn.ReLU())

        # self.last_layer = nn.ConvTranspose2d(d, in_channels, kernel_size=9, stride=scale_factor, padding=9 // 2,
        #                                     output_padding=scale_factor - 1)
        self.ReLU=nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x1_3 = self.conv1_3(x)
        x1 = x1_1+x1_2+x1_3
        x1_tcw = self.ReLU(x1)
        x2_tcw = self.layer1(x1_tcw)
        x3_tcw = self.layer2(x2_tcw)
        x4_tcw = self.layer3(x3_tcw)
        x5_tcw = self.layer4(x4_tcw)
        x6_tcw = self.layer5(x5_tcw)
        x7_tcw = self.layer6(x6_tcw)
        x8_tcw = self.layer7(x7_tcw)
        x8_tcw = x2_tcw + x8_tcw
        x8_tcw = self.ReLU(x8_tcw)
        x9_tcw = self.layer8(x8_tcw)
        # x10_tcw = self.last_layer(x9_tcw)
        x10_tcw = self.layer9(x9_tcw)
        x11_tcw = self.layer10(x10_tcw)
        x12_tcw = self.layer11(x11_tcw)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.last_layer(x)

        return x12_tcw
#==================================================================================================================================================================================
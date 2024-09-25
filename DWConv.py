import torch
from torch import nn


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


def forward(self, input):
    out = self.depth_conv(input)
    out = self.point_conv(out)
    return out
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DwConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel
        )
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
        # 逐点卷积
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
if __name__ == '__main__':
    x= torch.rand(1,64,128,128)
    dwcon = DepthWiseConv(64,256)
    y= dwcon(x)
    print(y.shape)
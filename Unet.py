####################################
# author:Ankyo Chu
# date:2018.8.24
# update:null
# topic:Unet
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    # (conv->batchnorm->activate func)*2
    # input_channels , output channels
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            # 根据论文，Unet的第一个卷积是64个filter，卷积核是3*3，padding==1
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 暂时没搞懂这个的用途，先占位在这里。
class inconv(nn.Module):
    pass


# 下采样，其实就是一个pool层，pool的大小是2*2，步长为2
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


# 上采样，以前一直不知道怎么处理，看了几份代码才有了一点想法
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    # 这里还是有点不懂，慢慢看吧。
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 这里也一样，好像是输入输出层，等过明天再看看
class outconv(nn.Module):
    pass

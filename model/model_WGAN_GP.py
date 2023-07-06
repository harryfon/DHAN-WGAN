import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from turtle import forward
from unicodedata import name
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_Colorization import ColorizationHAN


class Generator(ColorizationHAN):
    def __init__(self, in_channels=1, out_channels=2, num_classes=365, num_obgects=80, use_global_feature=True, use_seg_feature=True):
        super().__init__(in_channels, out_channels, num_classes, num_obgects, use_global_feature, use_seg_feature)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            Residual(in_channels, 64, strides=2), # 112, 112, 64
            Residual(64, 128, strides=2), # 56, 56, 128
            Residual(128, 256, strides=2), # 28, 28, 256
            nn.Conv2d(256, 512, 3, 1, 1), # 28, 28, 512
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 1) # 28, 28, 1
        )

    def forward(self, x):
        y = self.block(x)     
        return y


class Residual(nn.Module):
    '''No BN'''
    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels, num_channels,
                                kernel_size=1, stride=strides)

    def forward(self, X):
        Y = F.leaky_relu(self.conv1(X), 0.1, inplace=True)
        Y = self.conv2(Y)
        X = self.conv3(X)
        Y += X
        return F.leaky_relu(Y, 0.1, inplace=True)


def main_Gen():
    torch.backends.cudnn.benchmark = True
    img = torch.randn(2, 1, 224, 224, requires_grad=False).cuda()
    model = Generator().cuda()
    with torch.no_grad():
        # class_out, seg_out
        y, else_info = model(img)
    print(y.shape)
    try:
        print(else_info[0].shape)
    except Exception as e:
        print("类信息错误")
    try:
        print(else_info[1].shape)
    except Exception as e:
        print("语义信息错误") 


def main_Disc():
    torch.backends.cudnn.benchmark = True
    img = torch.randn(2, 3, 224, 224, requires_grad=False).cuda()
    netD = Discriminator(3).cuda()
    with torch.no_grad():
        y = netD(img)
    print(y.shape)


if __name__ == '__main__':
    main_Disc()

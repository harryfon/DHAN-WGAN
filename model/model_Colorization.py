import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from d2l import torch as d2l
from model.model_HAN import DynamicDownSamplingBlock, HybridAttentionUpsampling


# ******************************************** #
# *********** 着色任务中使用的网络 *********** #
# ******************************************** #

# 关于黑白图像上色的混合注意力网络
class ColorizationHAN(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, num_classes=365, num_obgects=80, use_global_feature=True, use_seg_feature=True):
        super(ColorizationHAN, self).__init__()
        self.block1 = nn.Sequential(Inception(in_channels, 8, (16, 32), (8, 16), 8))
        self.block2 = nn.Sequential(
            DeeperResidualHybridAttentionBlock(
                in_channels=64, num_classes=num_classes, num_obgects=num_obgects,
                use_global_feature=use_global_feature, use_seg_feature=use_seg_feature))
        self.block3 = nn.Sequential(nn.Conv2d(64, out_channels*4, 1), nn.PReLU(),
                                    nn.Conv2d(out_channels*4, out_channels, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.block1.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_freeze(self, require_grad=True):
        for child in self.block2.children():
            child.set_freeze(require_grad)
    
    def change_seg_model(self, num_classes, device):
        self.block2[0].change_seg_model(num_classes, device)

    def forward(self, x):
        y = self.block1(x)
        all_info = self.block2(y)
        y = self.block3(all_info[0])
        return y, all_info[1:]
    

# 定义一个4层的残差混合注意力块+全局分类网络+语义分割网络
class DeeperResidualHybridAttentionBlock(nn.Module):
    def __init__(self, in_channels=64, num_classes=365, num_obgects=80, use_global_feature=True,  use_seg_feature=True):
        super(DeeperResidualHybridAttentionBlock, self).__init__()
        self.num_obgects = num_obgects
        # 各层通道制定
        layer1_channels = in_channels
        layer2_channels = in_channels*2
        layer3_channels = in_channels*4
        layer4_channels = in_channels*8
        # 着色基本模块
        self.haum1_1 = HybridAttentionUpsampling(layer2_channels, layer1_channels)
        self.haum1_2 = HybridAttentionUpsampling(layer2_channels, layer1_channels)
        self.haum2_1 = HybridAttentionUpsampling(layer3_channels, layer2_channels)
        self.haum2_2 = HybridAttentionUpsampling(layer3_channels, layer2_channels)
        self.ddm1_1 = DynamicDownSamplingBlock(layer1_channels, layer2_channels)
        self.ddm1_2 = DynamicDownSamplingBlock(layer1_channels, layer2_channels)
        self.ddm1_3 = DynamicDownSamplingBlock(layer1_channels, layer2_channels)
        self.ddm2_1 = DynamicDownSamplingBlock(layer2_channels, layer3_channels)
        self.ddm2_2 = DynamicDownSamplingBlock(layer2_channels, layer3_channels)
        self.ddm3_1 = DynamicDownSamplingBlock(layer3_channels, layer4_channels)
        # 类、语义信息相关模块
        if use_global_feature or use_seg_feature:
            self.res1 = nn.Sequential(Residual(layer4_channels, layer4_channels, 2))
            self.res2 = nn.Sequential(Residual(layer4_channels, layer4_channels, 2))
        # 类信息相关模块
        self.use_global_feature = use_global_feature
        if use_global_feature:
            self.class_arch = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
                nn.Linear(layer4_channels, layer4_channels*8), nn.ReLU(), nn.Dropout(0.5)
            )
            self.fc_col = nn.Linear(layer4_channels*8, 256)
            self.fc_class = nn.Linear(layer4_channels*8, num_classes)
            self.haum3_1 = HybridAttentionUpsampling(layer4_channels+256, layer3_channels)
        else:
            self.haum3_1 = HybridAttentionUpsampling(layer4_channels, layer3_channels)
        # 语义信息相关模块
        self.use_seg_feature = use_seg_feature
        if use_seg_feature:
            self.conv_32s = nn.Sequential(nn.Conv2d(layer4_channels, num_obgects, 1), nn.PReLU(), nn.Conv2d(num_obgects, num_obgects, 1))
            self.conv_16s = nn.Sequential(nn.Conv2d(layer4_channels, num_obgects, 1), nn.PReLU(), nn.Conv2d(num_obgects, num_obgects, 1))
            self.conv_4s = nn.Sequential(nn.Conv2d(layer3_channels, num_obgects, 3, 1, 1), nn.PReLU(), nn.Conv2d(num_obgects, num_obgects, 1))
            self.bilinear_32s = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(num_obgects, num_obgects, 1))
            self.bilinear_16s = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.Conv2d(num_obgects, num_obgects, 1))
            self.bilinear_4s = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                nn.Conv2d(num_obgects, num_obgects, 3, 1, 1), nn.PReLU(),
                nn.Conv2d(num_obgects, num_obgects, 1)
            )
            self.bilinear_2s = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(num_obgects, num_obgects, 3, 1, 1), nn.PReLU(),
                nn.Conv2d(num_obgects, num_obgects, 1)
            )
            self.haum1_3 = HybridAttentionUpsampling(layer2_channels+80, layer1_channels)
        else:
            self.haum1_3 = HybridAttentionUpsampling(layer2_channels, layer1_channels)
    
    def set_freeze(self, require_grad=True):
        self.ddm1_1.requires_grad_(require_grad)
        self.ddm2_1.requires_grad_(require_grad)
        self.ddm3_1.requires_grad_(require_grad)
        # seg ***********
        self.ddm1_2.requires_grad_(require_grad)
        self.ddm2_2.requires_grad_(require_grad)
        self.haum1_1.requires_grad_(require_grad)
        self.haum2_1.requires_grad_(require_grad)
        # seg ***********
        self.haum3_1.requires_grad_(require_grad)
        if self.use_global_feature or self.use_seg_feature:
            self.res1.requires_grad_(require_grad)
            self.res2.requires_grad_(require_grad)
        if self.use_global_feature:
            self.class_arch.requires_grad_(require_grad)
            self.fc_class.requires_grad_(require_grad)
            self.fc_col.requires_grad_(require_grad)
        if self.use_seg_feature:
            self.conv_32s.requires_grad_(require_grad)
            self.conv_16s.requires_grad_(require_grad)
            self.conv_4s.requires_grad_(require_grad)
            self.bilinear_32s.requires_grad_(require_grad)
            self.bilinear_16s.requires_grad_(require_grad)
            self.bilinear_4s.requires_grad_(require_grad)
            self.bilinear_2s.requires_grad_(require_grad)

    def change_seg_model(self, num_classes, device):
        self.bilinear_2s = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(self.num_obgects, num_classes, 3, 1, 1), nn.PReLU(),
                nn.Conv2d(num_classes, num_classes, 1)
        ).to(device)

    def forward(self, f1_1):
        f2_1 = self.ddm1_1(f1_1)
        f1_2 = self.haum1_1(f2_1, f1_1)
        f3_1 = self.ddm2_1(f2_1)
        f2_2 = self.haum2_1(f3_1, f2_1)+self.ddm1_2(f1_2)
        f1_3 = self.haum1_2(f2_2, f1_2)
        f4_1 = self.ddm3_1(f3_1)
        if self.use_global_feature or self.use_seg_feature:
            f_res1 = self.res1(f4_1)
            f_res2 = self.res2(f_res1)
        if self.use_global_feature:
            f_class = self.class_arch(f_res2)
            f4_1_gol = self.fc_col(f_class).unsqueeze(-1).unsqueeze(-1)
            f4_1_gol = f4_1_gol.repeat([1, 1, f4_1.size(2), f4_1.size(3)])
            f4_1 = torch.cat([f4_1, f4_1_gol], dim=1)
            f_class = self.fc_class(f_class)
        f3_2 = self.haum3_1(f4_1, f3_1)+self.ddm2_2(f2_2)
        f2_3 = self.haum2_2(f3_2, f2_2)+self.ddm1_3(f1_3)
        if self.use_seg_feature:
            f_32s = self.conv_32s(f_res2)
            f_16s = self.conv_16s(f_res1)
            f_4s = self.conv_4s(f3_2)
            f_16s += self.bilinear_32s(f_32s)
            f_4s += self.bilinear_16s(f_16s)
            f_2s = self.bilinear_4s(f_4s)
            f2_3 = torch.cat([f2_3, f_2s], dim=1)
            f_seg = self.bilinear_2s(f_2s)
        f1_4 = self.haum1_3(f2_3, f1_3)
        # col,class,seg 
        all_info = [None]*3
        if self.use_global_feature and self.use_seg_feature:
            all_info[0], all_info[1], all_info[2]= f1_4, f_class, f_seg
        elif self.use_global_feature:
            all_info[0], all_info[1] = f1_4, f_class
        elif self.use_seg_feature:
            all_info[0], all_info[2] = f1_4, f_seg
        else:
            all_info[0] = f1_4
        return all_info


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels, num_channels,
                                kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.leaky_relu(self.bn1(self.conv1(X)), 0.1, inplace=True)
        Y = self.bn2(self.conv2(Y))
        X = self.conv3(X)
        Y += X
        return F.leaky_relu(Y, 0.1, inplace=True)


def main_RHAB():
    torch.backends.cudnn.benchmark = True
    feature = torch.randn(8, 64, 224, 224, requires_grad=False).cuda()
    model = DeeperResidualHybridAttentionBlock(64, use_global_feature=True).cuda()
    with torch.no_grad():
        y, class_out = model(feature)
    print(y.shape)
    print(class_out.shape)


def main_CHAN():
    torch.backends.cudnn.benchmark = True
    device = d2l.try_gpu(0)
    img = torch.randn(2, 1, 224, 224, requires_grad=False).to(device)
    model = ColorizationHAN(1, 2, use_global_feature=True, use_seg_feature=True).to(device)
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


if __name__ == '__main__':
    main_CHAN()

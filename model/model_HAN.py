from torch import nn
import torch
import torch.nn.functional as F
from math import pow


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes*ratios)
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            # print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        # return F.softmax(x/self.temperature, 1)
        return torch.sigmoid(x)


class DyConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, ratio=0.25,
                 bias=True, K=4, temperature=34, init_weight=True):
        super(DyConv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn([K, out_planes, in_planes//groups, kernel_size, kernel_size]),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn([K, out_planes]))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class DynamicDownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicDownSamplingBlock, self).__init__()
        # dynamic down-sampling branch
        self.dyconv = DyConv2d(in_channels, out_channels, 3, 2, 1)
        # self.dyconv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        # bilinear down-sampling branch
        self.bilinear_expand = nn.Conv2d(in_channels, out_channels, 1)
        # self.bilinear_down = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.bilinear_fine = nn.Conv2d(out_channels, out_channels, 1)
        # activation
        self.prelu = nn.PReLU()

    def forward(self, x):
        # dynamic down-sampling branch
        dynamic_feature = self.prelu(self.dyconv(x))
        # bilinear down-sampling branch
        bilinear_feature = self.prelu(self.bilinear_expand(x))
        # bilinear_feature = self.bilinear_down(bilinear_feature)
        bilinear_feature = F.interpolate(bilinear_feature, scale_factor=0.5, mode='bilinear', align_corners=False,
                                         recompute_scale_factor=True)
        bilinear_feature = self.prelu(self.bilinear_fine(bilinear_feature))
        # feature fusion
        y = bilinear_feature + dynamic_feature
        return y


class SpatialAttentionModule(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(SpatialAttentionModule, self).__init__()
        # high-resolution branch
        self.high_conv1 = nn.Conv2d(high_channels, 1, 1)
        self.high_conv2 = nn.Conv2d(2, 1, 1)
        # low-resolution branch
        self.low_conv1 = nn.Conv2d(low_channels, 1, 1)
        self.low_bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.low_conv2 = nn.Conv2d(2, 1, 1)
        self.low_conv3 = nn.Conv2d(low_channels, high_channels, 1)
        # activation
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, low, high):
        high_spatial_attention = self.prelu(self.high_conv1(high))
        low_spatial_attention = self.low_bilinear(self.prelu(self.low_conv1(low)))
        spatial_attention = torch.cat([high_spatial_attention, low_spatial_attention], dim=1)
        high_spatial_attention = self.sigmoid(self.high_conv2(spatial_attention))
        low_spatial_attention = self.sigmoid(self.low_conv2(spatial_attention))
        high_out = high * high_spatial_attention
        # low_out = self.low_bilinear(self.low_conv3(low)) * low_spatial_attention
        low_out = self.low_conv3(self.low_bilinear(low)) * low_spatial_attention
        y = torch.cat([high_out, low_out], dim=1)
        return y


class ChannelAttentionModule(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(ChannelAttentionModule, self).__init__()
        # high-resolution branch
        self.high_gap = nn.AdaptiveAvgPool2d(1)
        self.high_fc1 = nn.Conv2d(high_channels, high_channels//2, 1)
        self.high_fc2 = nn.Conv2d(high_channels, high_channels, 1)
        # low-resolution branch
        self.low_gap = nn.AdaptiveAvgPool2d(1)
        self.low_fc1 = nn.Conv2d(low_channels, high_channels-(high_channels//2), 1)
        self.low_fc2 = nn.Conv2d(high_channels, high_channels, 1)
        self.low_bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.low_conv3 = nn.Conv2d(low_channels, high_channels, 1)
        # activation
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, low, high):
        high_channel_attention = self.prelu(self.high_fc1(self.high_gap(high)))
        low_channel_attention = self.prelu(self.low_fc1(self.low_gap(low)))
        channel_attention = torch.cat([high_channel_attention, low_channel_attention], dim=1)
        high_channel_attention = self.sigmoid(self.high_fc2(channel_attention))
        low_channel_attention = self.sigmoid(self.low_fc2(channel_attention))
        high_out = high_channel_attention * high
        # low_out = low_channel_attention * self.low_bilinear(self.low_conv3(low))
        low_out = low_channel_attention * self.low_conv3(self.low_bilinear(low))
        y = torch.cat([high_out, low_out], dim=1)
        return y


class HybridAttentionUpsampling(nn.Module):
    def __init__(self, low_channels, high_channels):
        super(HybridAttentionUpsampling, self).__init__()
        # pre conv
        self.low_conv_1 = nn.Conv2d(low_channels, low_channels, 3, 1, 1)
        self.low_conv_2 = nn.Conv2d(low_channels, low_channels, 3, 1, 1)
        self.high_conv_1 = nn.Conv2d(high_channels, high_channels, 3, 1, 1)
        self.high_conv_2 = nn.Conv2d(high_channels, high_channels, 3, 1, 1)
        # spatial attention
        self.spatial_attention = SpatialAttentionModule(low_channels=low_channels, high_channels=high_channels)
        # channel attention
        self.channel_attention = ChannelAttentionModule(low_channels=low_channels, high_channels=high_channels)
        # attention fusion
        self.feature_fusion = nn.Conv2d(4 * high_channels, high_channels, 1)
        # activation
        self.prelu = nn.PReLU()

    def forward(self, low, high):
        low = self.low_conv_2(self.prelu(self.low_conv_1(low)))
        high = self.high_conv_2(self.prelu(self.high_conv_1(high)))
        spatial_attention = self.spatial_attention(low, high)
        channel_attention = self.channel_attention(low, high)
        y = self.feature_fusion(torch.cat([spatial_attention, channel_attention], dim=1))
        return y + high


class ResidualHybridAttentionBlock(nn.Module):
    def __init__(self, in_chanels):
        super(ResidualHybridAttentionBlock, self).__init__()
        x1_channels = in_chanels
        x2_channels = in_chanels * 2
        x4_channels = in_chanels * 4
        self.input = nn.Conv2d(in_chanels, in_chanels, 3, 1, 1)
        # 1x branch
        self.hau_1_1 = HybridAttentionUpsampling(x2_channels, x1_channels)
        self.hau_1_2 = HybridAttentionUpsampling(x2_channels, x1_channels)
        self.out = nn.Conv2d(x1_channels, x1_channels, 3, 1, 1)
        # multi-scale down-sampling from 1x to 2x
        self.msdb_1to2_1 = DynamicDownSamplingBlock(x1_channels, x2_channels)
        self.msdb_1to2_2 = DynamicDownSamplingBlock(x1_channels, x2_channels)
        # 2x branch
        self.hau_2_1 = HybridAttentionUpsampling(x4_channels, x2_channels)
        # multi-scale down-sampling from 2x to 4x
        self.msdb_2to4_1 = DynamicDownSamplingBlock(x2_channels, x4_channels)

    def forward(self, x):
        x_1_1 = self.input(x)
        x_2_1 = self.msdb_1to2_1(x_1_1)
        x_4_1 = self.msdb_2to4_1(x_2_1)

        x_1_2 = self.hau_1_1(x_2_1, x_1_1)
        x_2_2 = self.msdb_1to2_2(x_1_2) + self.hau_2_1(x_4_1, x_2_1)

        x_1_3 = self.hau_1_2(x_2_2, x_1_2)

        out = self.out(x_1_3) + x
        return out


class HybridAttentionNetwork(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=8, upscale=4):
        super(HybridAttentionNetwork, self).__init__()
        self.scale = upscale
        # shallow feature extraction
        self.shallow_feature_extraction = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1)
        )
        # residual hybrid attention blocks
        rhabs = []
        for _ in range(nb):
            rhabs.append(ResidualHybridAttentionBlock(nf))
        self.rhabs = nn.Sequential(*rhabs)

        if upscale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2)
            )

        # final feature reconstruction
        self.final_feature_reconstruction = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        #x = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        shallow_feature = self.shallow_feature_extraction(x)
        hybrid_feature = self.rhabs(shallow_feature)
        upscaled_feature = self.up(hybrid_feature)
        out = self.final_feature_reconstruction(upscaled_feature)
        return out + F.interpolate(x, scale_factor=self.scale, mode='bilinear')


if __name__ == '__main__':
    # high = torch.randn([2, 16, 64, 64])
    # low = torch.randn([2, 32, 32, 32])
    # HAN = ChannelAttentionModule(32, 16)
    # y = HAN(low, high)
    torch.backends.cudnn.benchmark = True
    img = torch.randn([1, 3, 1080//2, 1920//2], requires_grad=False).cuda()
    HAN = HybridAttentionNetwork(in_nc=3, out_nc=3, nf=64, nb=8, upscale=4).cuda()
    with torch.no_grad():
        y = HAN(img)
    print(y.shape)


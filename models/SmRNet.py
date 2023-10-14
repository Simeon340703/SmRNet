import torch
import torch.nn as nn
import math
import torch.nn.functional as F



def basicConv(in_planes: int, out_planes: int, kernel_size: int = 3, groups: int = 1,
              bias: bool = False, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        padding=(kernel_size // 2) + dilation - 1,
        groups=groups,
        bias=bias,
        dilation=dilation
    )


class Depthwise_separable_conv(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, kernels_per_layer: int = 8,
                 bias: bool = False, dilation: int = 1):
        super(Depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                                   padding=(kernel_size // 2) + dilation - 1,
                                   groups=in_planes // kernels_per_layer, bias=bias, dilation=dilation)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def conv1x1(in_planes: int, out_planes: int, kernel_size: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, bias=False)


def dwt(x, name='haar'):
    c = x[:, :, 0::2, :] / 2  # Coarse coefficient
    d = x[:, :, 1::2, :] / 2  # Detail coefficient

    x1, x2, x3, x4 = c[:, :, :, 0::2], d[:, :, :, 0::2], c[:, :, :, 1::2], d[:, :, :, 1::2]

    if name == 'haar':
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
    else:
        sqrt2, sqrt3 = math.sqrt(2), math.sqrt(3)
        mu0 = (1 + sqrt3) / (2 * sqrt2)
        mu1 = (3 + sqrt3) / (2 * sqrt2)
        mu2 = (3 - sqrt3) / (2 * sqrt2)
        mu3 = (1 - sqrt3) / (2 * sqrt2)

        x_LL = mu0 * x1 + mu1 * x2 + mu2 * x3 + mu3 * x4
        x_HL = -mu0 * x1 - mu1 * x2 + mu2 * x3 + mu3 * x4
        x_LH = -mu0 * x1 + mu1 * x2 - mu2 * x3 + mu3 * x4 #The X_HH component is discarded due to excessive noise.
    return torch.cat((x_LL, x_HL, x_LH), 1)


def idwt(x, name='haar'):
    r = 2
    out_batch, out_channel, out_height, out_width = x.size(0), int(x.size(1) / (r ** 2)), r * x.size(2), r * x.size(3)
    x1, x2, x3, x4 = (x[:, :out_channel, :, :] / 2, x[:, out_channel:out_channel * 2, :, :] / 2,
                      x[:, out_channel * 2:out_channel * 3,:, :] / 2, x[:, out_channel * 3:out_channel * 4,:, :] / 2)
    h = torch.zeros([out_batch, out_channel, out_height, out_width], dtype=torch.float32).cuda()

    if name == 'haar':
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    else:
        sqrt2, sqrt3 = math.sqrt(2), math.sqrt(3)
        mu0 = (1 + sqrt3) / (2 * sqrt2)
        mu1 = (3 + sqrt3) / (2 * sqrt2)
        mu2 = (3 - sqrt3) / (2 * sqrt2)
        mu3 = (1 - sqrt3) / (2 * sqrt2)

        h[:, :, 0::2, 0::2] = mu0 * x1 - mu1 * x2 - mu2 * x3 + mu3 * x4
        h[:, :, 1::2, 0::2] = mu0 * x1 - mu1 * x2 + mu2 * x3 - mu3 * x4
        h[:, :, 0::2, 1::2] = mu0 * x1 + mu1 * x2 - mu2 * x3 - mu3 * x4
        h[:, :, 1::2, 1::2] = mu0 * x1 + mu1 * x2 + mu2 * x3 + mu3 * x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt(x)


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return idwt(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, expansion_factor=4, bias=False):
        super(CNNBlock, self).__init__()
        expanded_channels = out_channels * expansion_factor  # Expansion factor of 4
        self.conv1 = conv1x1(in_channels, expanded_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(expanded_channels, eps=1e-4, momentum=0.95)

        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = Depthwise_separable_conv(expanded_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95)
        self.conv3 = conv1x1(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95)
        self.shortcut = nn.Sequential(
            conv1x1(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95)
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        identity = self.shortcut(identity)
        x += identity
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, expansion_factor=4, bias=False,
                 use_residual=True):
        super(BasicBlock, self).__init__()
        expanded_channels = out_channels * expansion_factor  # Expansion factor of 4
        self.conv1 = Depthwise_separable_conv(in_channels, expanded_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(expanded_channels, eps=1e-4, momentum=0.90)

        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = Depthwise_separable_conv(expanded_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.90)

        self.shortcut = nn.Sequential(
            conv1x1(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.90)
        )

    def forward(self, x):
        identity = x
        x = (self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = self.bn2(x)

        identity = self.shortcut(identity)
        x += identity
        x = self.relu(x)

        return x


def downsample_block(in_channels, out_channels, expansion_factor=4):
    layers = []
    layers = [CNNBlock(out_channels, out_channels, expansion_factor=expansion_factor),
              BasicBlock(in_channels, out_channels, expansion_factor=expansion_factor)
              ]
    return nn.Sequential(*layers)


def upsample_block(in_channels, out_channels, expansion_factor=4):
    layers = []
    layers = [BasicBlock(out_channels, out_channels, expansion_factor=expansion_factor),
              CNNBlock(in_channels, out_channels, expansion_factor=expansion_factor)
              ]
    return nn.Sequential(*layers)


def make_layer(in_channels, out_channels, num_repeats=1, expansion_factor=4):
    layers = []
    layers = nn.Sequential(*[downsample_block(in_channels, out_channels, expansion_factor=expansion_factor) for layer in
                             range(num_repeats)])
    return layers


def make_layer_inv(in_channels, out_channels, num_repeats=1, expansion_factor=4):
    layers = []

    layers = nn.Sequential(
        *[upsample_block(in_channels, out_channels, expansion_factor=expansion_factor) for layer in range(num_repeats)])
    return layers


class SmRNet(nn.Module):
    def __init__(self, downlayers, uplayers, num_classes=100, expansion_factor=1):
        super(SmRNet, self).__init__()
        self.expansion_factor = expansion_factor
        self.conv = basicConv(3, 64)
        self.bn = nn.BatchNorm2d(64)

        self.relu = nn.LeakyReLU(0.1)

        self.layer1 = make_layer(64, 64, downlayers[0], expansion_factor=expansion_factor)
        self.layer2 = make_layer(128, 128, downlayers[1], expansion_factor=expansion_factor)
        self.layer3 = make_layer(192, 192, downlayers[2], expansion_factor=expansion_factor)

        self.up1 = make_layer_inv(512, 512, uplayers[0], expansion_factor=expansion_factor)
        self.up2 = make_layer_inv(512, 512, uplayers[1], expansion_factor=expansion_factor)
        self.up3 = make_layer_inv(192, 192, uplayers[2], expansion_factor=expansion_factor)
        self.up4 = make_layer_inv(128, 128, uplayers[3], expansion_factor=expansion_factor)

        self.avg_pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)

        self.DWT = DWT()
        self.IWT = IDWT()

        self.conv0 = conv1x1(192, 128)
        self.conv1 = conv1x1(128, 64)
        self.conv2 = conv1x1(576, 512)
        self.conv2_1 = conv1x1(384, 512)
        self.conv3 = conv1x1(128, 192)
        self.conv4 = conv1x1(48, 128)
        self.conv5 = conv1x1(32, 64)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # outputs = []
        x0 = self.conv(x)
        x0 = self.bn(x0)
        x0 = self.relu(x0)

        x1 = self.conv0(self.DWT(self.layer1(x0)))
        x2 = self.DWT(self.conv1(self.layer2(x1)))
        x3 = self.conv2(self.DWT(self.layer3(x2)))

        x_ = self.conv2_1(self.IWT(self.DWT(self.up1(x3)))) + x3
        x_ = self.conv3(self.IWT(self.up2(x_))) + x2
        x_ = self.conv4(self.IWT(self.up3(x_))) + x1
        x_ = self.conv5(self.IWT(self.up4(x_))) + x0

        x_ = self.avg_pool(x_)
        x_ = self.dropout(x_)
        x_ = x_.view(x_.size(0), -1)
        x_ = self.fc(x_)

        return x_


class SmRNet_l(SmRNet):
    def __init__(self, **kwargs):
        super(SmRNet_l, self).__init__(
            downlayers=[1, 2, 2],
            uplayers=[1, 2, 1, 2],
            num_classes=100,
            expansion_factor=4)


class SmRNet_m(SmRNet):
    def __init__(self, **kwargs):
        super(SmRNet_m, self).__init__(
            downlayers=[1, 2, 2],
            uplayers=[1, 2, 1, 2],
            num_classes=100,
            expansion_factor=2)

class SmRNet_s(SmRNet):
    def __init__(self, **kwargs):
        super(SmRNet_s, self).__init__(
            downlayers=[1, 1, 1],
            uplayers=[1, 1, 1, 1],
            num_classes=100,
            expansion_factor=1)
